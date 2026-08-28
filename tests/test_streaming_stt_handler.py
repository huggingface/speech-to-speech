from __future__ import annotations

import base64
import json
from collections.abc import Callable
from queue import Queue
from threading import Event, Thread
from time import monotonic
from typing import Any

import numpy as np
import pytest
from websockets.sync.server import serve

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.pipeline.control import SESSION_END
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, TranscriptionFailure, VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.STT.streaming_handler import (
    OpenAIRealtimeProtocol,
    OpenAIRealtimeSTTHandler,
    VLLMRealtimeProtocol,
    VLLMRealtimeSTTHandler,
    _StreamingPCMResampler,
)
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier

DIALECT_CASES = [
    (OpenAIRealtimeSTTHandler, "openai"),
    (VLLMRealtimeSTTHandler, "vllm"),
]


class _FakeSocket:
    def __init__(self, on_send: Callable[[dict[str, Any], "_FakeSocket"], None], *, dialect: str) -> None:
        self.on_send = on_send
        self.sent: list[dict[str, Any]] = []
        self.incoming: Queue[str | Exception] = Queue()
        self.closed = False
        self.incoming.put(json.dumps({"type": "session.created", "session": {"id": "sess_fake"}}))
        self.dialect = dialect

    def send(self, raw: str) -> None:
        if self.closed:
            raise RuntimeError("socket closed")
        event = json.loads(raw)
        self.sent.append(event)
        self.on_send(event, self)

    def recv(self, timeout: float | None = None) -> str:
        item = self.incoming.get(timeout=timeout)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


class _SocketFactory:
    def __init__(
        self,
        dialect: str,
        on_send: Callable[[dict[str, Any], _FakeSocket], None],
    ) -> None:
        self.dialect = dialect
        self.on_send = on_send
        self.instances: list[_FakeSocket] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, **kwargs: Any) -> _FakeSocket:
        self.calls.append((url, kwargs))
        socket = _FakeSocket(self.on_send, dialect=self.dialect)
        self.instances.append(socket)
        return socket


class _BlockingPartialQueue(Queue[Any]):
    def __init__(self) -> None:
        super().__init__()
        self.partial_put_started = Event()
        self.release_partial = Event()

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        if isinstance(item, PartialTranscription) and not self.partial_put_started.is_set():
            self.partial_put_started.set()
            assert self.release_partial.wait(timeout=1)
        super().put(item, block=block, timeout=timeout)


def _vad_final(*, revision: int = 0) -> VADAudio:
    return VADAudio(
        audio=np.zeros(160, dtype=np.float32),
        mode="final",
        turn_id="turn_1",
        turn_revision=revision,
    )


def _handler(
    handler_type: type[OpenAIRealtimeSTTHandler] | type[VLLMRealtimeSTTHandler],
    socket_factory: _SocketFactory,
    *,
    tracker: SpeculativeTurnTracker | None = None,
    audio_sample_rate: int = 16000,
    queue_out: Queue[Any] | None = None,
):
    return handler_type(
        Event(),
        queue_in=Queue(),
        queue_out=queue_out if queue_out is not None else Queue(),
        setup_kwargs={
            "base_url": "ws://transcription.example/v1",
            "model": "test-model",
            "audio_sample_rate": audio_sample_rate,
            "connect_timeout": 0.5,
            "final_timeout": 1.0,
            "speculative_turns": tracker,
            "connect_factory": socket_factory,
        },
    )


def _append_events(socket: _FakeSocket) -> list[dict[str, Any]]:
    return [event for event in socket.sent if event["type"] == "input_audio_buffer.append"]


def _ack_session_update(event: dict[str, Any], socket: _FakeSocket, dialect: str) -> bool:
    if event["type"] != "session.update":
        return False
    if dialect == "openai":
        socket.incoming.put(json.dumps({"type": "session.updated"}))
    return True


def _is_final_commit(event: dict[str, Any], dialect: str) -> bool:
    return event["type"] == "input_audio_buffer.commit" and (dialect == "openai" or event.get("final") is True)


def _completion_event(dialect: str, text: str, *, item_id: str = "item_current") -> dict[str, Any]:
    if dialect == "openai":
        return {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": item_id,
            "transcript": text,
        }
    return {"type": "transcription.done", "text": text}


def test_openai_realtime_streams_each_chunk_once_then_explicitly_commits(caplog) -> None:
    caplog.set_level("INFO")

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if event["type"] == "session.update":
            socket.incoming.put(json.dumps({"type": "session.updated"}))
        elif event["type"] == "input_audio_buffer.commit":
            socket.incoming.put(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "item_id": "item_1",
                        "delta": "hello ",
                    }
                )
            )
            socket.incoming.put(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "item_id": "item_1",
                        "delta": "world",
                    }
                )
            )
            socket.incoming.put(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "item_id": "item_1",
                        "transcript": "hello world",
                    }
                )
            )

    factory = _SocketFactory("openai", on_send)
    handler = _handler(OpenAIRealtimeSTTHandler, factory)
    first = b"\x01\x00" * 512
    second = b"\x02\x00" * 512

    handler.start_turn("turn_1", 0)
    handler.append_audio(first)
    handler.append_audio(second)
    outputs = list(handler.process(_vad_final()))

    assert outputs == [
        Transcription(
            text="hello world",
            turn_id="turn_1",
            turn_revision=0,
            speech_stopped_at_s=outputs[0].speech_stopped_at_s,
        )
    ]
    socket = factory.instances[0]
    assert [event["type"] for event in socket.sent] == [
        "session.update",
        "input_audio_buffer.append",
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
    ]
    assert [base64.b64decode(event["audio"]) for event in _append_events(socket)] == [first, second]
    assert factory.calls[0][0] == "ws://transcription.example/v1/realtime?model=test-model"
    partials = [handler.queue_out.get(timeout=1), handler.queue_out.get(timeout=1)]
    assert partials == [
        PartialTranscription(text="hello", turn_id="turn_1", turn_revision=0),
        PartialTranscription(text="hello world", turn_id="turn_1", turn_revision=0),
    ]
    assert "openai-realtime STT connection setup completed" in caplog.text
    assert "VAD commit to final transcript completed" in caplog.text
    handler.cleanup()


@pytest.mark.parametrize(
    ("handler_type", "dialect"),
    [
        (OpenAIRealtimeSTTHandler, "openai"),
        (VLLMRealtimeSTTHandler, "vllm"),
    ],
)
def test_streaming_backends_interoperate_with_fake_websocket_server(handler_type, dialect) -> None:
    received: Queue[dict[str, Any]] = Queue()

    def server_handler(socket) -> None:
        socket.send(json.dumps({"type": "session.created", "session": {"id": "sess_server"}}))
        for raw in socket:
            event = json.loads(raw)
            received.put(event)
            if event["type"] == "session.update" and dialect == "openai":
                socket.send(json.dumps({"type": "session.updated"}))
            elif event["type"] == "input_audio_buffer.append" and dialect == "vllm":
                socket.send(json.dumps({"type": "transcription.delta", "delta": "live"}))
            elif event["type"] == "input_audio_buffer.commit" and (dialect == "openai" or event.get("final") is True):
                if dialect == "openai":
                    socket.send(
                        json.dumps(
                            {
                                "type": "conversation.item.input_audio_transcription.delta",
                                "item_id": "item_1",
                                "delta": "live",
                            }
                        )
                    )
                completed = (
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "item_id": "item_1",
                        "transcript": "live final",
                    }
                    if dialect == "openai"
                    else {"type": "transcription.done", "text": "live final"}
                )
                socket.send(json.dumps(completed))

    server = serve(server_handler, "127.0.0.1", 0)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    port = server.socket.getsockname()[1]
    handler = handler_type(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_kwargs={
            "base_url": f"ws://127.0.0.1:{port}/v1",
            "model": "test-model",
            "audio_sample_rate": 24000 if dialect == "openai" else 16000,
            "connect_timeout": 1.0,
            "final_timeout": 1.0,
        },
    )

    try:
        chunk = b"\x04\x00" * 512
        handler.start_turn("turn_1", 0)
        handler.append_audio(chunk)

        outputs = list(handler.process(_vad_final()))

        assert outputs[0].text == "live final"
        assert handler.queue_out.get(timeout=1).text == "live"
        events = [received.get(timeout=1) for _ in range(4 if dialect == "vllm" else 3)]
        assert sum(event["type"] == "input_audio_buffer.append" for event in events) == 1
        transmitted_samples = sum(
            len(base64.b64decode(event["audio"])) // 2
            for event in events
            if event["type"] == "input_audio_buffer.append"
        )
        assert transmitted_samples == (768 if dialect == "openai" else 512)
    finally:
        handler.cleanup()
        server.shutdown()
        server_thread.join(timeout=1)


def test_openai_realtime_session_disables_remote_turn_detection() -> None:
    protocol = OpenAIRealtimeProtocol(model="gpt-live-transcribe", language="en", audio_sample_rate=24000)

    update = protocol.session_update()

    audio_input = update["session"]["audio"]["input"]
    assert update["session"]["type"] == "transcription"
    assert audio_input["format"] == {"type": "audio/pcm", "rate": 24000}
    assert audio_input["transcription"] == {"model": "gpt-live-transcribe", "languages": ["en"]}
    assert audio_input["turn_detection"] is None


def test_vllm_realtime_uses_start_then_final_commit_lifecycle() -> None:
    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if event["type"] == "input_audio_buffer.append":
            socket.incoming.put(json.dumps({"type": "transcription.delta", "delta": "hello"}))
        elif event == {"type": "input_audio_buffer.commit", "final": True}:
            socket.incoming.put(json.dumps({"type": "transcription.done", "text": "hello"}))

    factory = _SocketFactory("vllm", on_send)
    handler = _handler(VLLMRealtimeSTTHandler, factory)
    chunk = b"\x03\x00" * 512

    handler.start_turn("turn_1", 0)
    handler.append_audio(chunk)
    outputs = list(handler.process(_vad_final()))

    socket = factory.instances[0]
    assert socket.sent == [
        {"type": "session.update", "model": "test-model"},
        {"type": "input_audio_buffer.commit", "final": False},
        {"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode("ascii")},
        {"type": "input_audio_buffer.commit", "final": True},
    ]
    assert outputs[0].text == "hello"
    assert handler.queue_out.get(timeout=1) == PartialTranscription(
        text="hello",
        turn_id="turn_1",
        turn_revision=0,
    )
    handler.cleanup()


def test_vllm_protocol_is_separate_from_openai_wire_shape() -> None:
    protocol = VLLMRealtimeProtocol(model="Qwen/Qwen3-ASR-1.7B", language=None, audio_sample_rate=16000)

    assert protocol.session_update() == {"type": "session.update", "model": "Qwen/Qwen3-ASR-1.7B"}
    assert protocol.start_utterance() == {"type": "input_audio_buffer.commit", "final": False}
    assert protocol.finish_utterance() == {"type": "input_audio_buffer.commit", "final": True}


def test_streaming_resampler_is_continuous_across_input_chunks() -> None:
    samples = (np.sin(2 * np.pi * 997 * np.arange(4096) / 16000) * 20000).astype(np.int16)
    audio = samples.tobytes()
    chunked = _StreamingPCMResampler(16000, 24000)
    chunked_audio = (
        b"".join(chunked.push(audio[index : index + 1024]) for index in range(0, len(audio), 1024))
        + chunked.finish_utterance()
    )
    contiguous = _StreamingPCMResampler(16000, 24000)
    contiguous_audio = contiguous.push(audio) + contiguous.finish_utterance()

    assert len(chunked_audio) // 2 == 6144
    chunked_samples = np.frombuffer(chunked_audio, dtype=np.int16).astype(np.int32)
    contiguous_samples = np.frombuffer(contiguous_audio, dtype=np.int16).astype(np.int32)
    difference = chunked_samples - contiguous_samples
    assert np.max(np.abs(difference)) <= 2
    assert np.sqrt(np.mean(difference.astype(np.float64) ** 2)) < 1


@pytest.mark.parametrize(("handler_type", "dialect"), DIALECT_CASES)
def test_vad_commit_boundary_cannot_be_overtaken_by_next_turn_audio(handler_type, dialect) -> None:
    completions = iter(["old", "new"])

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if _ack_session_update(event, socket, dialect):
            return
        if _is_final_commit(event, dialect):
            text = next(completions)
            socket.incoming.put(json.dumps(_completion_event(dialect, text, item_id=f"item_{text}")))

    factory = _SocketFactory(dialect, on_send)
    handler = _handler(handler_type, factory)
    old_chunk = b"\x01\x00" * 512
    new_chunk = b"\x02\x00" * 512

    handler.start_turn("turn_1", 0)
    handler.append_audio(old_chunk)
    handler.commit_boundary("turn_1", 0)
    handler.start_turn("turn_2", 0)
    handler.append_audio(new_chunk)
    handler.commit_boundary("turn_2", 0)

    old = list(handler.process(_vad_final()))
    new = list(
        handler.process(
            VADAudio(
                audio=np.zeros(160, dtype=np.float32),
                mode="final",
                turn_id="turn_2",
                turn_revision=0,
            )
        )
    )

    assert [output.text for output in old + new] == ["old", "new"]
    socket = factory.instances[0]
    append_indexes = [index for index, event in enumerate(socket.sent) if event["type"] == "input_audio_buffer.append"]
    final_commit_indexes = [index for index, event in enumerate(socket.sent) if _is_final_commit(event, dialect)]
    assert append_indexes[0] < final_commit_indexes[0] < append_indexes[1] < final_commit_indexes[1]
    assert [base64.b64decode(event["audio"]) for event in _append_events(socket)] == [old_chunk, new_chunk]
    handler.cleanup()


@pytest.mark.parametrize(("handler_type", "dialect"), DIALECT_CASES)
def test_commit_to_final_metric_starts_at_the_vad_boundary(handler_type, dialect, caplog) -> None:
    caplog.set_level("INFO")
    first_commit_seen = Event()
    commit_count = 0

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        nonlocal commit_count
        if _ack_session_update(event, socket, dialect):
            return
        if not _is_final_commit(event, dialect):
            return
        commit_count += 1
        if commit_count == 1:
            first_commit_seen.set()
        else:
            socket.incoming.put(json.dumps(_completion_event(dialect, "new", item_id="item_new")))

    factory = _SocketFactory(dialect, on_send)
    handler = _handler(handler_type, factory)
    handler.start_turn("turn_1", 0)
    handler.append_audio(b"\x01\x00" * 512)
    handler.commit_boundary("turn_1", 0)
    first_result: list[Transcription | TranscriptionFailure] = []
    first_thread = Thread(target=lambda: first_result.extend(handler.process(_vad_final())))
    first_thread.start()
    assert first_commit_seen.wait(timeout=1)

    handler.start_turn("turn_2", 0)
    handler.append_audio(b"\x02\x00" * 512)
    handler.commit_boundary("turn_2", 0)
    Event().wait(0.12)
    factory.instances[0].incoming.put(json.dumps(_completion_event(dialect, "old", item_id="item_old")))
    first_thread.join(timeout=1)

    assert [output.text for output in first_result] == ["old"]
    second = list(
        handler.process(
            VADAudio(
                audio=np.zeros(160, dtype=np.float32),
                mode="final",
                turn_id="turn_2",
                turn_revision=0,
            )
        )
    )
    assert [output.text for output in second] == ["new"]
    metric = next(
        record.getMessage()
        for record in caplog.records
        if "VAD commit to final transcript" in record.getMessage() and "turn=turn_2" in record.getMessage()
    )
    elapsed_s = float(metric.split("completed in ", 1)[1].split("s ", 1)[0])
    assert elapsed_s >= 0.1
    handler.cleanup()


@pytest.mark.parametrize(("handler_type", "dialect"), DIALECT_CASES)
def test_late_reopen_starts_new_remote_utterance_and_combines_final_text(handler_type, dialect) -> None:
    completions = iter(["hello", "world"])

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if _ack_session_update(event, socket, dialect):
            return
        if _is_final_commit(event, dialect):
            text = next(completions)
            socket.incoming.put(json.dumps(_completion_event(dialect, text, item_id=f"item_{text}")))

    factory = _SocketFactory(dialect, on_send)
    handler = _handler(handler_type, factory)

    handler.start_turn("turn_1", 0)
    handler.append_audio(b"\x01\x00" * 512)
    assert list(handler.process(_vad_final()))[0].text == "hello"

    handler.start_turn("turn_1", 1)
    handler.append_audio(b"\x02\x00" * 512)
    reopened = list(handler.process(_vad_final(revision=1)))

    assert reopened[0].text == "hello world"
    assert reopened[0].turn_revision == 1
    assert len(_append_events(factory.instances[0])) == 2
    handler.cleanup()


def test_openai_realtime_ignores_late_completion_for_an_older_remote_item() -> None:
    commit_count = 0

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        nonlocal commit_count
        if event["type"] == "session.update":
            socket.incoming.put(json.dumps({"type": "session.updated"}))
        elif event["type"] == "input_audio_buffer.commit":
            commit_count += 1
            item_id = f"item_{commit_count}"
            socket.incoming.put(json.dumps({"type": "input_audio_buffer.committed", "item_id": item_id}))
            if commit_count == 2:
                socket.incoming.put(
                    json.dumps(
                        {
                            "type": "conversation.item.input_audio_transcription.completed",
                            "item_id": "item_1",
                            "transcript": "late old result",
                        }
                    )
                )
            socket.incoming.put(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "item_id": item_id,
                        "transcript": f"result {commit_count}",
                    }
                )
            )

    factory = _SocketFactory("openai", on_send)
    handler = _handler(OpenAIRealtimeSTTHandler, factory)
    handler.start_turn("turn_1", 0)
    handler.append_audio(b"\x01\x00" * 512)
    assert list(handler.process(_vad_final()))[0].text == "result 1"

    handler.start_turn("turn_2", 0)
    handler.append_audio(b"\x02\x00" * 512)
    second = list(
        handler.process(
            VADAudio(
                audio=np.zeros(160, dtype=np.float32),
                mode="final",
                turn_id="turn_2",
                turn_revision=0,
            )
        )
    )

    assert second[0].text == "result 2"
    handler.cleanup()


def test_openai_realtime_ignores_late_delta_from_a_completed_remote_item() -> None:
    commit_count = 0

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        nonlocal commit_count
        if _ack_session_update(event, socket, "openai"):
            return
        if not _is_final_commit(event, "openai"):
            return
        commit_count += 1
        item_id = f"item_{commit_count}"
        socket.incoming.put(json.dumps(_completion_event("openai", f"result {commit_count}", item_id=item_id)))
        if commit_count == 1:
            socket.incoming.put(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "item_id": item_id,
                        "delta": "late old text",
                    }
                )
            )

    factory = _SocketFactory("openai", on_send)
    handler = _handler(OpenAIRealtimeSTTHandler, factory)
    handler.start_turn("turn_1", 0)
    handler.append_audio(b"\x01\x00" * 512)
    assert [output.text for output in handler.process(_vad_final())] == ["result 1"]

    handler.start_turn("turn_2", 0)
    handler.append_audio(b"\x02\x00" * 512)
    second = list(
        handler.process(
            VADAudio(
                audio=np.zeros(160, dtype=np.float32),
                mode="final",
                turn_id="turn_2",
                turn_revision=0,
            )
        )
    )

    assert [output.text for output in second] == ["result 2"]
    assert handler.queue_out.empty()
    handler.cleanup()


def test_empty_final_is_authoritative_over_an_earlier_partial() -> None:
    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if event["type"] == "session.update":
            socket.incoming.put(json.dumps({"type": "session.updated"}))
        elif event["type"] == "input_audio_buffer.commit":
            socket.incoming.put(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "item_id": "item_1",
                        "delta": "tentative",
                    }
                )
            )
            socket.incoming.put(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "item_id": "item_1",
                        "transcript": "",
                    }
                )
            )

    factory = _SocketFactory("openai", on_send)
    handler = _handler(OpenAIRealtimeSTTHandler, factory)
    handler.start_turn("turn_1", 0)
    handler.append_audio(b"\x01\x00" * 512)

    assert list(handler.process(_vad_final()))[0].text == ""
    assert handler.queue_out.get(timeout=1).text == "tentative"
    handler.cleanup()


@pytest.mark.parametrize(("handler_type", "dialect"), DIALECT_CASES)
def test_connection_failure_is_not_replayed_and_only_fails_affected_turn(handler_type, dialect) -> None:
    disconnect_once = True

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        nonlocal disconnect_once
        if _ack_session_update(event, socket, dialect):
            return
        if event["type"] == "input_audio_buffer.append" and disconnect_once:
            disconnect_once = False
            socket.incoming.put(ConnectionError("connection lost"))
        elif _is_final_commit(event, dialect):
            socket.incoming.put(json.dumps(_completion_event(dialect, "recovered", item_id="item_recovered")))

    factory = _SocketFactory(dialect, on_send)
    handler = _handler(handler_type, factory)
    lost_chunk = b"\x01\x00" * 512

    handler.start_turn("turn_1", 0)
    handler.append_audio(lost_chunk)
    failed = list(handler.process(_vad_final()))

    assert len(failed) == 1
    assert isinstance(failed[0], TranscriptionFailure)
    assert len(factory.instances) == 1
    assert [base64.b64decode(event["audio"]) for event in _append_events(factory.instances[0])] == [lost_chunk]

    handler.start_turn("turn_2", 0)
    handler.append_audio(b"\x02\x00" * 512)
    recovered = list(
        handler.process(
            VADAudio(
                audio=np.zeros(160, dtype=np.float32),
                mode="final",
                turn_id="turn_2",
                turn_revision=0,
            )
        )
    )

    assert recovered[0].text == "recovered"
    assert len(factory.instances) == 2
    assert all(base64.b64decode(event["audio"]) != lost_chunk for event in _append_events(factory.instances[1]))
    handler.cleanup()


@pytest.mark.parametrize(("handler_type", "dialect"), DIALECT_CASES)
def test_cancel_and_session_reuse_fence_late_results(handler_type, dialect) -> None:
    commit_seen = Event()

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if _ack_session_update(event, socket, dialect):
            return
        if _is_final_commit(event, dialect):
            commit_seen.set()

    factory = _SocketFactory(dialect, on_send)
    handler = _handler(handler_type, factory)
    handler.start_turn("turn_1", 0)
    handler.append_audio(b"\x01\x00" * 512)

    result: list[Transcription | TranscriptionFailure] = []

    def finish_turn() -> None:
        result.extend(handler.process(_vad_final()))

    thread = Thread(target=finish_turn)
    thread.start()
    assert commit_seen.wait(timeout=1)
    old_socket = factory.instances[0]
    handler.cancel_session()
    old_socket.incoming.put(json.dumps(_completion_event(dialect, "stale", item_id="item_old")))
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert result == []
    assert handler.queue_out.empty()
    assert old_socket.closed

    handler.start_turn("turn_1", 0)
    handler.append_audio(b"\x02\x00" * 512)
    handler.on_session_end()
    assert handler.queue_out.empty()
    handler.cleanup()


@pytest.mark.parametrize(("handler_type", "dialect"), DIALECT_CASES)
def test_partial_publication_finishes_before_session_end_can_cross_the_barrier(handler_type, dialect) -> None:
    queue_out = _BlockingPartialQueue()

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if _ack_session_update(event, socket, dialect):
            return
        if not _is_final_commit(event, dialect):
            return
        delta = (
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": "item_old",
                "delta": "stale",
            }
            if dialect == "openai"
            else {"type": "transcription.delta", "delta": "stale"}
        )
        socket.incoming.put(json.dumps(delta))

    factory = _SocketFactory(dialect, on_send)
    handler = _handler(handler_type, factory, queue_out=queue_out)
    handler.start_turn("turn_1", 0)
    handler.append_audio(b"\x01\x00" * 512)
    final_result: list[Transcription | TranscriptionFailure] = []
    final_thread = Thread(target=lambda: final_result.extend(handler.process(_vad_final())))
    final_thread.start()
    assert queue_out.partial_put_started.wait(timeout=1)

    cancel_done = Event()

    def cancel() -> None:
        handler.cancel_session()
        cancel_done.set()

    cancel_thread = Thread(target=cancel)
    cancel_thread.start()
    deadline = monotonic() + 1
    while handler._session.generation == 0 and monotonic() < deadline:
        Event().wait(0.001)
    assert handler._session.generation == 1
    assert not cancel_done.is_set()

    queue_out.release_partial.set()
    cancel_thread.join(timeout=1)
    final_thread.join(timeout=1)
    assert cancel_done.is_set()
    assert final_result == []
    queue_out.put(SESSION_END)

    assert isinstance(queue_out.get_nowait(), PartialTranscription)
    assert queue_out.get_nowait() == SESSION_END
    assert queue_out.empty()
    handler.cleanup()


@pytest.mark.parametrize(("handler_type", "dialect"), DIALECT_CASES)
def test_session_end_reconnects_and_clears_prior_turn_prefixes(handler_type, dialect) -> None:
    completions = iter(["first", "second"])

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if _ack_session_update(event, socket, dialect):
            return
        if _is_final_commit(event, dialect):
            socket.incoming.put(json.dumps(_completion_event(dialect, next(completions))))

    factory = _SocketFactory(dialect, on_send)
    handler = _handler(handler_type, factory)
    handler.start_turn("turn_1", 0)
    handler.append_audio(b"\x01\x00" * 512)
    assert list(handler.process(_vad_final()))[0].text == "first"

    handler.on_session_end()
    handler.start_turn("turn_1", 1)
    handler.append_audio(b"\x02\x00" * 512)
    second = list(handler.process(_vad_final(revision=1)))

    assert second[0].text == "second"
    assert len(factory.instances) == 2
    assert factory.instances[0].closed
    handler.cleanup()


@pytest.mark.parametrize(("handler_type", "dialect"), DIALECT_CASES)
def test_reopen_audio_waits_for_the_active_revision_to_finish(handler_type, dialect) -> None:
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    first_commit_seen = Event()
    second_append_seen = Event()
    second_chunk = b"\x02\x00" * 512
    commit_count = 0

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        nonlocal commit_count
        if _ack_session_update(event, socket, dialect):
            return
        if event["type"] == "input_audio_buffer.append":
            if base64.b64decode(event["audio"]) == second_chunk:
                second_append_seen.set()
            return
        if not _is_final_commit(event, dialect):
            return
        commit_count += 1
        if commit_count == 1:
            first_commit_seen.set()
        else:
            socket.incoming.put(json.dumps(_completion_event(dialect, "new", item_id="item_new")))

    factory = _SocketFactory(dialect, on_send)
    handler = _handler(handler_type, factory, tracker=tracker)
    handler.start_turn("turn_1", 0)
    handler.append_audio(b"\x01\x00" * 512)
    first_result: list[Transcription | TranscriptionFailure] = []

    first_thread = Thread(target=lambda: first_result.extend(handler.process(_vad_final())))
    first_thread.start()
    assert first_commit_seen.wait(timeout=1)

    tracker.observe("turn_1", 1)
    handler.start_turn("turn_1", 1)
    handler.append_audio(second_chunk)
    assert not second_append_seen.wait(timeout=0.1)

    factory.instances[0].incoming.put(json.dumps(_completion_event(dialect, "old", item_id="item_old")))
    first_thread.join(timeout=1)

    assert not first_thread.is_alive()
    assert first_result == []
    reopened = list(handler.process(_vad_final(revision=1)))
    assert second_append_seen.is_set()
    assert [output.text for output in reopened] == ["old new"]
    transmitted = [base64.b64decode(event["audio"]) for socket in factory.instances for event in _append_events(socket)]
    assert transmitted.count(second_chunk) == 1
    handler.cleanup()


def test_openai_item_scoped_failure_returns_prompt_sanitized_failure_and_keeps_connection() -> None:
    commit_count = 0

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        nonlocal commit_count
        if _ack_session_update(event, socket, "openai"):
            return
        if not _is_final_commit(event, "openai"):
            return
        commit_count += 1
        if commit_count == 1:
            socket.incoming.put(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.failed",
                        "item_id": "item_failed",
                        "error": {"message": "provider detail that must not reach the client"},
                    }
                )
            )
        else:
            socket.incoming.put(json.dumps(_completion_event("openai", "recovered")))

    factory = _SocketFactory("openai", on_send)
    handler = _handler(OpenAIRealtimeSTTHandler, factory)
    handler.start_turn("turn_1", 0)
    handler.append_audio(b"\x01\x00" * 512)

    started_at = monotonic()
    failed = list(handler.process(_vad_final()))

    assert monotonic() - started_at < 0.5
    assert len(failed) == 1
    assert isinstance(failed[0], TranscriptionFailure)
    assert failed[0].message == "remote streaming transcription failed"

    handler.start_turn("turn_2", 0)
    handler.append_audio(b"\x02\x00" * 512)
    recovered = list(
        handler.process(
            VADAudio(
                audio=np.zeros(160, dtype=np.float32),
                mode="final",
                turn_id="turn_2",
                turn_revision=0,
            )
        )
    )
    assert [output.text for output in recovered] == ["recovered"]
    assert len(factory.instances) == 1
    handler.cleanup()


@pytest.mark.parametrize(("handler_type", "dialect"), DIALECT_CASES)
def test_discarded_audio_cannot_contaminate_the_next_turn(handler_type, dialect) -> None:
    first_append_seen = Event()
    rejected_chunk = b"\x01\x00" * 512
    accepted_chunk = b"\x02\x00" * 512

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if _ack_session_update(event, socket, dialect):
            return
        if event["type"] == "input_audio_buffer.append" and base64.b64decode(event["audio"]) == rejected_chunk:
            first_append_seen.set()
        elif _is_final_commit(event, dialect):
            socket.incoming.put(json.dumps(_completion_event(dialect, "accepted")))

    factory = _SocketFactory(dialect, on_send)
    handler = _handler(handler_type, factory)
    handler.append_audio(rejected_chunk)
    assert first_append_seen.wait(timeout=1)

    handler.discard_utterance()
    handler.start_turn("turn_1", 0)
    handler.append_audio(accepted_chunk)
    outputs = list(handler.process(_vad_final()))

    assert [output.text for output in outputs] == ["accepted"]
    if dialect == "openai":
        events = factory.instances[0].sent
        clear_index = events.index({"type": "input_audio_buffer.clear"})
        accepted_index = next(
            index
            for index, event in enumerate(events)
            if event["type"] == "input_audio_buffer.append" and base64.b64decode(event["audio"]) == accepted_chunk
        )
        assert clear_index < accepted_index
        assert len(factory.instances) == 1
    else:
        assert len(factory.instances) == 2
        assert factory.instances[0].closed
        assert [base64.b64decode(event["audio"]) for event in _append_events(factory.instances[1])] == [accepted_chunk]
    handler.cleanup()


def test_stale_completed_revision_yields_exactly_one_llm_request_for_latest_revision() -> None:
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    commit_count = 0

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        nonlocal commit_count
        if event["type"] == "session.update":
            socket.incoming.put(json.dumps({"type": "session.updated"}))
        elif event["type"] == "input_audio_buffer.commit":
            commit_count += 1
            if commit_count == 1:
                tracker.observe("turn_1", 1)
            socket.incoming.put(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "item_id": f"item_{commit_count}",
                        "transcript": "old" if commit_count == 1 else "latest",
                    }
                )
            )
            if commit_count == 1:
                socket.incoming.put(
                    json.dumps(
                        {
                            "type": "conversation.item.input_audio_transcription.completed",
                            "item_id": "item_1",
                            "transcript": "duplicate",
                        }
                    )
                )

    factory = _SocketFactory("openai", on_send)
    handler = _handler(OpenAIRealtimeSTTHandler, factory, tracker=tracker)
    handler.start_turn("turn_1", 0)
    handler.append_audio(b"\x01\x00" * 512)

    assert list(handler.process(_vad_final())) == []
    assert handler.queue_out.empty()

    handler.start_turn("turn_1", 1)
    handler.append_audio(b"\x02\x00" * 512)
    latest = list(handler.process(_vad_final(revision=1)))
    assert [output.text for output in latest] == ["old latest"]

    transcription_events = Queue()
    notifier = TranscriptionNotifier(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_kwargs={"text_output_queue": transcription_events},
    )
    list(notifier.process(latest[0]))
    llm_requests = Queue()
    service = RealtimeService(text_prompt_queue=llm_requests, speculative_turns=tracker)
    connection_id = service.register()
    service._state(connection_id).runtime_config = RuntimeConfig()
    service.dispatch_pipeline_event(connection_id, transcription_events.get_nowait())

    request = llm_requests.get_nowait()
    assert (request.turn_id, request.turn_revision) == ("turn_1", 1)
    assert llm_requests.empty()
    service.unregister(connection_id)
    handler.cleanup()


@pytest.mark.parametrize("sample_count", [16000 * 4 + 8000, 16000 * 5, 16000 * 5 + 8000])
def test_vllm_boundary_audio_is_sent_once_without_window_reupload(sample_count: int) -> None:
    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if event == {"type": "input_audio_buffer.commit", "final": True}:
            socket.incoming.put(json.dumps({"type": "transcription.done", "text": "done"}))

    factory = _SocketFactory("vllm", on_send)
    handler = _handler(VLLMRealtimeSTTHandler, factory)
    audio = np.arange(sample_count, dtype=np.int16).tobytes()
    chunks = [audio[index : index + 1024] for index in range(0, len(audio), 1024)]
    handler.start_turn("turn_1", 0)
    for chunk in chunks:
        handler.append_audio(chunk)

    assert list(handler.process(_vad_final()))[0].text == "done"
    transmitted = b"".join(base64.b64decode(event["audio"]) for event in _append_events(factory.instances[0]))
    assert transmitted == audio
    handler.cleanup()
