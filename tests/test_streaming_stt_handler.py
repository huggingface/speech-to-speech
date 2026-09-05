from __future__ import annotations

import base64
import json
from collections.abc import Callable
from queue import Queue
from threading import Event, Thread
from time import monotonic
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
import torch
from websockets.sync.server import serve

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.pipeline.control import SESSION_END
from speech_to_speech.pipeline.events import SpeechStartedEvent
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, TranscriptionFailure, VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.STT import streaming_handler
from speech_to_speech.STT.streaming_handler import (
    OpenAIRealtimeProtocol,
    OpenAIRealtimeSTTHandler,
    VLLMRealtimeProtocol,
    VLLMRealtimeSTTHandler,
    _StreamingPCMResampler,
)
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier
from tests.test_speculative_turns import _audio_bytes, _StaticVADIterator, _vad_handler_for_iterator

DIALECT_CASES = [
    (OpenAIRealtimeSTTHandler, "openai"),
    (VLLMRealtimeSTTHandler, "vllm"),
]


class _FakeSocket:
    def __init__(self, on_send: Callable[[dict[str, Any], "_FakeSocket"], None]) -> None:
        self.on_send = on_send
        self.sent: list[dict[str, Any]] = []
        self.incoming: Queue[str | Exception] = Queue()
        self.closed = False
        self.incoming.put(json.dumps({"type": "session.created", "session": {"id": "sess_fake"}}))

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
        on_send: Callable[[dict[str, Any], _FakeSocket], None],
        *,
        fail_connects: int = 0,
    ) -> None:
        self.on_send = on_send
        self.fail_connects = fail_connects
        self.instances: list[_FakeSocket] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, **kwargs: Any) -> _FakeSocket:
        self.calls.append((url, kwargs))
        if self.fail_connects:
            self.fail_connects -= 1
            raise ConnectionError("connection setup failed")
        socket = _FakeSocket(self.on_send)
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
    socket_factory: _SocketFactory,
    *,
    handler_type: type[OpenAIRealtimeSTTHandler] | type[VLLMRealtimeSTTHandler] = OpenAIRealtimeSTTHandler,
    tracker: SpeculativeTurnTracker | None = None,
    audio_sample_rate: int = 16000,
    queue_out: Queue[Any] | None = None,
    final_timeout: float = 1.0,
    connect_timeout: float = 0.5,
    base_url: str = "ws://transcription.example/v1",
    api_key: str | None = None,
    model: str = "test-model",
):
    return handler_type(
        Event(),
        queue_in=Queue(),
        queue_out=queue_out if queue_out is not None else Queue(),
        setup_kwargs={
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
            "audio_sample_rate": audio_sample_rate,
            "connect_timeout": connect_timeout,
            "final_timeout": final_timeout,
            "speculative_turns": tracker,
            "connect_factory": socket_factory,
        },
    )


def _append_events(socket: _FakeSocket) -> list[dict[str, Any]]:
    return [event for event in socket.sent if event["type"] == "input_audio_buffer.append"]


def _ack_session_update(event: dict[str, Any], socket: _FakeSocket, dialect: str = "openai") -> bool:
    if event["type"] != "session.update":
        return False
    if dialect == "openai":
        socket.incoming.put(json.dumps({"type": "session.updated"}))
    return True


def _is_final_commit(event: dict[str, Any], dialect: str = "openai") -> bool:
    return event["type"] == "input_audio_buffer.commit" and (dialect == "openai" or event.get("final") is True)


def _completion_event(
    text: str,
    *,
    item_id: str = "item_current",
    content_index: int = 0,
    dialect: str = "openai",
) -> dict[str, Any]:
    if dialect == "vllm":
        return {"type": "transcription.done", "text": text}
    return {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": item_id,
        "content_index": content_index,
        "transcript": text,
    }


def test_openai_realtime_streams_each_chunk_once_then_explicitly_commits(caplog) -> None:
    caplog.set_level("INFO")
    append_count = 0

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        nonlocal append_count
        if event["type"] == "session.update":
            socket.incoming.put(json.dumps({"type": "session.updated"}))
        elif event["type"] == "input_audio_buffer.append":
            append_count += 1
            socket.incoming.put(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "item_id": "item_1",
                        "content_index": 0,
                        "delta": "hello " if append_count == 1 else "world",
                    }
                )
            )
        elif event["type"] == "input_audio_buffer.commit":
            socket.incoming.put(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "item_id": "item_1",
                        "content_index": 0,
                        "transcript": "hello world",
                    }
                )
            )

    factory = _SocketFactory(on_send)
    handler = _handler(factory)
    first = b"\x01\x00" * 512
    second = b"\x02\x00" * 512

    handler.start_turn("turn_1", 0)
    handler.append_audio(first)
    first_partial = handler.queue_out.get(timeout=1)
    handler.append_audio(second)
    second_partial = handler.queue_out.get(timeout=1)
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
    assert [first_partial, second_partial] == [
        PartialTranscription(text="hello", turn_id="turn_1", turn_revision=0),
        PartialTranscription(text="hello world", turn_id="turn_1", turn_revision=0),
    ]
    assert "openai-realtime STT connection setup completed" in caplog.text
    assert "VAD commit to final transcript completed" in caplog.text
    handler.cleanup()


@pytest.mark.parametrize(("handler_type", "dialect"), DIALECT_CASES)
@pytest.mark.parametrize("fallback_start", [False, True])
def test_vad_speech_start_precedes_buffered_streaming_partial(
    handler_type, dialect, fallback_start, monkeypatch
) -> None:
    chunks = [torch.zeros(512) for _ in range(20)]
    vad = _vad_handler_for_iterator(
        _StaticVADIterator(
            triggered=not fallback_start,
            vad_output=chunks if fallback_start else None,
            buffer_chunks=chunks,
            active_speech_samples=12 * 512,
            last_utterance_active_speech_samples=12 * 512,
        )
    )
    delta_queued = Event()

    def delta_event(text: str) -> dict[str, Any]:
        if dialect == "vllm":
            return {"type": "transcription.delta", "delta": text}
        return {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "remote_item",
            "content_index": 0,
            "delta": text,
        }

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if _ack_session_update(event, socket, dialect):
            return
        if event["type"] == "input_audio_buffer.append":
            socket.incoming.put(json.dumps(delta_event("Hello there")))
            delta_queued.set()

    factory = _SocketFactory(on_send)
    handler = _handler(factory, handler_type=handler_type, tracker=vad.speculative_turns)
    vad.streaming_stt_sink = handler
    notifier = TranscriptionNotifier(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_kwargs={"text_output_queue": vad.text_output_queue},
    )
    service = RealtimeService(speculative_turns=vad.speculative_turns)
    connection_id = service.register()
    service._state(connection_id).runtime_config = RuntimeConfig()
    start_turn = handler.start_turn

    def start_and_publish_before_vad_resumes(turn_id, turn_revision) -> None:
        # The worker reads the first delta after append, before handling this
        # start command. Force its buffered partial through the real notifier
        # before VAD can execute the statement after its notification call.
        assert delta_queued.wait(timeout=1)
        start_turn(turn_id, turn_revision)
        partial = handler.queue_out.get(timeout=1)
        assert partial.text == "Hello there"
        list(notifier.process(partial))

    monkeypatch.setattr(handler, "start_turn", start_and_publish_before_vad_resumes)
    try:
        list(vad.process(_audio_bytes()))
        factory.instances[0].incoming.put(json.dumps(delta_event(" friend")))
        partial = handler.queue_out.get(timeout=1)
        assert partial.text == "Hello there friend"
        list(notifier.process(partial))

        pipeline_events = []
        wire_events = []
        while not vad.text_output_queue.empty():
            event = vad.text_output_queue.get_nowait()
            pipeline_events.append(event)
            wire_events.extend(service.dispatch_pipeline_event(connection_id, event))
        deltas = [event for event in wire_events if event.type == "conversation.item.input_audio_transcription.delta"]
        assert [event.delta for event in deltas] == ["Hello"]
        assert isinstance(pipeline_events[0], SpeechStartedEvent)
        speech_started = next(event for event in wire_events if event.type == "input_audio_buffer.speech_started")
        assert deltas[0].item_id == speech_started.item_id
        assert deltas[0].content_index == 0
    finally:
        service.unregister(connection_id)
        handler.cleanup()


@pytest.mark.parametrize(("handler_type", "dialect"), DIALECT_CASES)
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
            elif _is_final_commit(event, dialect):
                if dialect == "openai":
                    socket.send(
                        json.dumps(
                            {
                                "type": "conversation.item.input_audio_transcription.delta",
                                "item_id": "item_1",
                                "content_index": 0,
                                "delta": "live",
                            }
                        )
                    )
                socket.send(json.dumps(_completion_event("live final", item_id="item_1", dialect=dialect)))

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


@pytest.mark.parametrize(
    ("model", "language", "language_field"),
    [
        ("gpt-live-transcribe", "en", {"languages": ["en"]}),
        ("gpt-transcribe", "fr", {"languages": ["fr"]}),
        ("gpt-4o-transcribe", "fr", {"language": "fr"}),
    ],
)
def test_openai_realtime_session_disables_remote_turn_detection(model, language, language_field) -> None:
    protocol = OpenAIRealtimeProtocol(model=model, language=language, audio_sample_rate=24000)

    update = protocol.session_update()

    audio_input = update["session"]["audio"]["input"]
    assert update["session"]["type"] == "transcription"
    assert audio_input["format"] == {"type": "audio/pcm", "rate": 24000}
    assert audio_input["transcription"] == {"model": model, **language_field}
    assert audio_input["turn_detection"] is None


@pytest.mark.parametrize(
    ("base_url", "api_key", "expected_key"),
    [
        ("wss://api.openai.com/v1", None, "environment-key"),
        ("https://api.openai.com/v1", None, "environment-key"),
        (" https://api.openai.com/v1/ ", None, "environment-key"),
        ("https://api.openai.com/v1/realtime?intent=transcription", None, "environment-key"),
        ("https://api.openai.com/v1", "explicit-key", "explicit-key"),
        ("https://api.openai.com/v1", "", None),
        ("https://transcription.example/v1", None, None),
        ("https://api.openai.com@transcription.example/v1", None, None),
    ],
)
def test_openai_realtime_resolves_auth_after_endpoint_normalization(
    monkeypatch, base_url, api_key, expected_key
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    session_updated = Event()

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if _ack_session_update(event, socket):
            session_updated.set()

    factory = _SocketFactory(on_send)
    handler = _handler(factory, base_url=base_url, api_key=api_key)
    try:
        handler.append_audio(b"\x01\x00" * 512)
        assert session_updated.wait(timeout=1)
        endpoint, options = factory.calls[0]
        expected_headers = {"Authorization": f"Bearer {expected_key}"} if expected_key else {}
        assert options["headers"] == expected_headers
        if expected_key == "environment-key":
            assert endpoint.split("?", 1)[0] == "wss://api.openai.com/v1/realtime"
    finally:
        handler.cleanup()


def test_openai_realtime_official_url_uses_transcription_intent(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    session_updated = Event()

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if _ack_session_update(event, socket):
            session_updated.set()

    factory = _SocketFactory(on_send)
    handler = _handler(
        factory,
        base_url="wss://api.openai.com/v1",
        model="gpt-live-transcribe",
    )
    try:
        handler.append_audio(b"\x01\x00" * 512)
        assert session_updated.wait(timeout=1)
        endpoint, _options = factory.calls[0]
        split = urlsplit(endpoint)
        query = parse_qs(split.query)
        assert split.geturl().split("?", 1)[0] == "wss://api.openai.com/v1/realtime"
        assert query.get("intent") == ["transcription"]
        assert "model" not in query
        update = next(event for event in factory.instances[0].sent if event["type"] == "session.update")
        assert update["session"]["type"] == "transcription"
        assert update["session"]["audio"]["input"]["transcription"]["model"] == "gpt-live-transcribe"
    finally:
        handler.cleanup()


def test_vllm_realtime_url_omits_intent_and_model_query() -> None:
    session_started = Event()

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if event["type"] == "input_audio_buffer.commit":
            session_started.set()

    factory = _SocketFactory(on_send)
    handler = _handler(
        factory,
        handler_type=VLLMRealtimeSTTHandler,
        base_url="ws://localhost:8000/v1",
        model="mistralai/Voxtral-Mini-4B-Realtime-2602",
    )
    try:
        handler.start_turn("turn_1", 0)
        handler.append_audio(b"\x01\x00" * 512)
        assert session_started.wait(timeout=1)
        endpoint, _options = factory.calls[0]
        split = urlsplit(endpoint)
        query = parse_qs(split.query)
        assert split.geturl().split("?", 1)[0] == "ws://localhost:8000/v1/realtime"
        assert "intent" not in query
        assert "model" not in query
    finally:
        handler.cleanup()


def test_vllm_realtime_uses_start_then_final_commit_lifecycle() -> None:
    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if event["type"] == "input_audio_buffer.append":
            socket.incoming.put(json.dumps({"type": "transcription.delta", "delta": "hello"}))
        elif event == {"type": "input_audio_buffer.commit", "final": True}:
            socket.incoming.put(json.dumps({"type": "transcription.done", "text": "hello"}))

    factory = _SocketFactory(on_send)
    handler = _handler(factory, handler_type=VLLMRealtimeSTTHandler)
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


@pytest.mark.parametrize(
    "event",
    [
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "content_index": 0,
            "delta": "missing item",
        },
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "",
            "content_index": 0,
            "transcript": "empty item",
        },
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_1",
            "transcript": "missing content index",
        },
        {
            "type": "conversation.item.input_audio_transcription.failed",
            "item_id": "item_1",
        },
        {
            "type": "conversation.item.input_audio_transcription.failed",
            "item_id": "item_1",
            "content_index": True,
        },
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "item_1",
            "content_index": -1,
            "delta": "negative content index",
        },
    ],
)
def test_openai_realtime_ignores_transcription_events_without_valid_identity(event) -> None:
    protocol = OpenAIRealtimeProtocol(model="gpt-live-transcribe", language=None, audio_sample_rate=24000)

    assert protocol.parse_event(event).kind == "ignore"


@pytest.mark.parametrize("content_index", [None, True, -1, 0.5, "0"])
def test_openai_realtime_rejects_explicit_invalid_delta_content_index(content_index) -> None:
    protocol = OpenAIRealtimeProtocol(model="gpt-live-transcribe", language=None, audio_sample_rate=24000)

    assert (
        protocol.parse_event(
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": "item_1",
                "content_index": content_index,
                "delta": "invalid index",
            }
        ).kind
        == "ignore"
    )


def test_openai_realtime_delta_without_content_index_emits_live_partial() -> None:
    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if _ack_session_update(event, socket):
            return
        if event["type"] == "input_audio_buffer.append":
            for delta in ["hello", " world"]:
                socket.incoming.put(
                    json.dumps(
                        {
                            "type": "conversation.item.input_audio_transcription.delta",
                            "item_id": "item_1",
                            "delta": delta,
                        }
                    )
                )
        elif _is_final_commit(event):
            socket.incoming.put(json.dumps(_completion_event("hello world", item_id="item_1")))

    handler = _handler(_SocketFactory(on_send))
    try:
        handler.start_turn("turn_1", 0)
        handler.append_audio(b"\x01\x00" * 512)
        assert handler.queue_out.get(timeout=1).text == "hello"
        assert handler.queue_out.get(timeout=1).text == "hello world"
        assert [output.text for output in handler.process(_vad_final())] == ["hello world"]
    finally:
        handler.cleanup()


def test_openai_realtime_matches_deltas_and_completion_by_item_and_content_index() -> None:
    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if _ack_session_update(event, socket):
            return
        if event["type"] == "input_audio_buffer.append":
            for item_id, content_index, delta in [
                ("item_1", 0, "kept"),
                ("item_1", 1, " wrong content"),
                ("item_other", 0, " wrong item"),
            ]:
                socket.incoming.put(
                    json.dumps(
                        {
                            "type": "conversation.item.input_audio_transcription.delta",
                            "item_id": item_id,
                            "content_index": content_index,
                            "delta": delta,
                        }
                    )
                )
        elif _is_final_commit(event):
            socket.incoming.put(json.dumps(_completion_event("kept", item_id="item_1")))

    factory = _SocketFactory(on_send)
    handler = _handler(factory)
    handler.start_turn("turn_1", 0)
    handler.append_audio(b"\x01\x00" * 512)

    assert [output.text for output in handler.process(_vad_final())] == ["kept"]
    assert handler.queue_out.get(timeout=1).text == "kept"
    assert handler.queue_out.empty()
    handler.cleanup()


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
            socket.incoming.put(json.dumps(_completion_event(text, item_id=f"item_{text}", dialect=dialect)))

    factory = _SocketFactory(on_send)
    handler = _handler(factory, handler_type=handler_type)
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
def test_commit_to_final_metric_starts_at_the_vad_boundary(caplog, handler_type, dialect) -> None:
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
            socket.incoming.put(json.dumps(_completion_event("new", item_id="item_new", dialect=dialect)))

    factory = _SocketFactory(on_send)
    handler = _handler(factory, handler_type=handler_type)
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
    factory.instances[0].incoming.put(json.dumps(_completion_event("old", item_id="item_old", dialect=dialect)))
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
@pytest.mark.parametrize("consume_final", [False, True])
def test_commit_to_final_metric_records_receipt_even_without_consumption(
    caplog,
    monkeypatch,
    handler_type,
    dialect,
    consume_final,
) -> None:
    caplog.set_level("INFO")
    now = 100.0
    monkeypatch.setattr(streaming_handler, "perf_counter", lambda: now)
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        nonlocal now
        if _ack_session_update(event, socket, dialect):
            return
        if _is_final_commit(event, dialect):
            if not consume_final:
                tracker.observe("turn_1", 1)
            now += 0.125
            socket.incoming.put(json.dumps(_completion_event("received", dialect=dialect)))

    handler = _handler(_SocketFactory(on_send), handler_type=handler_type, tracker=tracker)
    try:
        handler.start_turn("turn_1", 0)
        handler.append_audio(b"\x01\x00" * 512)
        handler.commit_boundary("turn_1", 0)
        pending = handler._session._pending_commits[(handler._session.generation, "turn_1", 0)][0]
        assert pending.done.wait(timeout=1)
        # Model the processing delay after the provider final already arrived.
        now += 0.6
        if consume_final:
            assert [output.text for output in handler.process(_vad_final())] == ["received"]
        else:
            assert not handler.should_process_input(_vad_final())
        metrics = [record for record in caplog.records if "VAD commit to final transcript" in record.getMessage()]
        assert len(metrics) == 1
        assert "completed in 0.125s turn=turn_1 rev=0" in metrics[0].getMessage()
        assert metrics[0].threadName == handler._session._thread.name
    finally:
        handler.cleanup()


@pytest.mark.parametrize(("handler_type", "dialect"), DIALECT_CASES)
def test_late_reopen_starts_new_remote_utterance_and_combines_final_text(handler_type, dialect) -> None:
    completions = iter(["hello", "world"])

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if _ack_session_update(event, socket, dialect):
            return
        if _is_final_commit(event, dialect):
            text = next(completions)
            socket.incoming.put(json.dumps(_completion_event(text, item_id=f"item_{text}", dialect=dialect)))

    factory = _SocketFactory(on_send)
    handler = _handler(factory, handler_type=handler_type)

    handler.start_turn("turn_1", 0)
    handler.append_audio(b"\x01\x00" * 512)
    assert list(handler.process(_vad_final()))[0].text == "hello"

    handler.append_audio(b"\x02\x00" * 512)
    handler.start_turn("turn_1", 1)
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
                            "content_index": 0,
                            "transcript": "late old result",
                        }
                    )
                )
            socket.incoming.put(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "item_id": item_id,
                        "content_index": 0,
                        "transcript": f"result {commit_count}",
                    }
                )
            )

    factory = _SocketFactory(on_send)
    handler = _handler(factory)
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
        if _ack_session_update(event, socket):
            return
        if not _is_final_commit(event):
            return
        commit_count += 1
        item_id = f"item_{commit_count}"
        socket.incoming.put(json.dumps(_completion_event(f"result {commit_count}", item_id=item_id)))
        if commit_count == 1:
            socket.incoming.put(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "item_id": item_id,
                        "content_index": 0,
                        "delta": "late old text",
                    }
                )
            )

    factory = _SocketFactory(on_send)
    handler = _handler(factory)
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
                        "content_index": 0,
                        "delta": "tentative",
                    }
                )
            )
            socket.incoming.put(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "item_id": "item_1",
                        "content_index": 0,
                        "transcript": "",
                    }
                )
            )

    factory = _SocketFactory(on_send)
    handler = _handler(factory)
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
            socket.incoming.put(json.dumps(_completion_event("recovered", item_id="item_recovered", dialect=dialect)))

    factory = _SocketFactory(on_send)
    handler = _handler(factory, handler_type=handler_type)
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
def test_connection_failure_before_turn_assignment_fails_the_first_concrete_turn(handler_type, dialect) -> None:
    disconnect_once = True

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        nonlocal disconnect_once
        if _ack_session_update(event, socket, dialect):
            return
        if event["type"] == "input_audio_buffer.append" and disconnect_once:
            disconnect_once = False
            socket.incoming.put(ConnectionError("connection lost"))
        elif _is_final_commit(event, dialect):
            socket.incoming.put(json.dumps(_completion_event("recovered", item_id="item_recovered", dialect=dialect)))

    factory = _SocketFactory(on_send)
    handler = _handler(factory, handler_type=handler_type)
    lost_chunk = b"\x01\x00" * 512
    skipped_chunk = b"\x02\x00" * 512

    handler.append_audio(lost_chunk)
    handler.start_turn("turn_1", 0)
    handler.append_audio(skipped_chunk)
    failed = list(handler.process(_vad_final()))

    assert len(failed) == 1
    assert isinstance(failed[0], TranscriptionFailure)
    assert len(factory.instances) == 1
    transmitted = [base64.b64decode(event["audio"]) for event in _append_events(factory.instances[0])]
    assert transmitted == [lost_chunk]

    handler.append_audio(b"\x03\x00" * 512)
    handler.start_turn("turn_2", 0)
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
    assert len(factory.instances) == 2
    handler.cleanup()


@pytest.mark.parametrize(("handler_type", "dialect"), DIALECT_CASES)
@pytest.mark.parametrize("discard_fragment", [False, True])
def test_setup_failure_discards_reopened_revisions_of_the_same_turn(handler_type, dialect, discard_fragment) -> None:
    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if _ack_session_update(event, socket, dialect):
            return
        if _is_final_commit(event, dialect):
            socket.incoming.put(json.dumps(_completion_event("recovered", item_id="item_recovered", dialect=dialect)))

    factory = _SocketFactory(on_send, fail_connects=1)
    handler = _handler(factory, handler_type=handler_type)

    handler.start_turn("turn_1", 0)
    handler.append_audio(b"\x01\x00" * 512)
    first = list(handler.process(_vad_final()))

    if discard_fragment:
        handler.append_audio(b"\x04\x00" * 512)
        handler.discard_utterance()

    handler.append_audio(b"\x02\x00" * 512)
    handler.start_turn("turn_1", 1)
    reopened = list(handler.process(_vad_final(revision=1)))

    assert len(first) == 1 and isinstance(first[0], TranscriptionFailure)
    assert len(reopened) == 1 and isinstance(reopened[0], TranscriptionFailure)
    assert len(factory.calls) == 1
    assert factory.instances == []

    handler.append_audio(b"\x03\x00" * 512)
    handler.start_turn("turn_2", 0)
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
    assert len(factory.calls) == 2
    handler.cleanup()


@pytest.mark.parametrize(("handler_type", "dialect"), DIALECT_CASES)
def test_discard_of_failed_unassigned_audio_allows_a_new_turn(handler_type, dialect) -> None:
    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if _ack_session_update(event, socket, dialect):
            return
        if _is_final_commit(event, dialect):
            socket.incoming.put(json.dumps(_completion_event("accepted", dialect=dialect)))

    factory = _SocketFactory(on_send, fail_connects=1)
    handler = _handler(factory, handler_type=handler_type)
    accepted_chunk = b"\x02\x00" * 512
    try:
        handler.append_audio(b"\x01\x00" * 512)
        handler.discard_utterance()
        handler.append_audio(accepted_chunk)
        handler.start_turn("turn_1", 0)

        outputs = list(handler.process(_vad_final()))

        assert [output.text for output in outputs] == ["accepted"]
        assert len(factory.calls) == 2
        assert [base64.b64decode(event["audio"]) for event in _append_events(factory.instances[0])] == [accepted_chunk]
    finally:
        handler.cleanup()


@pytest.mark.parametrize(("handler_type", "dialect"), DIALECT_CASES)
def test_unconsumed_stale_commit_times_out_and_releases_later_turns(handler_type, dialect) -> None:
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    first_commit_seen = Event()
    recovered_commit_seen = Event()
    commit_count = 0
    reopened_chunk = b"\x02\x00" * 512

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        nonlocal commit_count
        if _ack_session_update(event, socket, dialect):
            return
        if _is_final_commit(event, dialect):
            commit_count += 1
            if commit_count == 1:
                first_commit_seen.set()
            else:
                socket.incoming.put(json.dumps(_completion_event("recovered", dialect=dialect)))
                recovered_commit_seen.set()

    factory = _SocketFactory(on_send)
    handler = _handler(factory, handler_type=handler_type, tracker=tracker, final_timeout=0.4)
    try:
        handler.start_turn("turn_1", 0)
        handler.append_audio(b"\x01\x00" * 512)
        handler.commit_boundary("turn_1", 0)
        assert first_commit_seen.wait(timeout=1)
        pending = handler._session._pending_commits[(handler._session.generation, "turn_1", 0)][0]
        tracker.observe("turn_1", 1)
        assert not handler.should_process_input(_vad_final())
        # Never call process() for the stale final. Later commands must still
        # run, and reopening the timed-out turn must not transcribe its suffix.
        assert not pending.done.wait(timeout=0.2)
        handler.start_turn("turn_1", 1)
        handler.append_audio(reopened_chunk)
        handler.commit_boundary("turn_1", 1)
        tracker.observe("turn_2", 0)
        handler.start_turn("turn_2", 0)
        handler.append_audio(b"\x03\x00" * 512)
        handler.commit_boundary("turn_2", 0)

        assert pending.done.wait(timeout=0.8)
        assert pending.error == "streaming transcription timed out"
        assert factory.instances[0].closed
        factory.instances[0].incoming.put(json.dumps(_completion_event("late stale final", dialect=dialect)))
        assert recovered_commit_seen.wait(timeout=1)
        reopened = list(handler.process(_vad_final(revision=1)))
        assert len(reopened) == 1
        assert isinstance(reopened[0], TranscriptionFailure)
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
        assert all(
            base64.b64decode(event["audio"]) != reopened_chunk
            for socket in factory.instances
            for event in _append_events(socket)
        )
        assert len(factory.instances) == 2
        assert handler.queue_out.empty()
    finally:
        handler.cleanup()


@pytest.mark.parametrize(("handler_type", "dialect"), DIALECT_CASES)
def test_commit_deadline_includes_time_queued_before_remote_commit(handler_type, dialect) -> None:
    append_seen = Event()
    release_append = Event()

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if _ack_session_update(event, socket, dialect):
            return
        if event["type"] == "input_audio_buffer.append":
            append_seen.set()
            assert release_append.wait(timeout=1)
        elif _is_final_commit(event, dialect):
            socket.incoming.put(json.dumps(_completion_event("too late", dialect=dialect)))

    factory = _SocketFactory(on_send)
    handler = _handler(factory, handler_type=handler_type, final_timeout=0.05)
    try:
        handler.start_turn("turn_1", 0)
        handler.append_audio(b"\x01\x00" * 512)
        assert append_seen.wait(timeout=1)
        handler.commit_boundary("turn_1", 0)
        pending = handler._session._pending_commits[(handler._session.generation, "turn_1", 0)][0]
        assert not pending.done.wait(timeout=0.1)
        release_append.set()
        result = list(handler.process(_vad_final()))
        assert len(result) == 1
        assert isinstance(result[0], TranscriptionFailure)
        assert result[0].message == "streaming transcription timed out"
        assert pending.done.wait(timeout=1)
        assert factory.instances[0].closed
        assert not any(_is_final_commit(event, dialect) for event in factory.instances[0].sent)
    finally:
        release_append.set()
        handler.cleanup()


@pytest.mark.parametrize("already_expired", [False, True])
def test_caller_deadline_does_not_wait_for_session_setup_or_close(already_expired, monkeypatch) -> None:
    setup_seen = Event()
    close_started = Event()
    release_close = Event()

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if event["type"] == "session.update" and len(factory.instances) == 1:
            # session.created arrived, but the provider never acknowledges
            # session.update. Closing this connection can also block.
            def slow_close() -> None:
                close_started.set()
                assert release_close.wait(timeout=1)
                socket.closed = True

            monkeypatch.setattr(socket, "close", slow_close)
            setup_seen.set()
        elif _ack_session_update(event, socket):
            return
        elif _is_final_commit(event):
            socket.incoming.put(json.dumps(_completion_event("recovered")))

    factory = _SocketFactory(on_send)
    handler = _handler(factory, final_timeout=0.08, connect_timeout=0.6)
    try:
        handler.start_turn("turn_1", 0)
        handler.append_audio(b"\x01\x00" * 512)
        handler.commit_boundary("turn_1", 0)
        assert setup_seen.wait(timeout=1)
        pending = handler._session._pending_commits[(handler._session.generation, "turn_1", 0)][0]
        if already_expired:
            pending.boundary_queued_at_s -= 1.0

        started = monotonic()
        result = list(handler.process(_vad_final()))
        elapsed = monotonic() - started
        assert elapsed < (0.04 if already_expired else 0.25)
        assert len(result) == 1
        assert isinstance(result[0], TranscriptionFailure)
        assert result[0].message == "streaming transcription timed out"
        assert not close_started.is_set()

        # The worker still owns expiry and cleanup. A delayed setup reply and
        # final must not turn the already timed-out request into a success.
        release_close.set()
        factory.instances[0].incoming.put(json.dumps({"type": "session.updated"}))
        factory.instances[0].incoming.put(json.dumps(_completion_event("late final")))
        assert pending.done.wait(timeout=1)
        assert pending.result is None
        assert factory.instances[0].closed
        assert handler.queue_out.empty()

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
        assert len(factory.instances) == 2
    finally:
        release_close.set()
        handler.cleanup()


@pytest.mark.parametrize(("handler_type", "dialect"), DIALECT_CASES)
@pytest.mark.parametrize("late_partial", [False, True])
def test_transcription_received_after_commit_deadline_is_rejected(
    handler_type, dialect, late_partial, monkeypatch
) -> None:
    now = 100.0
    monkeypatch.setattr(streaming_handler, "perf_counter", lambda: now)

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        nonlocal now
        if _ack_session_update(event, socket, dialect):
            return
        if _is_final_commit(event, dialect):
            # The deadline passes while the worker is waiting on the provider.
            now += 2.0
            if late_partial:
                socket.incoming.put(
                    json.dumps(
                        {"type": "transcription.delta", "delta": "late"}
                        if dialect == "vllm"
                        else {
                            "type": "conversation.item.input_audio_transcription.delta",
                            "item_id": "item_late",
                            "content_index": 0,
                            "delta": "late",
                        }
                    )
                )
            socket.incoming.put(json.dumps(_completion_event("late", item_id="item_late", dialect=dialect)))

    factory = _SocketFactory(on_send)
    handler = _handler(factory, handler_type=handler_type)
    try:
        handler.start_turn("turn_1", 0)
        handler.append_audio(b"\x01\x00" * 512)
        handler.commit_boundary("turn_1", 0)
        pending = handler._session._pending_commits[(handler._session.generation, "turn_1", 0)][0]
        result = list(handler.process(_vad_final()))
        assert len(result) == 1
        assert isinstance(result[0], TranscriptionFailure)
        assert result[0].message == "streaming transcription timed out"
        assert pending.done.wait(timeout=1)
        assert handler.queue_out.empty()
        assert factory.instances[0].closed
    finally:
        handler.cleanup()


@pytest.mark.parametrize(("handler_type", "dialect"), DIALECT_CASES)
def test_connection_failure_discards_reopened_revisions_of_the_same_turn(handler_type, dialect) -> None:
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    first_commit_seen = Event()
    commit_count = 0
    reopened_chunk = b"\x02\x00" * 512

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
            socket.incoming.put(json.dumps(_completion_event("recovered", item_id="item_recovered", dialect=dialect)))

    factory = _SocketFactory(on_send)
    handler = _handler(factory, handler_type=handler_type, tracker=tracker)
    handler.start_turn("turn_1", 0)
    handler.append_audio(b"\x01\x00" * 512)
    first_result: list[Transcription | TranscriptionFailure] = []
    first_thread = Thread(target=lambda: first_result.extend(handler.process(_vad_final())))
    first_thread.start()
    assert first_commit_seen.wait(timeout=1)

    tracker.observe("turn_1", 1)
    handler.start_turn("turn_1", 1)
    handler.append_audio(reopened_chunk)
    handler.commit_boundary("turn_1", 1)
    factory.instances[0].incoming.put(ConnectionError("connection lost"))
    first_thread.join(timeout=1)

    assert not first_thread.is_alive()
    assert len(first_result) == 1
    assert isinstance(first_result[0], TranscriptionFailure)
    reopened = list(handler.process(_vad_final(revision=1)))
    assert len(reopened) == 1
    assert isinstance(reopened[0], TranscriptionFailure)
    assert all(
        base64.b64decode(event["audio"]) != reopened_chunk
        for socket in factory.instances
        for event in _append_events(socket)
    )
    assert len(factory.instances) == 1

    tracker.observe("turn_2", 0)
    handler.start_turn("turn_2", 0)
    handler.append_audio(b"\x03\x00" * 512)
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
    assert len(factory.instances) == 2
    handler.cleanup()


@pytest.mark.parametrize(("handler_type", "dialect"), DIALECT_CASES)
def test_cancel_and_session_reuse_fence_late_results(handler_type, dialect) -> None:
    commit_seen = Event()

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if _ack_session_update(event, socket, dialect):
            return
        if _is_final_commit(event, dialect):
            commit_seen.set()

    factory = _SocketFactory(on_send)
    handler = _handler(factory, handler_type=handler_type)
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
    old_socket.incoming.put(json.dumps(_completion_event("stale", item_id="item_old", dialect=dialect)))
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
                "content_index": 0,
                "delta": "stale",
            }
            if dialect == "openai"
            else {"type": "transcription.delta", "delta": "stale"}
        )
        socket.incoming.put(json.dumps(delta))

    factory = _SocketFactory(on_send)
    handler = _handler(factory, handler_type=handler_type, queue_out=queue_out)
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
            socket.incoming.put(json.dumps(_completion_event(next(completions), dialect=dialect)))

    factory = _SocketFactory(on_send)
    handler = _handler(factory, handler_type=handler_type)
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
            socket.incoming.put(json.dumps(_completion_event("new", item_id="item_new", dialect=dialect)))

    factory = _SocketFactory(on_send)
    handler = _handler(factory, handler_type=handler_type, tracker=tracker)
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

    factory.instances[0].incoming.put(json.dumps(_completion_event("old", item_id="item_old", dialect=dialect)))
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
        if _ack_session_update(event, socket):
            return
        if not _is_final_commit(event):
            return
        commit_count += 1
        if commit_count == 1:
            socket.incoming.put(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.failed",
                        "item_id": "item_failed",
                        "content_index": 0,
                        "error": {"message": "provider detail that must not reach the client"},
                    }
                )
            )
        else:
            socket.incoming.put(json.dumps(_completion_event("recovered")))

    factory = _SocketFactory(on_send)
    handler = _handler(factory)
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


def test_openai_precommit_item_failure_discards_same_turn_revisions() -> None:
    append_count = 0
    buffers: dict[_FakeSocket, bytes] = {}
    committed_audio: list[bytes] = []

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        nonlocal append_count
        if _ack_session_update(event, socket):
            buffers[socket] = b""
            return
        if event["type"] == "input_audio_buffer.append":
            buffers[socket] += base64.b64decode(event["audio"])
            append_count += 1
            if append_count == 1:
                socket.incoming.put(
                    json.dumps(
                        {
                            "type": "conversation.item.input_audio_transcription.failed",
                            "item_id": "item_failed",
                            "content_index": 0,
                            "error": {"message": "provider detail"},
                        }
                    )
                )
        elif _is_final_commit(event):
            committed_audio.append(buffers[socket])
            buffers[socket] = b""
            socket.incoming.put(json.dumps(_completion_event("recovered", item_id="item_recovered")))

    factory = _SocketFactory(on_send)
    handler = _handler(factory)
    handler.start_turn("turn_1", 0)
    handler.append_audio(b"\x01\x00" * 512)

    first = list(handler.process(_vad_final()))
    handler.append_audio(b"\x02\x00" * 512)
    handler.start_turn("turn_1", 1)
    reopened = list(handler.process(_vad_final(revision=1)))

    assert len(first) == 1 and isinstance(first[0], TranscriptionFailure)
    assert len(reopened) == 1 and isinstance(reopened[0], TranscriptionFailure)
    assert append_count == 1

    handler.append_audio(b"\x03\x00" * 512)
    handler.start_turn("turn_2", 0)
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
    assert append_count == 2
    assert committed_audio == [b"\x03\x00" * 512]
    assert len(factory.instances) == 2
    assert factory.instances[0].closed
    handler.cleanup()


@pytest.mark.parametrize(("handler_type", "dialect"), DIALECT_CASES)
def test_discard_behind_pending_final_preserves_accepted_turns(handler_type, dialect) -> None:
    first_commit_seen = Event()
    buffers: dict[_FakeSocket, bytes] = {}
    committed_audio: list[bytes] = []

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if _ack_session_update(event, socket, dialect):
            buffers[socket] = b""
        elif event["type"] == "input_audio_buffer.append":
            buffers[socket] += base64.b64decode(event["audio"])
        elif event["type"] == "input_audio_buffer.clear":
            buffers[socket] = b""
        elif _is_final_commit(event, dialect):
            committed_audio.append(buffers[socket])
            buffers[socket] = b""
            index = len(committed_audio)
            if index == 1:
                first_commit_seen.set()
            else:
                socket.incoming.put(json.dumps(_completion_event(str(index), item_id=f"item_{index}", dialect=dialect)))

    factory = _SocketFactory(on_send)
    handler = _handler(factory, handler_type=handler_type)
    first, second, rejected, third = [bytes([value, 0]) * 512 for value in range(1, 5)]
    try:
        handler.start_turn("turn_1", 0)
        handler.append_audio(first)
        handler.commit_boundary("turn_1", 0)
        assert first_commit_seen.wait(timeout=1)

        handler.start_turn("turn_2", 0)
        handler.append_audio(second)
        handler.commit_boundary("turn_2", 0)
        handler.append_audio(rejected)
        handler.discard_utterance()
        handler.start_turn("turn_3", 0)
        handler.append_audio(third)
        handler.commit_boundary("turn_3", 0)

        # Let the worker encounter the discard while the first final is pending.
        deadline = monotonic() + 1
        while not handler._session._commands.empty():
            assert monotonic() < deadline
            Event().wait(0.001)
        factory.instances[0].incoming.put(json.dumps(_completion_event("1", item_id="item_1", dialect=dialect)))

        outputs = []
        for index in range(1, 4):
            outputs.extend(
                handler.process(
                    VADAudio(
                        audio=np.zeros(512, dtype=np.float32), mode="final", turn_id=f"turn_{index}", turn_revision=0
                    )
                )
            )

        assert all(isinstance(output, Transcription) for output in outputs)
        assert [output.text for output in outputs] == ["1", "2", "3"]
        assert committed_audio == [first, second, third]
    finally:
        handler.cleanup()


@pytest.mark.parametrize(("handler_type", "dialect"), DIALECT_CASES)
@pytest.mark.parametrize("identity_known", [False, True])
def test_discarded_audio_cannot_contaminate_the_next_turn(handler_type, dialect, identity_known) -> None:
    first_append_seen = Event()
    rejected_chunk = b"\x01\x00" * 512
    accepted_chunk = b"\x02\x00" * 512

    def on_send(event: dict[str, Any], socket: _FakeSocket) -> None:
        if _ack_session_update(event, socket, dialect):
            return
        if event["type"] == "input_audio_buffer.append" and base64.b64decode(event["audio"]) == rejected_chunk:
            if identity_known:
                socket.incoming.put(
                    json.dumps(
                        {"type": "transcription.delta", "delta": "rejected"}
                        if dialect == "vllm"
                        else {
                            "type": "conversation.item.input_audio_transcription.delta",
                            "item_id": "item_rejected",
                            "content_index": 0,
                            "delta": "rejected",
                        }
                    )
                )
            first_append_seen.set()
        elif event["type"] == "input_audio_buffer.append":
            # A delayed event from the discarded buffer arrives only after
            # the next buffer has audio. It must not acquire the next turn.
            factory.instances[0].incoming.put(
                json.dumps(
                    {"type": "transcription.delta", "delta": " stale"}
                    if dialect == "vllm"
                    else {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "item_id": "item_rejected",
                        "content_index": 0,
                        "delta": " stale",
                    }
                )
            )
        elif _is_final_commit(event, dialect):
            socket.incoming.put(json.dumps(_completion_event("accepted", dialect=dialect)))

    factory = _SocketFactory(on_send)
    handler = _handler(factory, handler_type=handler_type)
    handler.start_turn("turn_rejected", 0)
    handler.append_audio(rejected_chunk)
    assert first_append_seen.wait(timeout=1)
    if identity_known:
        assert handler.queue_out.get(timeout=1).text == "rejected"

    handler.discard_utterance()
    handler.start_turn("turn_1", 0)
    handler.append_audio(accepted_chunk)
    outputs = list(handler.process(_vad_final()))

    assert [output.text for output in outputs] == ["accepted"]
    assert handler.queue_out.empty()
    if dialect == "openai" and identity_known:
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
                        "content_index": 0,
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
                            "content_index": 0,
                            "transcript": "duplicate",
                        }
                    )
                )

    factory = _SocketFactory(on_send)
    handler = _handler(factory, tracker=tracker)
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

    factory = _SocketFactory(on_send)
    handler = _handler(factory, handler_type=VLLMRealtimeSTTHandler)
    audio = np.arange(sample_count, dtype=np.int16).tobytes()
    chunks = [audio[index : index + 1024] for index in range(0, len(audio), 1024)]
    handler.start_turn("turn_1", 0)
    for chunk in chunks:
        handler.append_audio(chunk)

    assert list(handler.process(_vad_final()))[0].text == "done"
    transmitted = b"".join(base64.b64decode(event["audio"]) for event in _append_events(factory.instances[0]))
    assert transmitted == audio
    handler.cleanup()
