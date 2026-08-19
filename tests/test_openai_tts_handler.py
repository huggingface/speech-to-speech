from __future__ import annotations

import io
import socket
from queue import Queue
from threading import Event, Thread
from time import perf_counter

import numpy as np
import pytest
from scipy.io import wavfile

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.control import SESSION_END
from speech_to_speech.pipeline.events import ResponseFailedEvent
from speech_to_speech.pipeline.messages import (
    AUDIO_RESPONSE_DONE,
    PIPELINE_END,
    AudioOutput,
    EndOfResponse,
    TTSInput,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.TTS import openai_compatible_handler as tts_module
from speech_to_speech.TTS.openai_compatible_handler import (
    HttpSpeechOperation,
    OpenAICompatibleTTSHandler,
    SpeechRequestCancelled,
)


class _FakeSpeechOperation:
    instances: list["_FakeSpeechOperation"] = []
    startup_action = None
    startup_error: Exception | None = None
    failure_after_chunks: int | None = None
    response_bytes: bytes | None = None

    def __init__(self, **kwargs) -> None:
        self.payload = kwargs["payload"]
        self.cancelled = False
        type(self).instances.append(self)

    def iter_bytes(self, cancel_check):
        if type(self).startup_action is not None:
            type(self).startup_action()
        if type(self).startup_error is not None:
            raise type(self).startup_error
        encoded = type(self).response_bytes
        if encoded is None:
            samples = np.arange(2400, dtype="<i2")
            encoded = samples.tobytes()
        for index, offset in enumerate(range(0, len(encoded), 301)):
            if self.cancelled:
                raise SpeechRequestCancelled
            if type(self).failure_after_chunks == index:
                raise tts_module.SpeechRequestError("speech stream failed")
            if cancel_check():
                self.cancel()
                return
            yield encoded[offset : offset + 301]

    def cancel(self) -> None:
        self.cancelled = True


class _CountingSpeculativeTurnTracker(SpeculativeTurnTracker):
    def __init__(self) -> None:
        super().__init__()
        self.commit_calls = 0

    def commit_if_latest_after_reopen_grace(self, turn_id: str | None, revision: int | None) -> bool:
        self.commit_calls += 1
        return super().commit_if_latest_after_reopen_grace(turn_id, revision)


def _reset_fake_speech_operation() -> None:
    _FakeSpeechOperation.instances.clear()
    _FakeSpeechOperation.startup_action = None
    _FakeSpeechOperation.startup_error = None
    _FakeSpeechOperation.failure_after_chunks = None
    _FakeSpeechOperation.response_bytes = None


def _openai_tts_handler(
    monkeypatch,
    *,
    cancel_scope: CancelScope | None = None,
    speculative_turns: SpeculativeTurnTracker | None = None,
) -> OpenAICompatibleTTSHandler:
    _reset_fake_speech_operation()
    monkeypatch.setattr(tts_module, "HttpSpeechOperation", _FakeSpeechOperation)
    return OpenAICompatibleTTSHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_args=(Event(),),
        setup_kwargs={
            "sample_rate": 24000,
            "blocksize": 512,
            "cancel_scope": cancel_scope,
            "speculative_turns": speculative_turns,
        },
    )


def test_openai_tts_streams_resampled_fixed_size_pcm(monkeypatch):
    handler = _openai_tts_handler(monkeypatch, cancel_scope=CancelScope())
    handler.model = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    handler.voice = "aiden"
    handler.language = "Auto"

    chunks = list(handler.process(TTSInput(text="Hello", language_code="en")))

    assert chunks
    assert all(isinstance(chunk, np.ndarray) for chunk in chunks)
    assert all(chunk.dtype == np.int16 and chunk.shape == (512,) for chunk in chunks)
    assert sum(chunk.size for chunk in chunks) == 2048
    payload = _FakeSpeechOperation.instances[0].payload
    assert payload["input"] == "Hello"
    assert payload["voice"] == "aiden"
    assert payload["language"] == "English"
    assert payload["stream"] is True
    assert payload["stream_format"] == "audio"


def test_openai_tts_http_failure_before_audio_does_not_commit(monkeypatch):
    tracker = _CountingSpeculativeTurnTracker()
    tracker.observe("turn-1", 0)
    handler = _openai_tts_handler(monkeypatch, speculative_turns=tracker)
    _FakeSpeechOperation.startup_error = tts_module.SpeechRequestError("speech server returned HTTP 500")

    chunks = list(
        handler.process(
            TTSInput(
                text="Hello",
                turn_id="turn-1",
                turn_revision=0,
                response_key="response-1",
            )
        )
    )

    assert chunks == []
    assert tracker.commit_calls == 0
    assert not tracker.is_committed("turn-1", 0)
    failure = handler.queue_out.get_nowait()
    assert isinstance(failure, ResponseFailedEvent)
    assert failure.message == "speech server returned HTTP 500"
    assert failure.turn_id == "turn-1"
    assert failure.turn_revision == 0
    assert failure.response_key == "response-1"


def test_openai_tts_stale_keyed_terminal_becomes_cleanup(monkeypatch):
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn-1", 0)
    handler = _openai_tts_handler(monkeypatch, speculative_turns=tracker)
    terminal = EndOfResponse(
        response_key="response-1",
        turn_id="turn-1",
        turn_revision=0,
        cancel_generation=7,
    )
    tracker.observe("turn-1", 1)

    outputs = list(handler.process(terminal))
    queued = handler.output_for_queue(outputs[0], terminal)

    assert outputs == [AUDIO_RESPONSE_DONE]
    assert terminal.cleanup_only is True
    assert isinstance(queued, AudioOutput)
    assert queued.response_key == "response-1"
    assert queued.cancel_generation == 7
    assert queued.cleanup_only is True


def test_openai_tts_failure_after_audio_emits_response_failure(monkeypatch):
    handler = _openai_tts_handler(monkeypatch)
    _FakeSpeechOperation.failure_after_chunks = 8
    generation = handler.process(TTSInput(text="Hello", turn_id="turn-1", turn_revision=0))

    first_audio = next(generation)
    remaining = list(generation)

    assert isinstance(first_audio, np.ndarray)
    assert remaining == []
    failure = handler.queue_out.get_nowait()
    assert isinstance(failure, ResponseFailedEvent)
    assert failure.message == "speech stream failed"
    assert failure.turn_id == "turn-1"
    assert failure.turn_revision == 0


def test_openai_tts_serializes_failure_after_emitted_audio(monkeypatch):
    handler = _openai_tts_handler(monkeypatch)
    _FakeSpeechOperation.failure_after_chunks = 8
    handler.queue_in.put(
        TTSInput(
            text="Hello",
            turn_id="turn-1",
            turn_revision=0,
            cancel_generation=3,
            response_key="response-1",
        )
    )
    handler.queue_in.put(
        EndOfResponse(
            turn_id="turn-1",
            turn_revision=0,
            cancel_generation=3,
            response_key="response-1",
        )
    )
    handler.queue_in.put(PIPELINE_END)

    thread = Thread(target=handler.run)
    thread.start()
    thread.join(timeout=1)

    assert not thread.is_alive()
    outputs = []
    while not handler.queue_out.empty():
        outputs.append(handler.queue_out.get_nowait())
    audio_indexes = [
        index
        for index, output in enumerate(outputs)
        if isinstance(output, AudioOutput) and isinstance(output.audio, np.ndarray)
    ]
    failure_index = next(index for index, output in enumerate(outputs) if isinstance(output, ResponseFailedEvent))
    done_index = next(
        index
        for index, output in enumerate(outputs)
        if isinstance(output, AudioOutput) and isinstance(output.audio, bytes) and output.audio == AUDIO_RESPONSE_DONE
    )

    assert audio_indexes
    assert max(audio_indexes) < failure_index < done_index


def test_openai_tts_suppresses_remaining_chunks_after_failure_until_response_end(monkeypatch):
    handler = _openai_tts_handler(monkeypatch)
    _FakeSpeechOperation.startup_error = tts_module.SpeechRequestError("speech server returned HTTP 500")
    tts_input = TTSInput(
        text="Hello",
        turn_id="turn-1",
        turn_revision=0,
        cancel_generation=3,
    )

    assert list(handler.process(tts_input)) == []
    assert len(_FakeSpeechOperation.instances) == 1
    assert list(handler.process(tts_input)) == []
    assert len(_FakeSpeechOperation.instances) == 1

    assert list(
        handler.process(
            EndOfResponse(
                turn_id="turn-1",
                turn_revision=0,
                cancel_generation=3,
            )
        )
    ) == [AUDIO_RESPONSE_DONE]
    _FakeSpeechOperation.startup_error = None
    assert list(handler.process(tts_input))
    assert len(_FakeSpeechOperation.instances) == 2


def test_openai_tts_failure_does_not_suppress_another_response(monkeypatch):
    handler = _openai_tts_handler(monkeypatch)
    _FakeSpeechOperation.startup_error = tts_module.SpeechRequestError("speech server returned HTTP 500")

    assert (
        list(
            handler.process(
                TTSInput(
                    text="First",
                    turn_id="turn-1",
                    turn_revision=0,
                    cancel_generation=3,
                    response_key="response-1",
                )
            )
        )
        == []
    )
    _FakeSpeechOperation.startup_error = None

    chunks = list(
        handler.process(
            TTSInput(
                text="Second",
                turn_id="turn-1",
                turn_revision=0,
                cancel_generation=3,
                response_key="response-2",
            )
        )
    )

    assert chunks
    assert len(_FakeSpeechOperation.instances) == 2
    assert _FakeSpeechOperation.instances[1].payload["input"] == "Second"


def test_openai_tts_does_not_merge_queued_responses(monkeypatch):
    handler = _openai_tts_handler(monkeypatch)
    second = TTSInput(
        text="Second",
        turn_id="turn-1",
        turn_revision=0,
        cancel_generation=3,
        response_key="response-2",
    )
    handler.queue_in.put(second)

    first_chunks = list(
        handler.process(
            TTSInput(
                text="First",
                turn_id="turn-1",
                turn_revision=0,
                cancel_generation=3,
                response_key="response-1",
            )
        )
    )

    assert first_chunks
    assert _FakeSpeechOperation.instances[0].payload["input"] == "First"
    assert handler.queue_in.get_nowait() is second


def test_openai_tts_empty_success_response_emits_failure(monkeypatch):
    handler = _openai_tts_handler(monkeypatch)
    _FakeSpeechOperation.response_bytes = b""

    assert list(handler.process(TTSInput(text="Hello", response_key="response-1"))) == []

    failure = handler.queue_out.get_nowait()
    assert isinstance(failure, ResponseFailedEvent)
    assert failure.message == "speech endpoint returned no audio"
    assert failure.response_key == "response-1"


def test_openai_tts_decodes_non_streaming_wav(monkeypatch):
    handler = _openai_tts_handler(monkeypatch)
    handler.response_format = "wav"
    handler.stream = False
    encoded = io.BytesIO()
    wavfile.write(encoded, 24000, np.arange(2400, dtype=np.int16))
    _FakeSpeechOperation.response_bytes = encoded.getvalue()

    chunks = list(handler.process(TTSInput(text="Hello")))

    assert chunks
    assert all(chunk.dtype == np.int16 and chunk.shape == (512,) for chunk in chunks)
    assert sum(chunk.size for chunk in chunks) == 2048
    payload = _FakeSpeechOperation.instances[0].payload
    assert payload["response_format"] == "wav"
    assert "stream" not in payload


def test_openai_tts_cancellation_before_audio_does_not_commit(monkeypatch):
    tracker = _CountingSpeculativeTurnTracker()
    tracker.observe("turn-1", 0)
    cancel_scope = CancelScope()
    handler = _openai_tts_handler(
        monkeypatch,
        cancel_scope=cancel_scope,
        speculative_turns=tracker,
    )
    _FakeSpeechOperation.startup_action = cancel_scope.cancel

    chunks = list(
        handler.process(
            TTSInput(
                text="Hello",
                turn_id="turn-1",
                turn_revision=0,
                cancel_generation=cancel_scope.generation,
            )
        )
    )

    assert chunks == []
    assert tracker.commit_calls == 0
    assert not tracker.is_committed("turn-1", 0)
    assert _FakeSpeechOperation.instances[0].cancelled is True


def test_openai_tts_first_emitted_audio_commits_exactly_once(monkeypatch):
    tracker = _CountingSpeculativeTurnTracker()
    tracker.observe("turn-1", 0)
    handler = _openai_tts_handler(monkeypatch, speculative_turns=tracker)

    chunks = list(
        handler.process(
            TTSInput(
                text="Hello",
                turn_id="turn-1",
                turn_revision=0,
            )
        )
    )

    assert chunks
    assert tracker.commit_calls == 1
    assert tracker.is_committed("turn-1", 0)


def test_openai_tts_reopened_during_startup_suppresses_old_revision(monkeypatch):
    tracker = _CountingSpeculativeTurnTracker()
    tracker.observe("turn-1", 0)
    handler = _openai_tts_handler(monkeypatch, speculative_turns=tracker)
    _FakeSpeechOperation.startup_action = lambda: tracker.observe("turn-1", 1)

    chunks = list(
        handler.process(
            TTSInput(
                text="Hello",
                turn_id="turn-1",
                turn_revision=0,
            )
        )
    )

    assert chunks == []
    assert tracker.commit_calls == 0
    assert not tracker.is_committed("turn-1", 0)
    assert _FakeSpeechOperation.instances[0].cancelled is True


def test_openai_tts_cancellation_stops_publication_and_closes_operation(monkeypatch):
    cancel_scope = CancelScope()
    handler = _openai_tts_handler(monkeypatch, cancel_scope=cancel_scope)

    generation = handler.process(TTSInput(text="Hello", cancel_generation=cancel_scope.generation))
    first = next(generation)
    cancel_scope.cancel()

    assert isinstance(first, np.ndarray)
    assert list(generation) == []
    assert _FakeSpeechOperation.instances[0].cancelled is True


def test_openai_tts_session_teardown_stops_publication_and_closes_operation(monkeypatch):
    handler = _openai_tts_handler(monkeypatch)
    generation = handler.process(TTSInput(text="Hello"))
    first = next(generation)
    operation = _FakeSpeechOperation.instances[0]

    handler.on_session_end()

    assert isinstance(first, np.ndarray)
    assert list(generation) == []
    assert operation.cancelled is True
    assert handler._active_operation is None
    assert handler.queue_out.empty()


def test_http_speech_cancellation_after_completion_is_harmless(monkeypatch):
    transport = tts_module.httpx.MockTransport(
        lambda request: tts_module.httpx.Response(200, content=b"\x00\x00", request=request)
    )
    client = tts_module.httpx.Client(transport=transport)
    monkeypatch.setattr(tts_module.httpx, "Client", lambda **kwargs: client)
    operation = HttpSpeechOperation(
        endpoint_url="http://localhost:8000/v1/audio/speech",
        api_key=None,
        payload={"input": "hello"},
        timeout_s=2,
    )

    assert list(operation.iter_bytes(lambda: False)) == [b"\x00\x00"]
    assert client.is_closed
    operation.cancel()
    operation.cancel()

    assert client.is_closed


def test_http_speech_cancellation_during_client_construction_prevents_dispatch(monkeypatch):
    construction_started = Event()
    resume_construction = Event()
    stream_calls = 0

    class BlockingClient:
        def __init__(self, **kwargs) -> None:
            del kwargs
            construction_started.set()
            assert resume_construction.wait(1)

        def stream(self, *args, **kwargs):
            nonlocal stream_calls
            del args, kwargs
            stream_calls += 1
            raise AssertionError("cancelled operation must not enter client.stream()")

        def close(self) -> None:
            pass

    monkeypatch.setattr(tts_module.httpx, "Client", BlockingClient)
    operation = HttpSpeechOperation(
        endpoint_url="http://localhost:8000/v1/audio/speech",
        api_key=None,
        payload={"input": "hello"},
        timeout_s=2,
    )
    errors: list[BaseException] = []

    def consume() -> None:
        try:
            list(operation.iter_bytes(lambda: False))
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=consume)
    thread.start()
    assert construction_started.wait(1)
    operation.cancel()
    resume_construction.set()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], SpeechRequestCancelled)
    assert stream_calls == 0


def test_http_speech_timeout_is_a_total_deadline(monkeypatch):
    closed = Event()

    class TricklingResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            del args
            self.close()

        def close(self) -> None:
            closed.set()

        def raise_for_status(self) -> None:
            pass

        def iter_bytes(self):
            while not closed.wait(0.01):
                yield b"\x00\x00"

    response = TricklingResponse()

    class TricklingClient:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def stream(self, *args, **kwargs):
            del args, kwargs
            return response

        def close(self) -> None:
            response.close()

    monkeypatch.setattr(tts_module.httpx, "Client", TricklingClient)
    operation = HttpSpeechOperation(
        endpoint_url="http://localhost:8000/v1/audio/speech",
        api_key=None,
        payload={"input": "hello"},
        timeout_s=0.05,
    )

    started_at_s = perf_counter()
    with pytest.raises(tts_module.SpeechRequestError, match="speech request timed out"):
        list(operation.iter_bytes(lambda: False))

    assert closed.is_set()
    assert perf_counter() - started_at_s < 0.5


@pytest.mark.parametrize(
    "content_type",
    ["application/json; charset=utf-8", "application/problem+json", "text/plain"],
)
def test_openai_tts_rejects_non_audio_success_responses(monkeypatch, content_type):
    transport = tts_module.httpx.MockTransport(
        lambda request: tts_module.httpx.Response(
            200,
            headers={"Content-Type": content_type},
            content=b'{"error":"upstream failure"}',
            request=request,
        )
    )
    client = tts_module.httpx.Client(transport=transport)
    monkeypatch.setattr(tts_module.httpx, "Client", lambda **kwargs: client)
    handler = OpenAICompatibleTTSHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_args=(Event(),),
        setup_kwargs={"sample_rate": 24000, "blocksize": 512},
    )

    assert list(handler.process(TTSInput(text="Hello", response_key="response-1"))) == []

    failure = handler.queue_out.get_nowait()
    assert isinstance(failure, ResponseFailedEvent)
    assert failure.message == "speech endpoint returned a non-audio response"
    assert failure.response_key == "response-1"


def test_openai_tts_cancellation_unblocks_a_stalled_socket_and_session_end():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    host, port = listener.getsockname()
    headers_sent = Event()
    release_server = Event()

    def serve_stalled_response() -> None:
        connection, _ = listener.accept()
        with connection:
            request = b""
            while b"\r\n\r\n" not in request:
                data = connection.recv(4096)
                if not data:
                    return
                request += data
            connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\nTransfer-Encoding: chunked\r\n\r\n")
            headers_sent.set()
            release_server.wait(5)

    server_thread = Thread(target=serve_stalled_response, daemon=True)
    server_thread.start()
    cancel_scope = CancelScope()
    handler = OpenAICompatibleTTSHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_args=(Event(),),
        setup_kwargs={
            "base_url": f"http://{host}:{port}/v1",
            "sample_rate": 24000,
            "blocksize": 512,
            "timeout": 30,
            "cancel_scope": cancel_scope,
        },
    )
    handler.queue_in.put(
        TTSInput(
            text="Hello",
            cancel_generation=cancel_scope.generation,
        )
    )
    handler_thread = Thread(target=handler.run, daemon=True)
    handler_thread.start()

    try:
        assert headers_sent.wait(1)
        cancel_scope.cancel()
        handler.queue_in.put(SESSION_END)
        handler.queue_in.put(PIPELINE_END)

        assert handler.queue_out.get(timeout=1) == SESSION_END
        handler_thread.join(timeout=1)
        assert not handler_thread.is_alive()
        assert handler.queue_out.get_nowait() == PIPELINE_END
    finally:
        release_server.set()
        listener.close()
        server_thread.join(timeout=1)
