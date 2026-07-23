from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
from threading import Event, Thread

import numpy as np

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.STT.endpoint_admission import TranscriptionCancelled
from speech_to_speech.STT.openai_compatible_handler import (
    HttpTranscriptionOperation,
)
from speech_to_speech.TTS import openai_compatible_handler as tts_module
from speech_to_speech.TTS.openai_compatible_handler import OpenAICompatibleTTSHandler


class _TranscriptionServer(BaseHTTPRequestHandler):
    received_path = ""
    received_body = b""

    def do_POST(self) -> None:
        type(self).received_path = self.path
        length = int(self.headers["content-length"])
        type(self).received_body = self.rfile.read(length)
        body = json.dumps({"text": "hello", "language": "en"}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        del format, args


def test_http_transcription_operation_uploads_wav_multipart():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TranscriptionServer)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        operation = HttpTranscriptionOperation(
            request_id="request-1",
            endpoint_url=f"http://127.0.0.1:{server.server_port}/v1/audio/transcriptions",
            api_key=None,
            model="test-model",
            wav_bytes=b"RIFF-test-wave",
            language="en",
            response_format="json",
            timeout_s=2,
        )
        result = operation.run()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    assert result.text == "hello"
    assert result.language == "en"
    assert _TranscriptionServer.received_path == "/v1/audio/transcriptions"
    assert b'form-data; name="model"' in _TranscriptionServer.received_body
    assert b"test-model" in _TranscriptionServer.received_body
    assert b'filename="audio.wav"' in _TranscriptionServer.received_body
    assert b"RIFF-test-wave" in _TranscriptionServer.received_body


def test_http_transcription_operation_can_select_nim_model_by_language():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TranscriptionServer)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        operation = HttpTranscriptionOperation(
            request_id="request-2",
            endpoint_url=f"http://127.0.0.1:{server.server_port}/v1/audio/transcriptions",
            api_key=None,
            model=None,
            wav_bytes=b"RIFF-test-wave",
            language="en-US",
            response_format="json",
            timeout_s=2,
        )
        operation.run()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    assert b'form-data; name="model"' not in _TranscriptionServer.received_body
    assert b'form-data; name="language"' in _TranscriptionServer.received_body
    assert b"en-US" in _TranscriptionServer.received_body


def test_http_transcription_operation_preserves_cancellation_reason_and_request_id():
    operation = HttpTranscriptionOperation(
        request_id="request-cancelled",
        endpoint_url="http://127.0.0.1:1/v1/audio/transcriptions",
        api_key=None,
        model="test-model",
        wav_bytes=b"RIFF-test-wave",
        language=None,
        response_format="json",
        timeout_s=2,
    )

    operation.cancel("turn_reopened")

    try:
        operation.run()
    except TranscriptionCancelled as exc:
        assert exc.request_id == "request-cancelled"
        assert exc.reason == "turn_reopened"
    else:
        raise AssertionError("cancelled operation did not raise TranscriptionCancelled")


class _FakeSpeechOperation:
    instances: list["_FakeSpeechOperation"] = []
    startup_action = None
    startup_error: Exception | None = None

    def __init__(self, **kwargs) -> None:
        self.payload = kwargs["payload"]
        self.cancelled = False
        type(self).instances.append(self)

    def iter_bytes(self, cancel_check):
        if type(self).startup_action is not None:
            type(self).startup_action()
        if type(self).startup_error is not None:
            raise type(self).startup_error
        samples = np.arange(2400, dtype="<i2")
        encoded = samples.tobytes()
        for offset in range(0, len(encoded), 301):
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
            )
        )
    )

    assert chunks == []
    assert tracker.commit_calls == 0
    assert not tracker.is_committed("turn-1", 0)


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
