from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
from threading import Event, Thread

import numpy as np

from speech_to_speech.pipeline.messages import (
    PartialTranscription,
    Transcription,
    TranscriptionFailure,
    VADAudio,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.STT import openai_compatible_handler as stt_module
from speech_to_speech.STT.openai_compatible_handler import (
    HttpTranscriptionOperation,
    HttpTranscriptionResult,
    OpenAICompatibleSTTHandler,
    TranscriptionRequestCancelled,
    TranscriptionRequestError,
)


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

    assert result == HttpTranscriptionResult(text="hello", language="en")
    assert _TranscriptionServer.received_path == "/v1/audio/transcriptions"
    assert b'form-data; name="model"' in _TranscriptionServer.received_body
    assert b"test-model" in _TranscriptionServer.received_body
    assert b'filename="audio.wav"' in _TranscriptionServer.received_body
    assert b"RIFF-test-wave" in _TranscriptionServer.received_body


def test_http_transcription_operation_can_select_model_by_language():
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


def test_http_transcription_operation_preserves_cancellation_context():
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
    except TranscriptionRequestCancelled as exc:
        assert exc.request_id == "request-cancelled"
        assert exc.reason == "turn_reopened"
    else:
        raise AssertionError("cancelled operation did not raise TranscriptionRequestCancelled")


class _FakeOperation:
    results: list[HttpTranscriptionResult] = []
    error: Exception | None = None
    instances: list[_FakeOperation] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.cancel_reason: str | None = None
        type(self).instances.append(self)

    def run(self, cancel_check):
        if type(self).error is not None:
            raise type(self).error
        if cancel_check():
            self.cancel("stale")
            raise TranscriptionRequestCancelled(self.kwargs["request_id"], "stale")
        return type(self).results.pop(0)

    def cancel(self, reason: str) -> None:
        self.cancel_reason = reason


def _handler(monkeypatch, *, tracker: SpeculativeTurnTracker | None = None) -> OpenAICompatibleSTTHandler:
    _FakeOperation.results = []
    _FakeOperation.error = None
    _FakeOperation.instances = []
    monkeypatch.setattr(stt_module, "HttpTranscriptionOperation", _FakeOperation)
    return OpenAICompatibleSTTHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_kwargs={"speculative_turns": tracker},
    )


def _audio(mode: str = "final", *, revision: int = 0) -> VADAudio:
    return VADAudio(
        audio=np.zeros(160, dtype=np.float32),
        mode=mode,
        turn_id="turn-1",
        turn_revision=revision,
    )


def test_openai_stt_returns_final_transcription(monkeypatch):
    handler = _handler(monkeypatch)
    _FakeOperation.results = [HttpTranscriptionResult(text="hello", language="en")]

    outputs = list(handler.process(_audio()))

    assert len(outputs) == 1
    assert isinstance(outputs[0], Transcription)
    assert outputs[0].text == "hello"
    assert outputs[0].language_code == "en"
    assert _FakeOperation.instances[0].kwargs["endpoint_url"].endswith("/v1/audio/transcriptions")
    assert _FakeOperation.instances[0].kwargs["wav_bytes"].startswith(b"RIFF")


def test_remote_progressive_hypotheses_emit_only_extension_deltas(monkeypatch):
    handler = _handler(monkeypatch)
    _FakeOperation.results = [
        HttpTranscriptionResult(text="hello"),
        HttpTranscriptionResult(text="hello world"),
    ]

    first = list(handler.process(_audio("progressive")))
    second = list(handler.process(_audio("progressive")))

    assert first == [PartialTranscription(text="hello", turn_id="turn-1", turn_revision=0)]
    assert second == [PartialTranscription(text=" world", turn_id="turn-1", turn_revision=0)]


def test_remote_progressive_hypothesis_corrections_are_suppressed(monkeypatch):
    handler = _handler(monkeypatch)
    _FakeOperation.results = [
        HttpTranscriptionResult(text="hello there"),
        HttpTranscriptionResult(text="hello their"),
    ]

    assert list(handler.process(_audio("progressive")))
    assert list(handler.process(_audio("progressive"))) == []


def test_final_transport_failure_does_not_create_a_transcription(monkeypatch):
    handler = _handler(monkeypatch)
    _FakeOperation.error = TranscriptionRequestError("transcription request timed out")

    outputs = list(handler.process(_audio()))

    assert len(outputs) == 1
    assert isinstance(outputs[0], TranscriptionFailure)
    assert outputs[0].message == "transcription request timed out"
    assert outputs[0].turn_id == "turn-1"


def test_progressive_transport_failure_is_discarded(monkeypatch):
    handler = _handler(monkeypatch)
    _FakeOperation.error = TranscriptionRequestError("transcription request timed out")

    assert list(handler.process(_audio("progressive"))) == []


def test_stale_revision_is_cancelled_before_publication(monkeypatch):
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn-1", 0)
    handler = _handler(monkeypatch, tracker=tracker)

    class _ReopeningOperation(_FakeOperation):
        def run(self, cancel_check):
            tracker.observe("turn-1", 1)
            assert cancel_check()
            self.cancel("stale")
            raise TranscriptionRequestCancelled(self.kwargs["request_id"], "stale")

    monkeypatch.setattr(stt_module, "HttpTranscriptionOperation", _ReopeningOperation)

    assert list(handler.process(_audio())) == []
    assert _ReopeningOperation.instances[-1].cancel_reason == "stale"


def test_session_end_cancels_active_transport(monkeypatch):
    handler = _handler(monkeypatch)
    operation = _FakeOperation(request_id="active")
    handler._active_operation = operation

    handler.on_session_end()

    assert operation.cancel_reason == "session_end"
