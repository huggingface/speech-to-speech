from __future__ import annotations

import io
import json
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
from threading import Event, Thread

import numpy as np
from openai.types.realtime import ConversationItemInputAudioTranscriptionDeltaEvent

from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.pipeline.events import SpeechStartedEvent
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
    TranscriptionRequestError,
)
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier


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
            endpoint_url=f"http://127.0.0.1:{server.server_port}/v1/audio/transcriptions",
            api_key=None,
            model="test-model",
            wav_bytes=OpenAICompatibleSTTHandler._encode_wav(np.zeros(160, dtype=np.float32)),
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
    assert b"RIFF" in _TranscriptionServer.received_body


def test_http_transcription_operation_can_select_model_by_language():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TranscriptionServer)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        operation = HttpTranscriptionOperation(
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


def test_http_transcription_operation_parses_plain_text():
    operation = HttpTranscriptionOperation(
        endpoint_url="http://127.0.0.1:1/v1/audio/transcriptions",
        api_key=None,
        model="test-model",
        wav_bytes=b"RIFF-test-wave",
        language="en",
        response_format="text",
        timeout_s=2,
    )

    result = operation._parse_response(b" hello world\n", "text/plain; charset=utf-8")

    assert result == HttpTranscriptionResult(text="hello world", language="en")


def test_openai_stt_encodes_mono_pcm16_16khz_wav():
    encoded = OpenAICompatibleSTTHandler._encode_wav(np.array([-1.0, 0.0, 1.0], dtype=np.float32))

    with wave.open(io.BytesIO(encoded), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        assert wav.getnframes() == 3


class _FakeOperation:
    results: list[HttpTranscriptionResult] = []
    error: Exception | None = None
    instances: list[_FakeOperation] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        type(self).instances.append(self)

    def run(self):
        if type(self).error is not None:
            raise type(self).error
        return type(self).results.pop(0)


def _handler(
    monkeypatch,
    *,
    tracker: SpeculativeTurnTracker | None = None,
    **setup_overrides,
) -> OpenAICompatibleSTTHandler:
    _FakeOperation.results = []
    _FakeOperation.error = None
    _FakeOperation.instances = []
    monkeypatch.setattr(stt_module, "HttpTranscriptionOperation", _FakeOperation)
    return OpenAICompatibleSTTHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_kwargs={"speculative_turns": tracker, **setup_overrides},
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


def test_remote_progressive_hypotheses_remain_cumulative(monkeypatch):
    handler = _handler(monkeypatch)
    _FakeOperation.results = [
        HttpTranscriptionResult(text="hello"),
        HttpTranscriptionResult(text="hello world"),
    ]

    first = list(handler.process(_audio("progressive")))
    second = list(handler.process(_audio("progressive")))

    assert first == [PartialTranscription(text="hello", turn_id="turn-1", turn_revision=0)]
    assert second == [PartialTranscription(text="hello world", turn_id="turn-1", turn_revision=0)]


def test_remote_progressive_hypothesis_corrections_reach_the_router(monkeypatch):
    handler = _handler(monkeypatch)
    _FakeOperation.results = [
        HttpTranscriptionResult(text="hello there"),
        HttpTranscriptionResult(text="hello their"),
    ]

    assert list(handler.process(_audio("progressive"))) == [
        PartialTranscription(text="hello there", turn_id="turn-1", turn_revision=0)
    ]
    assert list(handler.process(_audio("progressive"))) == [
        PartialTranscription(text="hello their", turn_id="turn-1", turn_revision=0)
    ]


def test_remote_progressive_hypotheses_emit_realtime_deltas(monkeypatch):
    handler = _handler(monkeypatch)
    _FakeOperation.results = [
        HttpTranscriptionResult(text="hello"),
        HttpTranscriptionResult(text="hello world"),
        HttpTranscriptionResult(text="hello world again"),
        HttpTranscriptionResult(text="hello world again today"),
    ]
    text_output_queue = Queue()
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(text_output_queue=text_output_queue)
    service = RealtimeService()
    conn_id = service.register()
    service.dispatch_pipeline_event(
        conn_id,
        SpeechStartedEvent(turn_id="turn-1", turn_revision=0),
    )

    wire_events = []
    for _ in range(4):
        for partial in handler.process(_audio("progressive")):
            assert list(notifier.process(partial)) == []
            wire_events.extend(service.dispatch_pipeline_event(conn_id, text_output_queue.get_nowait()))

    assert all(isinstance(event, ConversationItemInputAudioTranscriptionDeltaEvent) for event in wire_events)
    assert [event.delta for event in wire_events] == ["hello", " world"]
    service.unregister(conn_id)


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


def test_stale_revision_is_dropped_after_request(monkeypatch):
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn-1", 0)
    handler = _handler(monkeypatch, tracker=tracker)

    class _ReopeningOperation(_FakeOperation):
        def run(self):
            tracker.observe("turn-1", 1)
            return HttpTranscriptionResult(text="stale")

    monkeypatch.setattr(stt_module, "HttpTranscriptionOperation", _ReopeningOperation)

    assert list(handler.process(_audio())) == []


def test_openai_api_key_is_not_sent_to_other_endpoints(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "official-secret")

    local_handler = _handler(monkeypatch, base_url="http://localhost:8000/v1")
    official_handler = _handler(monkeypatch, base_url="https://api.openai.com/v1/")
    explicit_handler = _handler(
        monkeypatch,
        base_url="https://transcription.example/v1",
        api_key="endpoint-secret",
    )

    assert local_handler.api_key is None
    assert official_handler.api_key == "official-secret"
    assert explicit_handler.api_key == "endpoint-secret"
