from __future__ import annotations

import io
import json
import wave
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
from threading import Barrier, Event, Thread

import numpy as np
import pytest
from openai.types.realtime import ConversationItemInputAudioTranscriptionDeltaEvent

from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.pipeline.events import SpeechStartedEvent
from speech_to_speech.pipeline.messages import (
    PIPELINE_END,
    PartialTranscription,
    Transcription,
    TranscriptionFailure,
    VADAudio,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.STT import openai_compatible_handler as stt_module
from speech_to_speech.STT.openai_compatible_handler import (
    PIPELINE_SAMPLE_RATE,
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


def test_http_transcription_operation_uses_gpt_transcribe_language_contract():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TranscriptionServer)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        operation = HttpTranscriptionOperation(
            endpoint_url=f"http://127.0.0.1:{server.server_port}/v1/audio/transcriptions",
            api_key=None,
            model="gpt-transcribe",
            wav_bytes=b"RIFF-test-wave",
            language="fr",
            response_format="json",
            timeout_s=2,
        )
        operation.run()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    assert b'form-data; name="languages[]"' in _TranscriptionServer.received_body
    assert b'form-data; name="language"' not in _TranscriptionServer.received_body
    assert b"fr" in _TranscriptionServer.received_body


def test_http_transcription_operation_parses_gpt_transcribe_languages():
    operation = HttpTranscriptionOperation(
        endpoint_url="http://127.0.0.1:1/v1/audio/transcriptions",
        api_key=None,
        model="gpt-transcribe",
        wav_bytes=b"RIFF-test-wave",
        language=None,
        response_format="json",
        timeout_s=2,
    )

    result = operation._parse_response(
        json.dumps({"text": "bonjour", "languages": [{"code": "fr"}]}).encode(),
        "application/json",
    )

    assert result == HttpTranscriptionResult(text="bonjour", language="fr")


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

    def cancel(self, reason="superseded"):
        self.cancel_reason = reason

    def run(self, cancel_check=lambda: False):
        if type(self).error is not None:
            raise type(self).error
        return type(self).results.pop(0)


def _handler(
    monkeypatch,
    *,
    tracker: SpeculativeTurnTracker | None = None,
    **setup_overrides,
) -> OpenAICompatibleSTTHandler:
    _FakeOperation.results = [HttpTranscriptionResult(text="")]
    _FakeOperation.error = None
    _FakeOperation.instances = []
    monkeypatch.setattr(stt_module, "HttpTranscriptionOperation", _FakeOperation)
    handler = OpenAICompatibleSTTHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_kwargs={"speculative_turns": tracker, **setup_overrides},
    )
    _FakeOperation.results = []
    return handler


def _audio(mode: str = "final", *, revision: int = 0, samples: int = 160) -> VADAudio:
    return VADAudio(
        audio=np.zeros(samples, dtype=np.float32),
        mode=mode,
        turn_id="turn-1",
        turn_revision=revision,
    )


def _run_final(handler: OpenAICompatibleSTTHandler) -> list:
    assert list(handler.process(_audio())) == []
    thread = handler._final_thread
    assert thread is not None
    thread.join(timeout=1)
    assert not thread.is_alive()
    outputs = []
    while not handler.queue_out.empty():
        outputs.append(handler.queue_out.get_nowait())
    return outputs


def _run_progressive(handler: OpenAICompatibleSTTHandler) -> list[PartialTranscription]:
    assert list(handler.process(_audio("progressive"))) == []
    thread = handler._progressive_thread
    assert thread is not None
    thread.join(timeout=1)
    assert not thread.is_alive()

    outputs = []
    while not handler.queue_out.empty():
        output = handler.queue_out.get_nowait()
        assert isinstance(output, PartialTranscription)
        outputs.append(output)
    return outputs


def test_openai_stt_warmup_uses_configured_operation_before_readiness(monkeypatch):
    handler = _handler(
        monkeypatch,
        base_url="https://transcription.example/v1/",
        api_key="endpoint-secret",
        model="test-model",
        language="en",
        response_format="json",
        timeout=2,
    )

    assert len(_FakeOperation.instances) == 1
    operation = _FakeOperation.instances[0].kwargs
    assert operation["endpoint_url"] == "https://transcription.example/v1/audio/transcriptions"
    assert operation["api_key"] == "endpoint-secret"
    assert operation["model"] == "test-model"
    assert operation["language"] == "en"
    assert operation["response_format"] == "json"
    assert operation["timeout_s"] == 2
    with wave.open(io.BytesIO(operation["wav_bytes"]), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == PIPELINE_SAMPLE_RATE
        assert wav.getnframes() == PIPELINE_SAMPLE_RATE
    assert handler.queue_out.empty()


def test_openai_stt_warmup_failure_prevents_handler_construction(monkeypatch):
    _FakeOperation.results = []
    _FakeOperation.error = TranscriptionRequestError("transcription server returned HTTP 404")
    _FakeOperation.instances = []
    monkeypatch.setattr(stt_module, "HttpTranscriptionOperation", _FakeOperation)

    with pytest.raises(TranscriptionRequestError, match="transcription server returned HTTP 404"):
        OpenAICompatibleSTTHandler(
            Event(),
            queue_in=Queue(),
            queue_out=Queue(),
            setup_kwargs={"model": "missing-model"},
        )


def test_openai_stt_returns_final_transcription(monkeypatch):
    handler = _handler(monkeypatch)
    _FakeOperation.results = [HttpTranscriptionResult(text="hello", language="en")]

    outputs = _run_final(handler)

    assert len(outputs) == 1
    assert isinstance(outputs[0], Transcription)
    assert outputs[0].text == "hello"
    assert outputs[0].language_code == "en"
    assert _FakeOperation.instances[-1].kwargs["endpoint_url"].endswith("/v1/audio/transcriptions")
    assert _FakeOperation.instances[-1].kwargs["wav_bytes"].startswith(b"RIFF")


def test_remote_progressive_hypotheses_remain_cumulative(monkeypatch):
    handler = _handler(monkeypatch)
    _FakeOperation.results = [
        HttpTranscriptionResult(text="hello"),
        HttpTranscriptionResult(text="hello world"),
    ]

    first = _run_progressive(handler)
    second = _run_progressive(handler)

    assert first == [PartialTranscription(text="hello", turn_id="turn-1", turn_revision=0)]
    assert second == [PartialTranscription(text="hello world", turn_id="turn-1", turn_revision=0)]


def test_remote_progressive_hypothesis_corrections_reach_the_router(monkeypatch):
    handler = _handler(monkeypatch)
    _FakeOperation.results = [
        HttpTranscriptionResult(text="hello there"),
        HttpTranscriptionResult(text="hello their"),
    ]

    assert _run_progressive(handler) == [PartialTranscription(text="hello there", turn_id="turn-1", turn_revision=0)]
    assert _run_progressive(handler) == [PartialTranscription(text="hello their", turn_id="turn-1", turn_revision=0)]


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
        for partial in _run_progressive(handler):
            assert list(notifier.process(partial)) == []
            wire_events.extend(service.dispatch_pipeline_event(conn_id, text_output_queue.get_nowait()))

    assert all(isinstance(event, ConversationItemInputAudioTranscriptionDeltaEvent) for event in wire_events)
    assert [event.delta for event in wire_events] == ["hello", " world"]
    service.unregister(conn_id)


def test_final_transport_failure_does_not_create_a_transcription(monkeypatch):
    handler = _handler(monkeypatch)
    _FakeOperation.error = TranscriptionRequestError("transcription request timed out")

    outputs = _run_final(handler)

    assert len(outputs) == 1
    assert isinstance(outputs[0], TranscriptionFailure)
    assert outputs[0].message == "transcription request timed out"
    assert outputs[0].turn_id == "turn-1"


def test_progressive_transport_failure_is_discarded(monkeypatch):
    handler = _handler(monkeypatch)
    _FakeOperation.error = TranscriptionRequestError("transcription request timed out")

    assert _run_progressive(handler) == []


def test_final_request_does_not_wait_for_in_flight_progressive(monkeypatch):
    handler = _handler(monkeypatch)
    progressive_started = Event()
    release_progressive = Event()
    final_started = Event()

    class _BlockingProgressiveOperation(_FakeOperation):
        def run(self, cancel_check=lambda: False):
            progressive_started.set()
            assert release_progressive.wait(timeout=2)
            return HttpTranscriptionResult(text="partial")

    class _FinalOperation(_FakeOperation):
        def run(self, cancel_check=lambda: False):
            final_started.set()
            return HttpTranscriptionResult(text="final", language="en")

    operations = iter([_BlockingProgressiveOperation(), _FinalOperation()])
    monkeypatch.setattr(handler, "_make_operation", lambda _audio: next(operations))
    handler_thread = Thread(target=handler.run, daemon=True)
    handler_thread.start()

    try:
        handler.queue_in.put(_audio("progressive"))
        assert progressive_started.wait(timeout=1)
        handler.queue_in.put(_audio())

        assert final_started.wait(timeout=1)
        output = handler.queue_out.get(timeout=1)
        assert isinstance(output, Transcription)
        assert output.text == "final"
        assert output.language_code == "en"
        assert output.turn_id == "turn-1"
        assert output.turn_revision == 0
        assert not release_progressive.is_set()

        release_progressive.set()
        thread = handler._progressive_thread
        assert thread is not None
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert handler.queue_out.empty()
    finally:
        release_progressive.set()
        handler.stop_event.set()
        handler.queue_in.put(PIPELINE_END)
        handler_thread.join(timeout=1)
    assert not handler_thread.is_alive()


def test_final_requests_from_pipelines_using_the_same_endpoint_can_overlap(monkeypatch):
    handlers = [_handler(monkeypatch, api_key="shared-endpoint-key") for _ in range(2)]
    both_requests_started = Barrier(2)

    class _ConcurrentOperation(_FakeOperation):
        def run(self, cancel_check=lambda: False):
            both_requests_started.wait(timeout=2)
            return HttpTranscriptionResult(text="final", language="en")

    for handler in handlers:
        monkeypatch.setattr(handler, "_make_operation", lambda _audio: _ConcurrentOperation())

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(lambda handler=handler: _run_final(handler)) for handler in handlers]
            outputs = [future.result(timeout=3) for future in futures]
        for output in outputs:
            assert len(output) == 1
            assert isinstance(output[0], Transcription)
            assert output[0].text == "final"
    finally:
        for handler in handlers:
            handler.cleanup()


def test_pending_progressive_requests_keep_only_the_latest_window(monkeypatch):
    handler = _handler(monkeypatch)
    progressive_started = Event()
    release_progressive = Event()
    dispatched_samples = []

    class _BlockingProgressiveOperation(_FakeOperation):
        def run(self, cancel_check=lambda: False):
            progressive_started.set()
            assert release_progressive.wait(timeout=2)
            return HttpTranscriptionResult(text="partial")

    def make_operation(_audio):
        dispatched_samples.append(len(_audio))
        return _BlockingProgressiveOperation()

    monkeypatch.setattr(handler, "_make_operation", make_operation)

    assert list(handler.process(_audio("progressive"))) == []
    assert progressive_started.wait(timeout=1)
    assert list(handler.process(_audio("progressive", samples=320))) == []
    assert list(handler.process(_audio("progressive", samples=480))) == []
    assert dispatched_samples == [160]

    release_progressive.set()
    thread = handler._progressive_thread
    assert thread is not None
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert dispatched_samples == [160, 480]
    assert handler.queue_out.qsize() == 2
    assert handler.queue_out.get_nowait() == PartialTranscription(
        text="partial",
        turn_id="turn-1",
        turn_revision=0,
    )


def test_session_end_suppresses_in_flight_progressive_result(monkeypatch):
    handler = _handler(monkeypatch)
    progressive_started = Event()
    release_progressive = Event()

    class _BlockingProgressiveOperation(_FakeOperation):
        def run(self, cancel_check=lambda: False):
            progressive_started.set()
            assert release_progressive.wait(timeout=2)
            return HttpTranscriptionResult(text="old session")

    monkeypatch.setattr(handler, "_make_operation", lambda _audio: _BlockingProgressiveOperation())

    assert list(handler.process(_audio("progressive"))) == []
    assert progressive_started.wait(timeout=1)
    handler.on_session_end()
    release_progressive.set()

    thread = handler._progressive_thread
    assert thread is not None
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert handler.queue_out.empty()


@pytest.mark.parametrize("superseded_by", ["final", "new_revision", "session_end", "shutdown", "cleanup"])
def test_obsolete_progressive_request_is_not_sent_before_worker_starts(monkeypatch, superseded_by):
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn-1", 0)
    handler = _handler(monkeypatch, tracker=tracker)
    worker_started = Event()
    release_worker = Event()
    run_request = handler._run_request

    def delayed_worker(request):
        if request.source.mode == "progressive":
            worker_started.set()
            assert release_worker.wait(timeout=2)
        run_request(request)

    monkeypatch.setattr(handler, "_run_request", delayed_worker)
    _FakeOperation.results = [HttpTranscriptionResult(text="final")]
    _FakeOperation.instances = []
    thread = None
    try:
        assert list(handler.process(_audio("progressive"))) == []
        thread = handler._progressive_thread
        assert thread is not None
        assert worker_started.wait(timeout=1)

        if superseded_by == "final":
            outputs = _run_final(handler)
            assert len(outputs) == 1
            assert isinstance(outputs[0], Transcription)
            assert outputs[0].text == "final"
        elif superseded_by == "new_revision":
            tracker.observe("turn-1", 1)
        elif superseded_by == "session_end":
            handler.on_session_end()
        elif superseded_by == "shutdown":
            handler.stop_event.set()
        else:
            handler.cleanup()

        release_worker.set()
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert len(_FakeOperation.instances) == (1 if superseded_by == "final" else 0)
        assert handler.queue_out.empty()
    finally:
        release_worker.set()
        if thread is not None:
            thread.join(timeout=1)
        handler.cleanup()


def test_stale_revision_is_dropped_after_request(monkeypatch):
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn-1", 0)
    handler = _handler(monkeypatch, tracker=tracker)

    class _ReopeningOperation(_FakeOperation):
        def run(self, cancel_check=lambda: False):
            tracker.observe("turn-1", 1)
            return HttpTranscriptionResult(text="stale")

    monkeypatch.setattr(stt_module, "HttpTranscriptionOperation", _ReopeningOperation)

    assert _run_final(handler) == []


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
