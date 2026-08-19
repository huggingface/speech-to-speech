from __future__ import annotations

import json
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
from threading import Event, Thread
from time import perf_counter

import numpy as np

from speech_to_speech.pipeline.control import SESSION_END
from speech_to_speech.pipeline.messages import (
    PartialTranscription,
    Transcription,
    TranscriptionFailure,
    VADAudio,
)
from speech_to_speech.STT import openai_compatible_handler as stt_module
from speech_to_speech.STT.endpoint_admission import AdmissionRejected, TranscriptionCancelled
from speech_to_speech.STT.openai_compatible_handler import (
    HttpTranscriptionOperation,
    HttpTranscriptionResult,
    OpenAICompatibleSTTHandler,
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


class _FakeAdmission:
    def __init__(self, *, rejection: Exception | None = None) -> None:
        self.rejection = rejection
        self.requests = []
        self.futures = []
        self.cancellations = []

    def submit(self, request):
        self.requests.append(request)
        future = Future()
        self.futures.append(future)
        if self.rejection is not None:
            future.set_exception(self.rejection)
        return future

    def cancel(self, message) -> int:
        self.cancellations.append(message)
        return 0


class _FakeAdmissionLease:
    def __init__(self, controller: _FakeAdmission) -> None:
        self.controller = controller
        self.released = False

    def release(self) -> None:
        self.released = True


def _openai_stt_handler(admission: _FakeAdmission | None = None) -> tuple[OpenAICompatibleSTTHandler, _FakeAdmission]:
    admission = admission or _FakeAdmission()
    handler = OpenAICompatibleSTTHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_args=(_FakeAdmissionLease(admission),),
    )
    return handler, admission


def _completed(source: VADAudio, generation: int = 0):
    return stt_module._CompletedRequest(
        source=source,
        future=Future(),
        session_generation=generation,
        started_at_s=perf_counter(),
    )


def test_stt_session_teardown_fences_a_completion_paused_before_publication():
    handler, _ = _openai_stt_handler()
    reached_publication = Event()
    resume_publication = Event()
    publication_finished = Event()
    original_publish = handler._publish_output

    def paused_publish(completed, output):
        reached_publication.set()
        assert resume_publication.wait(1)
        try:
            return original_publish(completed, output)
        finally:
            publication_finished.set()

    handler._publish_output = paused_publish
    source = VADAudio(
        audio=np.zeros(160, dtype=np.float32),
        mode="final",
        turn_id="turn-1",
        turn_revision=0,
    )
    completion = _completed(source)
    completion.future.set_result(HttpTranscriptionResult(text="old session"))

    try:
        handler._completion_queue.put(completion)
        assert reached_publication.wait(1)
        handler.on_session_end()
        handler.queue_out.put(SESSION_END)
        resume_publication.set()
        assert publication_finished.wait(1)

        assert handler.queue_out.get(timeout=1) == SESSION_END
        assert handler.queue_out.empty()
    finally:
        resume_publication.set()
        handler.cleanup()


def test_final_admission_rejection_publishes_transcription_failure():
    handler, _ = _openai_stt_handler(_FakeAdmission(rejection=AdmissionRejected("queue is full")))
    source = VADAudio(
        audio=np.zeros(160, dtype=np.float32),
        mode="final",
        turn_id="turn-1",
        turn_revision=0,
    )

    try:
        assert list(handler.process(source)) == []
        failure = handler.queue_out.get(timeout=1)
        assert isinstance(failure, TranscriptionFailure)
        assert failure.message == "transcription endpoint is overloaded"
        assert failure.turn_id == "turn-1"
        assert failure.turn_revision == 0
    finally:
        handler.cleanup()


def test_stt_rotates_admission_owner_between_sessions():
    handler, admission = _openai_stt_handler()
    first = VADAudio(
        audio=np.zeros(160, dtype=np.float32),
        mode="progressive",
        turn_id="turn-1",
        turn_revision=0,
    )

    try:
        assert list(handler.process(first)) == []
        first_owner = admission.requests[-1].owner_id
        handler.on_session_end()
        assert admission.cancellations[-1].owner_id == first_owner

        assert list(handler.process(first)) == []
        assert admission.requests[-1].owner_id != first_owner
    finally:
        handler.cleanup()


def test_remote_progressive_hypotheses_emit_only_extension_deltas():
    handler, _ = _openai_stt_handler()
    source = VADAudio(
        audio=np.zeros(160, dtype=np.float32),
        mode="progressive",
        turn_id="turn-1",
        turn_revision=0,
    )
    completion = _completed(source)

    try:
        assert handler._publish_output(
            completion,
            PartialTranscription(text="hello", turn_id="turn-1", turn_revision=0),
        )
        assert handler._publish_output(
            completion,
            PartialTranscription(text="hello world", turn_id="turn-1", turn_revision=0),
        )

        assert handler.queue_out.get_nowait().text == "hello"
        assert handler.queue_out.get_nowait().text == " world"
        assert handler.queue_out.empty()
    finally:
        handler.cleanup()


def test_remote_progressive_hypothesis_corrections_are_suppressed():
    handler, _ = _openai_stt_handler()
    source = VADAudio(
        audio=np.zeros(160, dtype=np.float32),
        mode="progressive",
        turn_id="turn-1",
        turn_revision=0,
    )
    completion = _completed(source)

    try:
        assert handler._publish_output(
            completion,
            PartialTranscription(text="hello there", turn_id="turn-1", turn_revision=0),
        )
        assert not handler._publish_output(
            completion,
            PartialTranscription(text="hello their", turn_id="turn-1", turn_revision=0),
        )

        assert handler.queue_out.get_nowait().text == "hello there"
        assert handler.queue_out.empty()
    finally:
        handler.cleanup()


def test_successful_final_completion_is_delivered_asynchronously():
    handler, admission = _openai_stt_handler()
    source = VADAudio(
        audio=np.zeros(160, dtype=np.float32),
        mode="final",
        turn_id="turn-1",
        turn_revision=0,
    )

    try:
        assert list(handler.process(source)) == []
        admission.futures[-1].set_result(HttpTranscriptionResult(text="hello", language="en"))
        transcription = handler.queue_out.get(timeout=1)
        assert isinstance(transcription, Transcription)
        assert transcription.text == "hello"
        assert transcription.language_code == "en"
        assert transcription.turn_id == "turn-1"
    finally:
        handler.cleanup()


def test_final_transport_failure_is_delivered_without_llm_input():
    handler, admission = _openai_stt_handler()
    source = VADAudio(
        audio=np.zeros(160, dtype=np.float32),
        mode="final",
        turn_id="turn-1",
        turn_revision=0,
    )

    try:
        assert list(handler.process(source)) == []
        admission.futures[-1].set_exception(stt_module.TranscriptionRequestError("transcription request timed out"))
        failure = handler.queue_out.get(timeout=1)
        assert isinstance(failure, TranscriptionFailure)
        assert failure.message == "transcription request timed out"
        assert failure.turn_id == "turn-1"
    finally:
        handler.cleanup()
