from __future__ import annotations

import json
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
from threading import Event, Thread
from time import perf_counter

import numpy as np

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.control import SESSION_END
from speech_to_speech.pipeline.events import ResponseFailedEvent
from speech_to_speech.pipeline.messages import (
    AUDIO_RESPONSE_DONE,
    EndOfResponse,
    PartialTranscription,
    TranscriptionFailure,
    TTSInput,
    VADAudio,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.STT import openai_compatible_handler as stt_module
from speech_to_speech.STT.endpoint_admission import AdmissionRejected, TranscriptionCancelled
from speech_to_speech.STT.openai_compatible_handler import (
    HttpTranscriptionOperation,
    HttpTranscriptionResult,
    OpenAICompatibleSTTHandler,
)
from speech_to_speech.TTS import openai_compatible_handler as tts_module
from speech_to_speech.TTS.openai_compatible_handler import (
    HttpSpeechOperation,
    OpenAICompatibleTTSHandler,
    SpeechRequestCancelled,
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
        self.cancellations = []

    def submit(self, request):
        self.requests.append(request)
        future = Future()
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


class _FakeSpeechOperation:
    instances: list["_FakeSpeechOperation"] = []
    startup_action = None
    startup_error: Exception | None = None
    failure_after_chunks: int | None = None

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
        for index, offset in enumerate(range(0, len(encoded), 301)):
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


def _openai_tts_handler(
    monkeypatch,
    *,
    cancel_scope: CancelScope | None = None,
    speculative_turns: SpeculativeTurnTracker | None = None,
    text_output_queue: Queue | None = None,
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
            "text_output_queue": text_output_queue,
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
    text_output_queue = Queue()
    handler = _openai_tts_handler(
        monkeypatch,
        speculative_turns=tracker,
        text_output_queue=text_output_queue,
    )
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
    failure = text_output_queue.get_nowait()
    assert isinstance(failure, ResponseFailedEvent)
    assert failure.message == "speech server returned HTTP 500"
    assert failure.turn_id == "turn-1"
    assert failure.turn_revision == 0


def test_openai_tts_failure_after_audio_emits_response_failure(monkeypatch):
    text_output_queue = Queue()
    handler = _openai_tts_handler(monkeypatch, text_output_queue=text_output_queue)
    _FakeSpeechOperation.failure_after_chunks = 8
    generation = handler.process(TTSInput(text="Hello", turn_id="turn-1", turn_revision=0))

    first_audio = next(generation)
    remaining = list(generation)

    assert isinstance(first_audio, np.ndarray)
    assert remaining == []
    failure = text_output_queue.get_nowait()
    assert isinstance(failure, ResponseFailedEvent)
    assert failure.message == "speech stream failed"
    assert failure.turn_id == "turn-1"
    assert failure.turn_revision == 0


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
