from __future__ import annotations

import asyncio
import io
import socket
import wave
from queue import Queue
from threading import Event, Thread

import anyio
import numpy as np
import pytest

from speech_to_speech.pipeline.control import SESSION_END
from speech_to_speech.pipeline.messages import PIPELINE_END, Transcription, TranscriptionFailure, VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.STT import openai_compatible_handler as stt_module

pytestmark = pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")


def _operation(url="http://127.0.0.1:1/v1/audio/transcriptions"):
    return stt_module.HttpTranscriptionOperation(
        endpoint_url=url,
        api_key=None,
        model="test-model",
        wav_bytes=b"RIFF-test-wave",
        language=None,
        response_format="json",
        timeout_s=10,
    )


def test_cancelled_transcription_does_not_start_http():
    operation = _operation()
    operation.cancel("session_end")
    operation.cancel("shutdown")

    with pytest.raises(stt_module.TranscriptionRequestCancelled) as cancelled:
        operation.run()

    assert cancelled.value.reason == "session_end"


def test_http_worker_startup_failure_reaches_the_caller(monkeypatch):
    def fail_loop_creation():
        raise OSError("event loop unavailable")

    monkeypatch.setattr(stt_module.asyncio, "new_event_loop", fail_loop_creation)
    operation = _operation()
    errors = []

    def run():
        try:
            operation.run()
        except Exception as exc:
            errors.append(exc)

    worker = Thread(target=run, daemon=True)
    worker.start()
    try:
        worker.join(timeout=1)
        assert not worker.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], OSError)
        assert str(errors[0]) == "event loop unavailable"
    finally:
        operation.cancel("shutdown")
        worker.join(timeout=1)


@pytest.fixture(params=["headers", "body"])
def stalled_stt_endpoint(request):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(3)
    received = Event()
    closed = Event()

    def serve():
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(3)
            headers = b""
            while b"\r\n\r\n" not in headers:
                chunk = connection.recv(4096)
                if not chunk:
                    return
                headers += chunk
            if request.param == "body":
                connection.sendall(
                    b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n{"text":'
                )
            received.set()
            try:
                while connection.recv(4096):
                    pass
            except ConnectionResetError:
                pass
            closed.set()

    server = Thread(target=serve, daemon=True)
    server.start()
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}/v1/audio/transcriptions", received, closed
    finally:
        listener.close()
        server.join(timeout=4)


@pytest.mark.parametrize("cancel_source", ["explicit", "stale"])
def test_cancellation_interrupts_stalled_http_and_closes_socket(stalled_stt_endpoint, cancel_source):
    url, received, closed = stalled_stt_endpoint
    operation = _operation(url)
    stale = Event()
    errors = []

    def run():
        try:
            operation.run(cancel_check=stale.is_set)
        except Exception as exc:
            errors.append(exc)

    worker = Thread(target=run, daemon=True)
    worker.start()
    try:
        assert received.wait(timeout=2)
        if cancel_source == "explicit":
            operation.cancel("session_end")
        else:
            stale.set()
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], stt_module.TranscriptionRequestCancelled)
        assert closed.wait(timeout=1)
        assert operation._worker_loop is None
        assert operation._worker_task is None
    finally:
        stale.set()
        operation.cancel("shutdown")
        worker.join(timeout=2)


@pytest.fixture
def blocked_upload_endpoint():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(0.1)
    stopped = Event()
    connections = []

    def serve():
        try:
            while not stopped.is_set():
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                connections.append(connection)
                if len(connections) == 1:
                    # Never read the old upload; it cannot fit in the socket buffers.
                    continue
                connection.settimeout(3)
                with connection, connection.makefile("rb") as request:
                    request.readline()
                    headers = {}
                    while (line := request.readline()) not in (b"\r\n", b""):
                        key, value = line.decode().split(":", 1)
                        headers[key.lower()] = value.strip()
                    body = request.read(int(headers["content-length"]))
                    assert len(body) == int(headers["content-length"])
                    response = b'{"text":"new session"}'
                    connection.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                        + str(len(response)).encode()
                        + b"\r\nConnection: close\r\n\r\n"
                        + response
                    )
        except OSError:
            if not stopped.is_set():
                raise

    def stop():
        stopped.set()
        listener.close()
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

    server = Thread(target=serve, daemon=True)
    server.start()
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}/v1/audio/transcriptions", stop
    finally:
        stop()
        server.join(timeout=2)
        assert not server.is_alive()


@pytest.mark.parametrize("cancel_at", ["connection_handoff", "upload"])
def test_connection_cancellation_survives_blocked_upload_and_session_reuse(
    blocked_upload_endpoint, monkeypatch, cancel_at
):
    url, stop_server = blocked_upload_endpoint
    connected, first_cancel_consumed = Event(), Event()
    connect_tcp = anyio.connect_tcp
    sockets = []

    async def connection_handoff(*args, **kwargs):
        stream = await connect_tcp(*args, **kwargs)
        sockets.append(stream.extra(anyio.abc.SocketAttribute.raw_socket))
        if len(sockets) == 1:
            sockets[0].setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
            if cancel_at == "connection_handoff":
                connected.set()
                try:
                    await anyio.sleep_forever()
                except asyncio.CancelledError:
                    # Deterministically reproduce a connection handoff consuming the first
                    # cancellation. The real HTTP upload must still remain cancellable.
                    first_cancel_consumed.set()
            else:
                send = stream.send

                async def send_upload(item):
                    if len(item) >= 65536:
                        connected.set()
                    await send(item)

                monkeypatch.setattr(stream, "send", send_upload)
        return stream

    monkeypatch.setattr(anyio, "connect_tcp", connection_handoff)
    monkeypatch.setattr(stt_module.OpenAICompatibleSTTHandler, "warmup", lambda self: None)
    handler = stt_module.OpenAICompatibleSTTHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_kwargs={"base_url": url.removesuffix("/audio/transcriptions")},
    )
    operations = []
    make_operation = handler._make_operation

    def record_operation(samples, **kwargs):
        operation = make_operation(samples, **kwargs)
        operations.append(operation)
        return operation

    monkeypatch.setattr(handler, "_make_operation", record_operation)
    worker = Thread(target=handler.run, daemon=True)
    worker.start()
    try:
        handler.queue_in.put(audio(samples=4 * 1024 * 1024))
        assert connected.wait(3)
        with wave.open(io.BytesIO(operations[0].wav_bytes)) as wav:
            assert wav.getnframes() == 4 * 1024 * 1024
            assert wav.getsampwidth() == 2
        assert operations[0].timeout_s == 60
        handler.queue_in.put(SESSION_END)
        assert handler.queue_out.get(timeout=1) == SESSION_END
        handler.queue_in.put(audio())  # Reuse the same turn/revision in a new session.
        if cancel_at == "connection_handoff":
            assert first_cancel_consumed.wait(1)
        result = handler.queue_out.get(timeout=3)
        assert isinstance(result, Transcription)
        assert result.text == "new session"
        assert len(operations) == 2
        assert operations[0]._cancel_reason == "session_end"
        assert operations[0]._worker_loop is None
        assert operations[0]._worker_task is None
        assert operations[0]._worker_cancel_scope is None
        assert sockets[0].fileno() == -1
        assert handler.queue_out.empty()
    finally:
        stop_server()
        handler.stop_event.set()
        handler.queue_in.put(PIPELINE_END)
        worker.join(timeout=3)
    assert not worker.is_alive()


class ControlledOperation:
    def __init__(self, text="final", *, ignore_cancel=False, error=None):
        self.text = text
        self.ignore_cancel = ignore_cancel
        self.error = error
        self.started = Event()
        self.release = Event()
        self.cancelled = Event()
        self.finished = Event()
        self.cancel_reason = None

    def cancel(self, reason="superseded"):
        self.cancel_reason = self.cancel_reason or reason
        self.cancelled.set()
        if not self.ignore_cancel:
            self.release.set()

    def run(self, cancel_check=lambda: False):
        self.started.set()
        try:
            while not self.release.wait(0.01):
                if cancel_check():
                    self.cancel()
            if self.cancelled.is_set() and not self.ignore_cancel:
                raise stt_module.TranscriptionRequestCancelled(self.cancel_reason)
            if self.error is not None:
                raise self.error
            return stt_module.HttpTranscriptionResult(text=self.text)
        finally:
            self.finished.set()


@pytest.fixture
def handler_factory(monkeypatch):
    handlers = []
    monkeypatch.setattr(stt_module.OpenAICompatibleSTTHandler, "warmup", lambda self: None)

    def make(*operations, tracker=None):
        handler = stt_module.OpenAICompatibleSTTHandler(
            Event(), queue_in=Queue(), queue_out=Queue(), setup_kwargs={"speculative_turns": tracker}
        )
        pending = iter(operations)
        monkeypatch.setattr(handler, "_make_operation", lambda audio, **kwargs: next(pending))
        handlers.append((handler, operations))
        return handler

    yield make
    for handler, operations in handlers:
        for operation in operations:
            operation.release.set()
        handler.cleanup()


def audio(mode="final", *, turn="turn-1", revision=0, samples=160):
    return VADAudio(audio=np.zeros(samples, dtype=np.float32), mode=mode, turn_id=turn, turn_revision=revision)


@pytest.mark.parametrize("mode", ["final", "progressive"])
def test_worker_start_failure_does_not_block_pipeline_end(handler_factory, monkeypatch, caplog, mode):
    class FailingWorker(Thread):
        def start(self):
            raise RuntimeError("sensitive startup detail")

    monkeypatch.setattr(stt_module, "Thread", FailingWorker)
    handler = handler_factory()
    source = audio(mode, revision=3)
    handler.queue_in.put(source)
    handler.queue_in.put(PIPELINE_END)

    handler.run()

    if mode == "final":
        failure = handler.queue_out.get_nowait()
        assert isinstance(failure, TranscriptionFailure)
        assert failure.message == "transcription worker could not start"
        assert failure.turn_id == source.turn_id
        assert failure.turn_revision == source.turn_revision
        assert failure.speech_stopped_at_s == source.created_at_s
    assert handler.queue_out.get_nowait() == PIPELINE_END
    assert handler.queue_out.empty()
    assert not handler._pending_finals
    assert handler._pending_progressive is None
    assert not handler._workers_running
    assert getattr(handler, f"_{mode}_thread") is None
    assert "sensitive startup detail" not in caplog.text


def test_final_worker_can_start_after_an_earlier_start_failure(handler_factory, monkeypatch):
    class FailsOnceWorker(Thread):
        failed = False

        def start(self):
            if not self.failed:
                type(self).failed = True
                raise RuntimeError("cannot start worker")
            super().start()

    monkeypatch.setattr(stt_module, "Thread", FailsOnceWorker)
    operation = ControlledOperation("later final")
    operation.release.set()
    handler = handler_factory(operation)

    assert list(handler.process(audio(turn="failed"))) == []
    failure = handler.queue_out.get_nowait()
    assert isinstance(failure, TranscriptionFailure)
    assert failure.turn_id == "failed"
    assert list(handler.process(audio(turn="later"))) == []
    result = handler.queue_out.get(timeout=1)
    assert isinstance(result, Transcription)
    assert result.turn_id == "later"
    assert result.text == "later final"
    handler._final_thread.join(timeout=1)
    assert not handler._final_thread.is_alive()
    assert handler.queue_out.empty()


def test_final_stt_does_not_block_session_end_and_old_results_cannot_leak(handler_factory):
    old = ControlledOperation("old session", ignore_cancel=True)
    new = ControlledOperation("new session")
    new.release.set()
    handler = handler_factory(old, new)
    worker = Thread(target=handler.run, daemon=True)
    worker.start()
    try:
        handler.queue_in.put(audio())
        assert old.started.wait(1)
        handler.queue_in.put(audio(turn="old-pending"))
        handler.queue_in.put(SESSION_END)
        assert handler.queue_out.get(timeout=1) == SESSION_END
        assert old.cancelled.is_set()
        assert not old.finished.is_set()
        # Reusing the same turn identifiers must not make the old result current.
        handler.queue_in.put(audio())
        old.release.set()
        output = handler.queue_out.get(timeout=1)
        assert isinstance(output, Transcription)
        assert output.text == "new session"
        assert handler.queue_out.empty()
    finally:
        old.release.set()
        handler.stop_event.set()
        handler.queue_in.put(PIPELINE_END)
        worker.join(timeout=2)
    assert not worker.is_alive()


@pytest.mark.parametrize("failure", [False, True])
def test_teardown_fences_results_paused_immediately_before_publication(handler_factory, monkeypatch, failure):
    operation = ControlledOperation(error=stt_module.TranscriptionRequestError("request failed") if failure else None)
    operation.release.set()
    handler = handler_factory(operation)
    reached = Event()
    resume = Event()
    publish = handler._publish_output

    def paused(request, output):
        reached.set()
        assert resume.wait(2)
        return publish(request, output)

    monkeypatch.setattr(handler, "_publish_output", paused)
    try:
        assert list(handler.process(audio())) == []
        assert reached.wait(1)
        handler.on_session_end()
        handler.queue_out.put(SESSION_END)
        resume.set()
        handler._final_thread.join(timeout=1)
        assert not handler._final_thread.is_alive()
        assert handler.queue_out.get_nowait() == SESSION_END
        assert handler.queue_out.empty()
    finally:
        resume.set()


def test_teardown_during_audio_encoding_prevents_http_dispatch(handler_factory, monkeypatch):
    operation = ControlledOperation()
    handler = handler_factory(operation)
    encoding = Event()
    resume = Event()

    def encode(audio, **kwargs):
        encoding.set()
        assert resume.wait(2)
        return operation

    monkeypatch.setattr(handler, "_make_operation", encode)
    try:
        assert list(handler.process(audio())) == []
        assert encoding.wait(1)
        handler.on_session_end()
        resume.set()
        handler._final_thread.join(timeout=1)
        assert not handler._final_thread.is_alive()
        assert operation.cancelled.is_set()
        assert not operation.started.is_set()
        assert handler.queue_out.empty()
    finally:
        resume.set()


def test_new_revision_cancels_active_final_before_another_request_arrives(handler_factory):
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn-1", 0)
    operation = ControlledOperation()
    handler = handler_factory(operation, tracker=tracker)
    assert list(handler.process(audio())) == []
    assert operation.started.wait(1)
    tracker.observe("turn-1", 1)
    assert operation.cancelled.wait(1)
    handler._final_thread.join(timeout=1)
    assert not handler._final_thread.is_alive()
    assert handler.queue_out.empty()


def test_final_cancels_progressive_without_waiting_for_its_result(handler_factory):
    progressive = ControlledOperation("partial", ignore_cancel=True)
    final = ControlledOperation("final")
    final.release.set()
    handler = handler_factory(progressive, final)
    assert list(handler.process(audio("progressive"))) == []
    assert progressive.started.wait(1)
    assert list(handler.process(audio())) == []
    assert progressive.cancelled.wait(1)
    output = handler.queue_out.get(timeout=1)
    assert isinstance(output, Transcription)
    assert output.text == "final"
    assert not progressive.finished.is_set()
    progressive.release.set()
    handler._progressive_thread.join(timeout=1)
    assert handler.queue_out.empty()


def test_final_requests_for_distinct_turns_keep_their_order(handler_factory):
    first, second = ControlledOperation("first"), ControlledOperation("second")
    second.release.set()
    handler = handler_factory(first, second)
    assert list(handler.process(audio(turn="first"))) == []
    assert first.started.wait(1)
    assert list(handler.process(audio(turn="second"))) == []
    assert not second.started.is_set()
    first.release.set()
    outputs = [handler.queue_out.get(timeout=1), handler.queue_out.get(timeout=1)]
    assert [output.text for output in outputs] == ["first", "second"]


def test_pending_final_limit_rejects_overflow_and_preserves_accepted_order(handler_factory):
    active = ControlledOperation("active")
    accepted = [ControlledOperation(f"pending-{index}") for index in range(8)]
    later = ControlledOperation("later")
    for operation in [*accepted, later]:
        operation.release.set()
    handler = handler_factory(active, *accepted, later)
    assert list(handler.process(audio(turn="active"))) == []
    assert active.started.wait(1)

    sources = [audio(turn=f"pending-{index}", revision=2) for index in range(100)]
    for source in sources:
        assert list(handler.process(source)) == []
    assert len(handler._pending_finals) == 8
    assert all(not operation.started.is_set() for operation in accepted)
    failures = [handler.queue_out.get_nowait() for _ in sources[8:]]
    assert all(isinstance(failure, TranscriptionFailure) for failure in failures)
    assert [failure.turn_id for failure in failures] == [source.turn_id for source in sources[8:]]
    assert all(failure.message == "transcription queue is full" for failure in failures)
    assert all(failure.turn_revision == 2 for failure in failures)
    assert [failure.speech_stopped_at_s for failure in failures] == [source.created_at_s for source in sources[8:]]
    assert handler.queue_out.empty()

    # Saturation is local to this pipeline, even with identical endpoint credentials.
    independent = ControlledOperation("independent")
    independent.release.set()
    other_handler = handler_factory(independent)
    assert other_handler.endpoint_url == handler.endpoint_url
    assert other_handler.api_key == handler.api_key
    assert list(other_handler.process(audio())) == []
    assert other_handler.queue_out.get(timeout=1).text == "independent"
    assert not active.finished.is_set()

    active.release.set()
    outputs = [handler.queue_out.get(timeout=1) for _ in range(9)]
    assert all(isinstance(output, Transcription) for output in outputs)
    assert [output.turn_id for output in outputs] == ["active", *[source.turn_id for source in sources[:8]]]
    assert [output.text for output in outputs] == ["active", *[operation.text for operation in accepted]]
    handler._final_thread.join(timeout=1)
    assert not handler._final_thread.is_alive()
    assert handler.queue_out.empty()
    assert not later.started.is_set()
    assert list(handler.process(audio(turn="later"))) == []
    assert handler.queue_out.get(timeout=1).text == "later"


def test_stale_pending_final_releases_space_before_queue_limit_is_checked(handler_factory, monkeypatch):
    tracker = SpeculativeTurnTracker()
    active, latest = ControlledOperation("active"), ControlledOperation("latest")
    latest.release.set()
    handler = handler_factory(active, latest, tracker=tracker)
    monkeypatch.setattr(handler, "_MAX_PENDING_FINAL_REQUESTS", 1)
    assert list(handler.process(audio(turn="active"))) == []
    assert active.started.wait(1)
    assert list(handler.process(audio(turn="pending"))) == []
    tracker.observe("pending", 1)
    assert list(handler.process(audio(turn="pending", revision=1))) == []
    assert len(handler._pending_finals) == 1
    assert handler.queue_out.empty()
    active.release.set()
    outputs = [handler.queue_out.get(timeout=1), handler.queue_out.get(timeout=1)]
    assert [(output.turn_id, output.turn_revision) for output in outputs] == [("active", 0), ("pending", 1)]
    assert outputs[1].text == "latest"


def test_session_end_fences_overflow_failure_waiting_to_publish(handler_factory, monkeypatch):
    active = ControlledOperation()
    handler = handler_factory(active)
    monkeypatch.setattr(handler, "_MAX_PENDING_FINAL_REQUESTS", 1)
    assert list(handler.process(audio(turn="active"))) == []
    assert active.started.wait(1)
    assert list(handler.process(audio(turn="pending"))) == []
    waiting, resume = Event(), Event()
    publish = handler._publish_output

    def pause_failure(request, output):
        if isinstance(output, TranscriptionFailure):
            waiting.set()
            resume.wait()
        return publish(request, output)

    monkeypatch.setattr(handler, "_publish_output", pause_failure)
    submitter = Thread(target=lambda: list(handler.process(audio(turn="overflow"))), daemon=True)
    submitter.start()
    try:
        assert waiting.wait(1)
        handler.on_session_end()
        assert not handler._pending_finals
        resume.set()
        submitter.join(timeout=1)
        handler._final_thread.join(timeout=1)
        assert not submitter.is_alive()
        assert not handler._final_thread.is_alive()
        assert handler.queue_out.empty()
    finally:
        resume.set()
        submitter.join(timeout=1)


def test_final_failure_is_delivered_asynchronously(handler_factory):
    operation = ControlledOperation(error=stt_module.TranscriptionRequestError("transcription request timed out"))
    handler = handler_factory(operation)
    assert list(handler.process(audio())) == []
    assert operation.started.wait(1)
    assert handler.queue_out.empty()
    operation.release.set()
    output = handler.queue_out.get(timeout=1)
    assert isinstance(output, TranscriptionFailure)
    assert output.message == "transcription request timed out"


def test_shutdown_cancels_active_request_and_rejects_later_work(handler_factory):
    operation = ControlledOperation()
    handler = handler_factory(operation)
    assert list(handler.process(audio())) == []
    assert operation.started.wait(1)
    handler.cleanup()
    assert operation.cancelled.is_set()
    assert not handler._final_thread.is_alive()
    assert list(handler.process(audio(turn="later"))) == []
    assert handler.queue_out.empty()


def test_final_discards_pending_progressive_and_rejects_later_windows(handler_factory):
    progressive = ControlledOperation("partial", ignore_cancel=True)
    final = ControlledOperation("final")
    unused = ControlledOperation("obsolete pending")
    handler = handler_factory(progressive, final, unused)
    assert list(handler.process(audio("progressive"))) == []
    assert progressive.started.wait(1)
    assert list(handler.process(audio("progressive", samples=320))) == []
    assert list(handler.process(audio())) == []
    assert final.started.wait(1)
    assert list(handler.process(audio("progressive", samples=480))) == []
    final.release.set()
    output = handler.queue_out.get(timeout=1)
    assert isinstance(output, Transcription)
    assert output.text == "final"
    progressive.release.set()
    handler._progressive_thread.join(timeout=1)
    assert not handler._progressive_thread.is_alive()
    assert not unused.started.is_set()
    assert handler.queue_out.empty()


def test_new_revision_replaces_old_active_and_pending_progressive_work(handler_factory):
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn-1", 0)
    old = ControlledOperation("old", ignore_cancel=True)
    latest = ControlledOperation("latest")
    latest.release.set()
    handler = handler_factory(old, latest, tracker=tracker)
    assert list(handler.process(audio("progressive"))) == []
    assert old.started.wait(1)
    assert list(handler.process(audio("progressive", samples=320))) == []
    tracker.observe("turn-1", 1)
    assert list(handler.process(audio("progressive", revision=1, samples=480))) == []
    assert old.cancelled.wait(1)
    old.release.set()
    output = handler.queue_out.get(timeout=1)
    assert output.text == "latest"
    assert output.turn_revision == 1
    handler._progressive_thread.join(timeout=1)
    assert not handler._progressive_thread.is_alive()
    assert handler.queue_out.empty()


def test_teardown_does_not_wait_for_a_completion_reopen_gate(handler_factory, monkeypatch):
    operation = ControlledOperation()
    operation.release.set()
    handler = handler_factory(operation)
    waiting = Event()
    resume = Event()
    teardown_done = Event()

    def wait_for_reopen(output):
        waiting.set()
        assert resume.wait(2)
        return True

    def teardown():
        handler.on_session_end()
        teardown_done.set()

    monkeypatch.setattr(handler, "should_emit_output", wait_for_reopen)
    assert list(handler.process(audio())) == []
    assert waiting.wait(1)
    worker = Thread(target=teardown, daemon=True)
    worker.start()
    try:
        assert teardown_done.wait(1)
    finally:
        resume.set()
        worker.join(timeout=1)
        handler._final_thread.join(timeout=1)
    assert handler.queue_out.empty()


def test_session_end_cancels_real_stalled_http(stalled_stt_endpoint, monkeypatch):
    url, received, closed = stalled_stt_endpoint
    monkeypatch.setattr(stt_module.OpenAICompatibleSTTHandler, "warmup", lambda self: None)
    handler = stt_module.OpenAICompatibleSTTHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_kwargs={"base_url": url.removesuffix("/audio/transcriptions"), "timeout": 10},
    )
    worker = Thread(target=handler.run, daemon=True)
    worker.start()
    try:
        handler.queue_in.put(audio())
        assert received.wait(2)
        handler.queue_in.put(SESSION_END)
        assert handler.queue_out.get(timeout=1) == SESSION_END
        assert closed.wait(1)
        handler._final_thread.join(timeout=1)
        assert not handler._final_thread.is_alive()
        assert handler.queue_out.empty()
    finally:
        handler.stop_event.set()
        handler.queue_in.put(PIPELINE_END)
        worker.join(timeout=2)
    assert not worker.is_alive()
