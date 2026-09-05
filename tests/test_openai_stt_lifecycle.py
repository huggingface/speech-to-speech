from __future__ import annotations

import socket
from threading import Event, Thread

import pytest

from speech_to_speech.STT import openai_compatible_handler as stt_module


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
