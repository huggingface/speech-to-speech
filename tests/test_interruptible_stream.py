from threading import Event, Thread
from types import SimpleNamespace

import httpx
import pytest

from speech_to_speech.LLM.interruptible_stream import InterruptibleStream


def _stream(*, version=b"HTTP/1.1", close=lambda: None, shutdown=lambda how: None):
    class Body(httpx.SyncByteStream):
        def __iter__(self):
            yield b"body"

        def close(self):
            close()

    network = SimpleNamespace(get_extra_info=lambda name: SimpleNamespace(shutdown=shutdown))
    response = httpx.Response(200, stream=Body(), extensions={"http_version": version, "network_stream": network})
    return response, InterruptibleStream(response)


def test_abort_shuts_down_socket_before_closing_and_is_idempotent():
    calls = []
    response, stream = _stream(close=lambda: calls.append("close"), shutdown=lambda _: calls.append("shutdown"))

    stream.abort()
    assert calls == ["shutdown"]  # Leave the descriptor open until the reader exits.
    response.close()
    stream.abort()

    assert calls == ["shutdown", "close"]


def test_normal_completion_then_abort_does_not_touch_reused_connection():
    calls = []
    response, stream = _stream(close=lambda: calls.append("close"), shutdown=lambda _: calls.append("shutdown"))

    assert list(response.iter_bytes()) == [b"body"]
    stream.abort()

    assert calls == ["close"]


def test_abort_racing_normal_closure_does_not_touch_reused_connection():
    closing, release, abort_started = Event(), Event(), Event()
    calls = []

    def close():
        closing.set()
        assert release.wait(5)
        calls.append("returned connection to pool")

    response, stream = _stream(close=close, shutdown=lambda _: calls.append("shutdown"))

    def abort():
        abort_started.set()
        stream.abort()

    closer, aborter = Thread(target=response.close), Thread(target=abort)
    closer.start()
    try:
        assert closing.wait(5)
        aborter.start()
        assert abort_started.wait(5)
    finally:
        release.set()
        closer.join(5)
        if aborter.ident is not None:
            aborter.join(5)

    assert not closer.is_alive() and not aborter.is_alive()
    assert calls == ["returned connection to pool"]


def test_http2_abort_never_shuts_down_sibling_streams():
    calls = []
    _, stream = _stream(
        version=b"HTTP/2", close=lambda: calls.append("close stream"), shutdown=lambda _: calls.append("shutdown")
    )

    stream.abort()

    assert calls == ["close stream"]


def test_disconnected_socket_still_closes_response():
    def shutdown(_):
        raise OSError("already disconnected")

    closed = Event()
    response, stream = _stream(close=closed.set, shutdown=shutdown)

    stream.abort()
    response.close()

    assert closed.is_set()


def test_failed_close_cannot_trigger_a_late_socket_shutdown():
    calls = []

    def close():
        raise RuntimeError("transport close failed")

    _, stream = _stream(close=close, shutdown=lambda _: calls.append("shutdown"))
    with pytest.raises(RuntimeError, match="transport close failed"):
        stream.close()

    stream.abort()

    assert calls == []
