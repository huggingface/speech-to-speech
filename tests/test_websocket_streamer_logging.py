import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from connections.websocket_streamer import _WebSocketHandshakeNoiseFilter


class _InvalidHandshake(Exception):
    pass


def _record(message, exc=None):
    return logging.LogRecord(
        name="websockets.server",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=(type(exc), exc, None) if exc is not None else None,
    )


def test_handshake_noise_filter_suppresses_empty_probe_disconnect():
    exc = _InvalidHandshake("did not receive a valid HTTP request")
    exc.__cause__ = EOFError("connection closed while reading HTTP request line")

    assert _WebSocketHandshakeNoiseFilter().filter(
        _record("opening handshake failed", exc)
    ) is False


def test_handshake_noise_filter_keeps_other_handshake_errors():
    exc = _InvalidHandshake("did not receive a valid HTTP request")
    exc.__cause__ = ValueError("bad request bytes")

    assert _WebSocketHandshakeNoiseFilter().filter(
        _record("opening handshake failed", exc)
    ) is True
    assert _WebSocketHandshakeNoiseFilter().filter(
        _record("different message", exc)
    ) is True
