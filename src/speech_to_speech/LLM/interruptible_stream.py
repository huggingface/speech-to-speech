"""Interrupt an HTTP/1 response without racing its return to the connection pool."""

from collections.abc import Iterator
from socket import SHUT_RDWR
from threading import Lock
from typing import Any

import httpx


class InterruptibleStream(httpx.SyncByteStream):
    def __init__(self, response: httpx.Response) -> None:
        assert isinstance(response.stream, httpx.SyncByteStream)
        self._stream = response.stream
        self._lock = Lock()
        self._closed = False
        self._interrupted = False
        self._socket: Any = None
        # HTTP/2 multiplexes responses on one socket; never shut down siblings.
        if response.http_version in {"HTTP/1.0", "HTTP/1.1"}:
            network_stream = response.extensions.get("network_stream")
            if network_stream is not None:
                self._socket = network_stream.get_extra_info("socket")
        response.stream = self

    def __iter__(self) -> Iterator[bytes]:
        yield from self._stream

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._stream.close()

    def abort(self) -> None:
        """Wake the read; its owner closes the response after joining the reader."""
        # Normal SDK closure uses this same lock. Once it releases a completed
        # connection to the pool, a late abort must not touch that socket again.
        with self._lock:
            if self._closed or self._interrupted:
                return
            self._interrupted = True
            if self._socket is None:
                # Custom transports and HTTP/2 can only close this response.
                self._closed = True
                self._stream.close()
            else:
                try:
                    self._socket.shutdown(SHUT_RDWR)
                except OSError:
                    pass  # Already disconnected by the provider.
                # Keep the descriptor open until the reader wakes. Closing it
                # immediately can strand another thread in select() on Unix.
