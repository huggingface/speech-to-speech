from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Condition

from openai import OpenAI


class SharedOpenAIClient:
    """Runtime-owned client, with leases for work that outlives a pipeline worker.

    Foreground workers are joined by the runtime before close. Prefetch and
    compaction workers borrow the client for their entire provider operation.
    Closing rejects new background work and waits for existing leases.
    """

    def __init__(self, client: OpenAI) -> None:
        self.client = client
        self._condition = Condition()
        self._borrowers = 0
        self._closing = False
        self._closed = False

    @contextmanager
    def borrow(self) -> Iterator[OpenAI]:
        with self._condition:
            if self._closing:
                raise RuntimeError("The shared LLM client is shutting down.")
            self._borrowers += 1
        try:
            yield self.client
        finally:
            with self._condition:
                self._borrowers -= 1
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            if self._closing:
                self._condition.wait_for(lambda: self._closed)
                return
            self._closing = True
            self._condition.wait_for(lambda: self._borrowers == 0)
        try:
            self.client.close()
        finally:
            with self._condition:
                self._closed = True
                self._condition.notify_all()
