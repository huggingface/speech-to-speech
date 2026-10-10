from __future__ import annotations

import logging
from collections.abc import Sequence
from threading import Event, Lock, Thread
from typing import Any

from speech_to_speech.LLM.shared_client import SharedOpenAIClient
from speech_to_speech.utils.thread_manager import ThreadManager

logger = logging.getLogger(__name__)


class PipelineRuntime:
    """Own the server workers and the selected LLM backend's shared client.

    ThreadManager retains its existing worker timeout. Cleanup runs separately
    and waits for timed-out workers and background client leases, so shutdown
    never closes a connection pool still in use.
    """

    CLEANUP_WAIT_S = 5.0

    def __init__(self, handlers: Sequence[Any], client: SharedOpenAIClient | None = None) -> None:
        self._manager = ThreadManager(handlers)
        self.client_resource = client
        self._lock = Lock()
        self._cleanup_thread: Thread | None = None
        self._started = False
        self._stopped = False
        self._startup_done = Event()
        self._startup_done.set()

    @property
    def handlers(self) -> Sequence[Any]:
        return self._manager.handlers

    @property
    def threads(self) -> list[Thread]:
        return self._manager.threads

    def add_handler(self, handler: Any) -> None:
        with self._lock:
            if self._started or self._stopped:
                raise RuntimeError("Cannot add a handler after the pipeline starts or stops.")
            self._manager.handlers = [*self.handlers, handler]

    def start(self) -> None:
        with self._lock:
            if self._started or self._stopped:
                raise RuntimeError("The pipeline runtime can only be started once.")
            self._started = True
            self._startup_done.clear()
        try:
            self._manager.start()
        except BaseException:
            self._startup_done.set()
            self.stop()
            raise
        finally:
            self._startup_done.set()

    def _join_workers(self) -> None:
        self._startup_done.wait()
        # start() can fail before Thread.start(), leaving an unstarted thread
        # in ThreadManager. Only successfully started workers can be joined.
        for thread in self.threads:
            if thread.ident is not None:
                thread.join()

    def _cleanup(self) -> None:
        self._join_workers()
        if self.client_resource is not None:
            try:
                self.client_resource.close()
            except Exception:
                logger.exception("Shared LLM client cleanup failed")

    def _request_cleanup(self) -> Thread:
        with self._lock:
            self._stopped = True
            if self._cleanup_thread is None:
                self._cleanup_thread = Thread(target=self._cleanup, name="pipeline-resource-cleanup", daemon=True)
                try:
                    self._cleanup_thread.start()
                except BaseException:
                    self._cleanup_thread = None
                    raise
            return self._cleanup_thread

    def _finish_cleanup(self, timeout: float | None) -> None:
        try:
            cleanup = self._request_cleanup()
        except Exception:
            # Thread exhaustion can cause both worker startup and cleanup
            # startup to fail. Safe synchronous cleanup still owns the client.
            logger.exception("Cannot start resource cleanup worker; waiting for cleanup synchronously")
            self._cleanup()
            return
        cleanup.join(timeout=timeout)
        if cleanup.is_alive():
            logger.warning("Shared LLM client cleanup deferred until active work finishes")

    def stop(self) -> None:
        self._manager.stop()
        if self.client_resource is None:
            with self._lock:
                self._stopped = True
            return
        self._finish_cleanup(self.CLEANUP_WAIT_S)

    def wait(self) -> None:
        self._join_workers()
        if self.client_resource is not None:
            self._finish_cleanup(None)
