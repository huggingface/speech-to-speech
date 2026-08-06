import logging
import threading
from collections.abc import Sequence
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ThreadManager:
    """
    Manages multiple threads used to execute given handler tasks.
    """

    def __init__(self, handlers: Sequence[Any], cleanup_callbacks: Sequence[Callable[[], None]] = ()) -> None:
        self.handlers = list(handlers)
        self.threads: list[threading.Thread] = []
        self.cleanup_callbacks = list(cleanup_callbacks)
        self._cleanup_lock = threading.Lock()
        self._cleaned_up = False
        self._cleanup_deferred = False
        self._thread_state_lock = threading.Lock()
        self._thread_states: dict[threading.Thread, str] = {}
        self._unstarted_handlers = list(self.handlers)
        self._lifecycle_lock = threading.Lock()
        self._lifecycle_state = "new"

    def _cleanup(self) -> None:
        with self._cleanup_lock:
            if self._cleaned_up:
                return
            self._cleaned_up = True
        first_error: BaseException | None = None
        with self._thread_state_lock:
            unstarted_handlers = list(reversed(self._unstarted_handlers))
            self._unstarted_handlers.clear()
        for handler in unstarted_handlers:
            cleanup = getattr(handler, "cleanup", None)
            if cleanup is None:
                continue
            try:
                cleanup()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        for callback in reversed(self.cleanup_callbacks):
            try:
                callback()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        with self._lifecycle_lock:
            self._lifecycle_state = "stopped"
        if first_error is not None:
            raise first_error

    def cleanup(self) -> None:
        """Run lifecycle callbacks once, including before threads are started."""

        with self._lifecycle_lock:
            if self._lifecycle_state in {"starting", "running", "stopping"}:
                raise RuntimeError("Cannot clean up while handler threads may be running; call stop() instead.")
            if self._lifecycle_state == "new":
                self._lifecycle_state = "stopping"
        self._cleanup()

    def _cleanup_safely(self, message: str) -> None:
        try:
            self._cleanup()
        except BaseException:
            logger.exception(message)

    def _run_handler(self, handler: Any) -> None:
        thread = threading.current_thread()
        with self._thread_state_lock:
            if self._thread_states[thread] == "cancelled":
                return
            self._thread_states[thread] = "started"
            self._unstarted_handlers.remove(handler)
        handler.run()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._lifecycle_state != "new":
                raise RuntimeError("ThreadManager.start() may only be called once and before cleanup.")
            self._lifecycle_state = "starting"
        try:
            for handler in self.handlers:
                with self._lifecycle_lock:
                    if self._lifecycle_state != "starting":
                        raise RuntimeError("ThreadManager startup was interrupted by shutdown.")
                    thread = threading.Thread(target=self._run_handler, args=(handler,))
                    thread.daemon = False  # Ensure threads are waited for on shutdown
                    with self._thread_state_lock:
                        self._thread_states[thread] = "pending"
                    self.threads.append(thread)
                    try:
                        thread.start()
                    except BaseException:
                        with self._thread_state_lock:
                            pending = self._thread_states[thread] == "pending"
                            launched = thread.ident is not None
                            cancelled = pending and not launched
                            if pending:
                                self._thread_states[thread] = "cancelled" if cancelled else "started"
                        if cancelled:
                            self.threads.remove(thread)
                        raise
            with self._lifecycle_lock:
                if self._lifecycle_state == "starting":
                    self._lifecycle_state = "running"
        except BaseException:
            with self._lifecycle_lock:
                if self._lifecycle_state == "starting":
                    self._lifecycle_state = "stopping"
            self._stop_and_cleanup("Failed to clean up resources after thread startup failed")
            raise

    def wait(self) -> None:
        with self._lifecycle_lock:
            if self._lifecycle_state == "stopped":
                return
            self._lifecycle_state = "stopping"
        try:
            for thread in self.threads:
                thread.join()
        except BaseException:
            self._stop_and_cleanup("Failed to clean up resources after thread wait failed")
            raise
        self._cleanup()

    def _stop_handlers_and_join(self) -> bool:
        all_stopped = True
        for handler in self.handlers:
            try:
                handler.stop_event.set()
            except BaseException:
                all_stopped = False
                logger.exception("Failed to signal handler to stop")

        for i, thread in enumerate(self.threads):
            try:
                with self._thread_state_lock:
                    state = self._thread_states.get(thread, "started")
                    if state == "pending":
                        state = "started" if thread.ident is not None else "cancelled"
                        self._thread_states[thread] = state
                if state == "cancelled":
                    continue
                if thread.is_alive():
                    thread.join(timeout=5.0)
                    if thread.is_alive():
                        all_stopped = False
                        logger.warning(f"Thread {i} ({thread.name}) did not terminate within timeout")
            except BaseException:
                all_stopped = False
                logger.exception("Failed while waiting for thread %d (%s) to stop", i, thread.name)
        return all_stopped

    def _defer_cleanup_until_threads_finish(self, message: str) -> None:
        with self._cleanup_lock:
            if self._cleaned_up or self._cleanup_deferred:
                return
            self._cleanup_deferred = True

        reaper = threading.Thread(
            target=self._wait_for_threads_and_cleanup,
            args=(message,),
            name="speech-to-speech-cleanup",
            daemon=False,
        )
        try:
            reaper.start()
        except BaseException:
            logger.exception("Unable to start cleanup reaper; waiting for handler threads synchronously")
            self._wait_for_threads_and_cleanup(message)

    def _wait_for_threads_and_cleanup(self, message: str) -> None:
        try:
            for thread in self.threads:
                with self._thread_state_lock:
                    if self._thread_states.get(thread, "started") == "cancelled":
                        continue
                thread.join()
        except BaseException:
            with self._cleanup_lock:
                self._cleanup_deferred = False
            logger.exception("Unable to wait for handler threads; shared resources remain open")
            return
        self._cleanup_safely(message)

    def _stop_and_cleanup(self, message: str) -> None:
        if self._stop_handlers_and_join():
            self._cleanup_safely(message)
        else:
            self._defer_cleanup_until_threads_finish(message)

    def stop(self) -> None:
        with self._lifecycle_lock:
            if self._lifecycle_state == "stopped":
                return
            self._lifecycle_state = "stopping"
        self._stop_and_cleanup("Failed to clean up resources while stopping threads")
