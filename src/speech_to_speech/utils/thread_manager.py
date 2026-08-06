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
        self.handlers = handlers
        self.threads: list[threading.Thread] = []
        self.cleanup_callbacks = list(cleanup_callbacks)
        self._cleanup_lock = threading.Lock()
        self._cleaned_up = False

    def _cleanup(self) -> None:
        with self._cleanup_lock:
            if self._cleaned_up:
                return
            self._cleaned_up = True
        first_error: BaseException | None = None
        for callback in reversed(self.cleanup_callbacks):
            try:
                callback()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def cleanup(self) -> None:
        """Run lifecycle callbacks once, including before threads are started."""

        self._cleanup()

    def _cleanup_safely(self, message: str) -> None:
        try:
            self._cleanup()
        except BaseException:
            logger.exception(message)

    def start(self) -> None:
        try:
            for handler in self.handlers:
                thread = threading.Thread(target=handler.run)
                thread.daemon = False  # Ensure threads are waited for on shutdown
                self.threads.append(thread)
                thread.start()
        except BaseException:
            self._stop_handlers_and_join()
            self._cleanup_safely("Failed to clean up resources after thread startup failed")
            raise

    def wait(self) -> None:
        try:
            for thread in self.threads:
                thread.join()
        except BaseException:
            self._stop_handlers_and_join()
            self._cleanup_safely("Failed to clean up resources after thread wait failed")
            raise
        self._cleanup()

    def _stop_handlers_and_join(self) -> None:
        for handler in self.handlers:
            try:
                handler.stop_event.set()
            except BaseException:
                logger.exception("Failed to signal handler to stop")

        for i, thread in enumerate(self.threads):
            try:
                if thread.is_alive():
                    thread.join(timeout=5.0)
                    if thread.is_alive():
                        logger.warning(f"Thread {i} ({thread.name}) did not terminate within timeout")
            except BaseException:
                logger.exception("Failed while waiting for thread %d (%s) to stop", i, thread.name)

    def stop(self) -> None:
        self._stop_handlers_and_join()
        self._cleanup_safely("Failed to clean up resources while stopping threads")
