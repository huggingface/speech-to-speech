import logging
import threading
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)


class ThreadManager:
    """
    Manages multiple threads used to execute given handler tasks.
    """

    def __init__(self, handlers: Sequence[Any]) -> None:
        self.handlers = handlers
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        for handler in self.handlers:
            thread = threading.Thread(target=handler.run)
            thread.daemon = False  # Ensure threads are waited for on shutdown
            self.threads.append(thread)
            thread.start()

    def stop(self) -> None:
        # Signal all handlers to stop
        for handler in self.handlers:
            handler.stop_event.set()

        # Wait for all threads to finish with timeout
        for i, thread in enumerate(self.threads):
            if thread.is_alive():
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logger.warning(f"Thread {i} ({thread.name}) did not terminate within timeout")
