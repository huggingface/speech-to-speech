from __future__ import annotations

import logging
from queue import Empty, Queue
from threading import Event
from time import perf_counter
from typing import Any, Generic, Iterator, TypeVar

from speech_to_speech.pipeline.control import SESSION_END, is_control_message
from speech_to_speech.pipeline.messages import PIPELINE_END

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseHandler(Generic[T]):
    """
    Base class for pipeline parts. Each part of the pipeline has an input and an output queue.
    The `setup` method along with `setup_args` and `setup_kwargs` can be used to address the specific requirements of the implemented pipeline part.
    To stop a handler properly, set the stop_event and, to avoid queue deadlocks, place PIPELINE_END in the input queue.
    Objects placed in the input queue will be processed by the `process` method, and the yielded results will be placed in the output queue.
    `SESSION_END` is a soft control message used to reset per-session state without stopping the handler thread.
    The cleanup method handles stopping the handler, and PIPELINE_END is placed in the output queue.
    """

    def __init__(
        self,
        stop_event: Event,
        queue_in: Queue[Any],
        queue_out: Queue[Any],
        setup_args: tuple[Any, ...] = (),
        setup_kwargs: dict[str, Any] = {},
    ) -> None:
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.setup(*setup_args, **setup_kwargs)
        self._times: list[float] = []

    def setup(self, *arg: Any, **kwargs: Any) -> None:
        pass

    def process(self, input: T) -> Iterator:
        raise NotImplementedError

    def run(self) -> None:
        logger.debug(f"{self.__class__.__name__}: Handler thread started")
        while not self.stop_event.is_set():
            try:
                # Use timeout to check stop_event periodically
                input = self.queue_in.get(timeout=0.1)
            except Empty:
                continue

            if is_control_message(input, SESSION_END.kind):
                logger.debug(f"{self.__class__.__name__}: session end received")
                try:
                    self.on_session_end()
                except Exception as e:
                    logger.error(
                        f"{self.__class__.__name__}: Error in on_session_end(): {type(e).__name__}: {e}",
                        exc_info=True,
                    )
                self.queue_out.put(input)
                continue

            if isinstance(input, bytes) and input == PIPELINE_END:
                # sentinel signal to avoid queue deadlock
                logger.debug("Stopping thread")
                break
            start_time = perf_counter()
            try:
                for output in self.process(input):
                    self._times.append(perf_counter() - start_time)
                    if self.last_time > self.min_time_to_debug:
                        logger.debug(f"{self.__class__.__name__}: {self.last_time: .3f} s")
                    self.queue_out.put(output)
                    start_time = perf_counter()
            except Exception as e:
                logger.error(f"{self.__class__.__name__}: Error in process(): {type(e).__name__}: {e}", exc_info=True)

        self.cleanup()
        self.queue_out.put(PIPELINE_END)

    @property
    def last_time(self) -> float:
        return self._times[-1]

    @property
    def min_time_to_debug(self) -> float:
        return 0.001

    def cleanup(self) -> None:
        pass

    def on_session_end(self) -> None:
        pass
