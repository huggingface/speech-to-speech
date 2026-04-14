from time import perf_counter
from queue import Empty
import logging

from pipeline_control import SESSION_END, is_control_message
from pipeline_messages import PIPELINE_END

logger = logging.getLogger(__name__)


class BaseHandler:
    """
    Base class for pipeline parts. Each part of the pipeline has an input and an output queue.
    The `setup` method along with `setup_args` and `setup_kwargs` can be used to address the specific requirements of the implemented pipeline part.
    To stop a handler properly, set the stop_event and, to avoid queue deadlocks, place PIPELINE_END in the input queue.
    Objects placed in the input queue will be processed by the `process` method, and the yielded results will be placed in the output queue.
    `SESSION_END` is a soft control message used to reset per-session state without stopping the handler thread.
    The cleanup method handles stopping the handler, and PIPELINE_END is placed in the output queue.
    """

    def __init__(self, stop_event, queue_in, queue_out, setup_args=(), setup_kwargs={}):
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.setup(*setup_args, **setup_kwargs)
        self._times = []

    def setup(self, *arg, **kwargs):
        pass

    def process(self):
        raise NotImplementedError

    def run(self):
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
    def last_time(self):
        return self._times[-1]
    
    @property
    def min_time_to_debug(self):
        return 0.001

    def cleanup(self):
        pass

    def on_session_end(self):
        pass
