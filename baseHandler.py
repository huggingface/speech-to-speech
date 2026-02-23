from time import perf_counter
from queue import Empty
import logging

logger = logging.getLogger(__name__)


class BaseHandler:
    """
    Base class for pipeline parts. Each part of the pipeline has an input and an output queue.
    The `setup` method along with `setup_args` and `setup_kwargs` can be used to address the specific requirements of the implemented pipeline part.
    To stop a handler properly, set the stop_event and, to avoid queue deadlocks, place b"END" in the input queue.
    Objects placed in the input queue will be processed by the `process` method, and the yielded results will be placed in the output queue.
    The cleanup method handles stopping the handler, and b"END" is placed in the output queue.
    """

    def __init__(self, stop_event, queue_in, queue_out, setup_args=(), setup_kwargs={}):
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.setup(*setup_args, **setup_kwargs)
        self._times = []
        self._last_timing_log_time = 0.0
        self._last_timing_value = None

    @property
    def timing_log_interval_s(self):
        # Log at most once every 5 seconds per handler
        return 5.0

    @property
    def timing_log_min_s(self):
        # Skip extremely fast no-op logs
        return 0.01

    @property
    def timing_log_change_ratio(self):
        # Log if duration changes >50% vs last logged duration
        return 0.5

    def setup(self):
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

            if isinstance(input, bytes) and input == b"END":
                # sentinelle signal to avoid queue deadlock
                logger.debug("Stopping thread")
                break
            input_start = perf_counter()
            start_time = input_start
            first_output_time = None
            output_count = 0
            try:
                for output in self.process(input):
                    if first_output_time is None:
                        first_output_time = perf_counter() - start_time
                    self._times.append(perf_counter() - start_time)
                    if self.last_time > self.min_time_to_debug:
                        logger.debug(f"{self.__class__.__name__}: {self.last_time: .3f} s")
                    self.queue_out.put(output)
                    output_count += 1
                    start_time = perf_counter()
                total_time = perf_counter() - input_start
                if first_output_time is None:
                    first_output_time = total_time
                now = perf_counter()
                should_log = total_time >= self.timing_log_min_s
                if should_log:
                    if now - self._last_timing_log_time < self.timing_log_interval_s:
                        should_log = False
                    if self._last_timing_value is not None:
                        if self._last_timing_value > 0:
                            change_ratio = abs(total_time - self._last_timing_value) / self._last_timing_value
                            if change_ratio >= self.timing_log_change_ratio:
                                should_log = True
                if should_log:
                    logger.info(
                        f"{self.__class__.__name__}: processed {output_count} output(s) "
                        f"in {total_time:.3f}s (ttfo {first_output_time:.3f}s)"
                    )
                    self._last_timing_log_time = now
                    self._last_timing_value = total_time
            except Exception as e:
                logger.error(f"{self.__class__.__name__}: Error in process(): {type(e).__name__}: {e}", exc_info=True)

        self.cleanup()
        self.queue_out.put(b"END")

    @property
    def last_time(self):
        return self._times[-1]
    
    @property
    def min_time_to_debug(self):
        return 0.001

    def cleanup(self):
        pass
