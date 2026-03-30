from time import perf_counter
from queue import Empty
import logging

from pipeline_control import SESSION_END, is_control_message

logger = logging.getLogger(__name__)


class BaseHandler:
    """
    Base class for pipeline parts. Each part of the pipeline has an input and an output queue.
    The `setup` method along with `setup_args` and `setup_kwargs` can be used to address the specific requirements of the implemented pipeline part.
    To stop a handler properly, set the stop_event and, to avoid queue deadlocks, place b"END" in the input queue.
    Objects placed in the input queue will be processed by the `process` method, and the yielded results will be placed in the output queue.
    `SESSION_END` is a soft control message used to reset per-session state without stopping the handler thread.
    The cleanup method handles stopping the handler, and b"END" is placed in the output queue.
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

            if isinstance(input, bytes) and input == b"END":
                # sentinelle signal to avoid queue deadlock
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
        self.queue_out.put(b"END")

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

    def _coalesce_pending_tts_input(self, current_input):
        """Combine already-queued text chunks before the next TTS synthesis call.

        This does not wait for more text. It only drains items that are already
        sitting in the input queue when the TTS handler becomes ready again.

        Returns:
            Tuple of (coalesced_input, saw_end_of_response)
        """
        if not hasattr(self.queue_in, "mutex") or not hasattr(self.queue_in, "queue"):
            return current_input, False

        def _decode(item):
            if isinstance(item, tuple):
                if item and item[0] == "__END_OF_RESPONSE__":
                    return None, None, True
                if len(item) == 2 and isinstance(item[0], str):
                    return item[0], item[1], False
            elif isinstance(item, str):
                return item, None, False
            return None, None, False

        text, language_code, _ = _decode(current_input)
        if text is None:
            return current_input, False

        parts = [text.strip()] if text and text.strip() else []
        saw_end_of_response = False

        # Queue.Queue has no public peek API. We inspect the protected deque
        # under its mutex so we can stop before consuming control messages.
        with self.queue_in.mutex:
            while self.queue_in.queue:
                next_item = self.queue_in.queue[0]
                if is_control_message(next_item, SESSION_END.kind):
                    break
                if isinstance(next_item, bytes) and next_item == b"END":
                    break

                next_text, next_language_code, is_end = _decode(next_item)
                if is_end:
                    self.queue_in.queue.popleft()
                    saw_end_of_response = True
                    break
                if next_text is None:
                    break
                if (
                    language_code is not None
                    and next_language_code is not None
                    and next_language_code != language_code
                ):
                    break

                self.queue_in.queue.popleft()
                if next_text.strip():
                    parts.append(next_text.strip())
                if language_code is None:
                    language_code = next_language_code

        combined_text = " ".join(parts).strip()
        if isinstance(current_input, tuple):
            return (combined_text, language_code), saw_end_of_response
        return combined_text, saw_end_of_response
