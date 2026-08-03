from __future__ import annotations

# ruff: noqa: I001

import logging
from queue import Empty, Queue
from threading import Event
from time import perf_counter
from typing import Any, Generic, Iterator, TypeVar, cast

import numpy as np

from speech_to_speech.pipeline.control import PipelineControlMessage, is_control_message, SESSION_END
from speech_to_speech.pipeline.log_context import pipeline_log_ctx
from speech_to_speech.pipeline.messages import PIPELINE_END, AudioOutput, EndOfResponse

logger = logging.getLogger(__name__)

InT = TypeVar("InT")
OutT = TypeVar("OutT")


class BaseHandler(Generic[InT, OutT]):
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
        queue_in: Queue[InT | PipelineControlMessage | bytes],
        queue_out: Queue[OutT | PipelineControlMessage | bytes],
        setup_args: tuple[Any, ...] = (),
        setup_kwargs: dict[str, Any] = {},
    ) -> None:
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.pipeline_index: int | None = None
        self.setup(*setup_args, **setup_kwargs)
        self._times: list[float] = []

    def setup(self, *arg: Any, **kwargs: Any) -> None:
        pass

    def process(self, input: InT) -> Iterator[OutT]:
        raise NotImplementedError

    def should_process_input(self, item: InT) -> bool:
        cancel_scope = getattr(self, "cancel_scope", None)
        cancel_generation = getattr(item, "cancel_generation", None)
        if (
            cancel_scope is not None
            and cancel_generation is not None
            and not isinstance(item, EndOfResponse)
            and cancel_scope.is_stale(cancel_generation)
        ):
            logger.debug(
                "%s: dropping stale input for cancel generation %s",
                self.__class__.__name__,
                cancel_generation,
            )
            return False
        return True

    def should_emit_output(self, output: OutT) -> bool:
        return True

    def before_emit_output(self, output: OutT) -> None:
        pass

    def output_for_queue(self, output: OutT, source_input: InT) -> OutT | AudioOutput:
        cancel_generation = getattr(source_input, "cancel_generation", None)
        if cancel_generation is not None and (isinstance(output, bytes) or hasattr(output, "tobytes")):
            audio = cast(bytes | np.ndarray, output)
            return AudioOutput(audio=audio, cancel_generation=cancel_generation)
        return output

    def run(self) -> None:
        if self.pipeline_index is not None:
            pipeline_log_ctx.set(self.pipeline_index)
        logger.debug(f"{self.__class__.__name__}: Handler thread started")
        while not self.stop_event.is_set():
            try:
                # Use timeout to check stop_event periodically
                item = self.queue_in.get(timeout=0.1)
            except Empty:
                continue

            if isinstance(item, PipelineControlMessage) and is_control_message(item, SESSION_END.kind):
                logger.debug(f"{self.__class__.__name__}: session end received")
                try:
                    self.on_session_end()
                except Exception as e:
                    logger.error(
                        f"{self.__class__.__name__}: Error in on_session_end(): {type(e).__name__}: {e}",
                        exc_info=True,
                    )
                self.queue_out.put(item)
                continue

            if isinstance(item, bytes) and item == PIPELINE_END:
                # sentinel signal to avoid queue deadlock
                logger.debug("Stopping thread")
                break

            if isinstance(item, PipelineControlMessage):
                logger.warning("%s: unexpected control message kind: %s", self.__class__.__name__, item.kind)
                continue

            typed_item = cast(InT, item)
            if not self.should_process_input(typed_item):
                continue

            start_time = perf_counter()
            try:
                for output in self.process(typed_item):
                    if not self.should_emit_output(output):
                        start_time = perf_counter()
                        continue
                    self._times.append(perf_counter() - start_time)
                    if self.should_log_timing(output):
                        logger.log(self.timing_log_level, "%s: %.3f s", self.__class__.__name__, self.last_time)
                    self.before_emit_output(output)
                    queued_output = cast(
                        OutT | PipelineControlMessage | bytes,
                        self.output_for_queue(output, typed_item),
                    )
                    self.queue_out.put(queued_output)
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

    @property
    def timing_log_level(self) -> int:
        return logging.DEBUG

    def should_log_timing(self, output: OutT) -> bool:
        return self.last_time > self.min_time_to_debug

    def cleanup(self) -> None:
        pass

    def on_session_end(self) -> None:
        pass
