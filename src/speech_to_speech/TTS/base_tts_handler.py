from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.pipeline.turn_latency import TTSResponseTimer, TurnLatencyStore

logger = logging.getLogger(__name__)


class _TurnCommitDropped(BaseException):
    """Internal control-flow abort when a speculative turn drops before first audio commit."""


class BaseTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """Shared base for TTS handlers: speculative turn gating and response latency bookkeeping."""

    # Handlers assign these in setup() or are wired after construction.
    cancel_scope: CancelScope | None = None
    turn_latency_store: TurnLatencyStore | None = None
    speculative_turns: SpeculativeTurnTracker | None = None
    _active_tts_input: TTSInput | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        raw_process = cls.__dict__.get("process")
        if raw_process is None or getattr(raw_process, "_tts_scoped", False):
            return

        @functools.wraps(raw_process)
        def _scoped_process(self: BaseTTSHandler, tts_input: TTSIn) -> Iterator[TTSOut]:
            self._active_tts_input = tts_input if isinstance(tts_input, TTSInput) else None
            try:
                yield from raw_process(self, tts_input)
            except _TurnCommitDropped:
                return
            finally:
                self._active_tts_input = None

        _scoped_process._tts_scoped = True  # type: ignore[attr-defined]
        setattr(cls, "process", _scoped_process)

    def _gate_end_of_response(self, end: EndOfResponse) -> bool:
        """Wait for the speculative turn gate and say whether to forward the end marker.

        A superseded turn with a ``response_key`` still forwards the marker, flagged
        ``cleanup_only``, so downstream state is released. Without a key it is dropped.
        """
        tracker = self.speculative_turns
        if tracker and not tracker.wait_for_gate(end.turn_id, end.turn_revision):
            if end.response_key is None:
                return False
            end.cleanup_only = True
        return True

    def _admit_input(self, tts_input: TTSInput, *, commit: bool = True, hold_for_grace: bool = True) -> bool:
        """Wait for the speculative turn gate and say whether to synthesize ``tts_input``.

        ``commit=True`` marks the turn as having produced audio. Speculative handlers
        pass ``commit=False, hold_for_grace=False`` and commit on first audio instead.
        """
        tracker = self.speculative_turns
        if tracker is None:
            return True
        admitted = (
            tracker.wait_for_gate(tts_input.turn_id, tts_input.turn_revision, commit=commit)
            if hold_for_grace
            else tracker.wait_for_gate(tts_input.turn_id, tts_input.turn_revision, hold_for_grace=False, commit=commit)
        )
        if not admitted:
            logger.debug("Dropping stale TTS input for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision)
            return False
        return True

    def _commit_first_audio(
        self,
        tts_input: TTSInput | None,
        *,
        generation: int | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> bool:
        """Wait for the reopen grace / Smart Turn gate and commit before yielding first audio."""
        effective_gen = generation
        if effective_gen is None and tts_input is not None:
            effective_gen = tts_input.cancel_generation
        if effective_gen is None and self.cancel_scope is not None:
            effective_gen = self.cancel_scope.generation
        stop_event = getattr(self, "stop_event", None)

        def _aborted() -> bool:
            return (
                (stop_event is not None and stop_event.is_set())
                or self._is_stale(effective_gen, tts_input)
                or (
                    tts_input is not None
                    and tts_input.cancel_generation is not None
                    and self._is_stale(tts_input.cancel_generation, tts_input)
                )
                or (cancelled is not None and cancelled())
            )

        if _aborted():
            return False
        tracker = self.speculative_turns
        if (
            tracker is not None
            and tts_input is not None
            and tts_input.turn_id is not None
            and tts_input.turn_revision is not None
        ):
            if not tracker.wait_for_gate(
                tts_input.turn_id,
                tts_input.turn_revision,
                commit=True,
                should_abort=_aborted,
            ):
                logger.debug(
                    "Dropping speculative TTS audio before commit for turn=%s rev=%s",
                    tts_input.turn_id,
                    tts_input.turn_revision,
                )
                return False
        return not _aborted()

    def _first_audio_emitter(
        self,
        tts_input: TTSInput | None,
        timer: TTSResponseTimer,
        generation: int | None = None,
        *,
        cancelled: Callable[[], bool] | None = None,
        on_first: Callable[[], None] | None = None,
    ) -> Callable[[np.ndarray], Iterator[np.ndarray]]:
        """Return an ``emit(block)`` helper that commits on first audio or aborts cleanly."""
        committed = False

        def emit(block: np.ndarray) -> Iterator[np.ndarray]:
            nonlocal committed
            if not committed:
                if not self._commit_first_audio(tts_input, generation=generation, cancelled=cancelled):
                    raise _TurnCommitDropped
                committed = True
                if on_first is not None:
                    on_first()
            timer.first_yield()
            yield block

        return emit

    def _is_stale(self, generation: int | None, tts_input: TTSInput | None = None) -> bool:
        """True once an interruption or speculative turn reopen supersedes this work."""
        if generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation):
            return True
        active = tts_input if tts_input is not None else self._active_tts_input
        if active is not None and active.turn_id is not None and active.turn_revision is not None:
            is_latest = getattr(self.speculative_turns, "is_latest", None)
            if callable(is_latest) and not is_latest(active.turn_id, active.turn_revision):
                return True
        return False

    def _latency_timer(
        self,
        tts_input: TTSInput | None,
        generation: int | None = None,
        *,
        also_cancelled: Callable[[], bool] | None = None,
    ) -> TTSResponseTimer:
        """Start timing one synthesis; call right before provider work begins.

        Output counts as cancelled once ``generation`` is stale, the speculative
        turn is superseded, or when ``also_cancelled`` returns True.
        """

        def cancelled() -> bool:
            return self._is_stale(generation, tts_input) or (also_cancelled is not None and also_cancelled())

        return TTSResponseTimer(
            self.turn_latency_store,
            tts_input.response_key if tts_input is not None else None,
            tts_input.speech_stopped_at_s if tts_input is not None else None,
            cancelled,
        )
