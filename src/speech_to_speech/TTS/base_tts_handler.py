from __future__ import annotations

import logging
from typing import Callable

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.pipeline.turn_latency import TTSResponseTimer, TurnLatencyStore

logger = logging.getLogger(__name__)


class BaseTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """Shared base for TTS handlers: speculative turn gating and response latency bookkeeping."""

    # Handlers assign these in setup() or are wired after construction.
    cancel_scope: CancelScope | None = None
    turn_latency_store: TurnLatencyStore | None = None
    speculative_turns: SpeculativeTurnTracker | None = None

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

    def _admit_input(
        self,
        tts_input: TTSInput,
        *,
        commit: bool = True,
        hold_for_grace: bool = True,
    ) -> bool:
        """Wait for the speculative turn gate and say whether to synthesize ``tts_input``.

        ``commit=True`` marks the turn as having produced audio. Handlers that
        synthesize speculatively during the reopen grace pass
        ``commit=False, hold_for_grace=False`` and call :meth:`_commit_first_audio`
        before yielding their first audio block.
        """
        tracker = self.speculative_turns
        if tracker is None:
            return True
        if hold_for_grace:
            admitted = tracker.wait_for_gate(tts_input.turn_id, tts_input.turn_revision, commit=commit)
        else:
            admitted = tracker.wait_for_gate(
                tts_input.turn_id,
                tts_input.turn_revision,
                hold_for_grace=False,
                commit=commit,
            )
        if not admitted:
            logger.debug("Dropping stale TTS input for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision)
            return False
        return True

    def _commit_first_audio(self, tts_input: TTSInput) -> bool:
        """Wait for the reopen grace / Smart Turn gate and commit before yielding first audio."""
        tracker = self.speculative_turns
        if tracker is None or tts_input.turn_id is None or tts_input.turn_revision is None:
            return True
        if not tracker.wait_for_gate(tts_input.turn_id, tts_input.turn_revision, commit=True):
            logger.debug(
                "Dropping speculative TTS audio before commit for turn=%s rev=%s",
                tts_input.turn_id,
                tts_input.turn_revision,
            )
            return False
        return True

    def _effective_cancel_generation(self, tts_input: TTSInput) -> int | None:
        if tts_input.cancel_generation is not None:
            return tts_input.cancel_generation
        return self.cancel_scope.generation if self.cancel_scope is not None else None

    def _is_turn_stale(self, tts_input: TTSInput) -> bool:
        tracker = self.speculative_turns
        if tracker is None or tts_input.turn_id is None or tts_input.turn_revision is None:
            return False
        is_latest = getattr(tracker, "is_latest", None)
        return callable(is_latest) and not is_latest(tts_input.turn_id, tts_input.turn_revision)

    def _is_synthesis_cancelled(self, tts_input: TTSInput | None, generation: int | None) -> bool:
        stop_event = getattr(self, "stop_event", None)
        if stop_event is not None and stop_event.is_set():
            return True
        if self._is_stale(generation):
            return True
        return tts_input is not None and self._is_turn_stale(tts_input)

    def _is_stale(self, generation: int | None) -> bool:
        """True once an interruption has superseded ``generation``."""
        return generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation)

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
            return (
                self._is_stale(generation)
                or (tts_input is not None and self._is_turn_stale(tts_input))
                or (also_cancelled is not None and also_cancelled())
            )

        return TTSResponseTimer(
            self.turn_latency_store,
            tts_input.response_key if tts_input is not None else None,
            tts_input.speech_stopped_at_s if tts_input is not None else None,
            cancelled,
        )
