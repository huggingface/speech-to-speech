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

    def _admit_input(self, tts_input: TTSInput, *, commit: bool = True) -> bool:
        """Wait for the speculative turn gate and say whether to synthesize ``tts_input``.

        ``commit=True`` marks the turn as having produced audio. Handlers that commit
        on first audio instead pass ``commit=False``.
        """
        tracker = self.speculative_turns
        if tracker and not tracker.wait_for_gate(tts_input.turn_id, tts_input.turn_revision, commit=commit):
            logger.debug("Dropping stale TTS input for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision)
            return False
        return True

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

        Output counts as cancelled once ``generation`` is stale, or when
        ``also_cancelled`` returns True (for handler-specific conditions).
        """

        def cancelled() -> bool:
            return self._is_stale(generation) or (also_cancelled is not None and also_cancelled())

        return TTSResponseTimer(
            self.turn_latency_store,
            tts_input.response_key if tts_input is not None else None,
            tts_input.speech_stopped_at_s if tts_input is not None else None,
            cancelled,
        )
