from __future__ import annotations

from typing import Callable

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.pipeline.turn_latency import TTSResponseTimer, TurnLatencyStore


class BaseTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """Shared base for TTS handlers: response latency bookkeeping."""

    # Handlers assign these in setup() or are wired after construction.
    cancel_scope: CancelScope | None = None
    turn_latency_store: TurnLatencyStore | None = None

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
