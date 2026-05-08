from __future__ import annotations

import logging

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)


class BaseSTTHandler(BaseHandler[STTIn, STTOut]):
    """Base STT handler with speculative-turn stale input filtering."""

    speculative_turns: SpeculativeTurnTracker | None = None

    def should_process_input(self, item: STTIn) -> bool:
        return self._is_latest_turn_item(item, wait_for_pending_reopen=True, stage="input")

    def should_emit_output(self, output: STTOut) -> bool:
        return self._is_latest_turn_item(output, wait_for_pending_reopen=True, stage="output")

    def _is_latest_turn_item(self, item: object, *, wait_for_pending_reopen: bool, stage: str) -> bool:
        if self.speculative_turns is None:
            return True
        turn_id = getattr(item, "turn_id", None)
        turn_revision = getattr(item, "turn_revision", None)
        if turn_id is None or turn_revision is None:
            return True

        if wait_for_pending_reopen:
            is_latest = self.speculative_turns.is_latest_after_pending_reopen(turn_id, turn_revision)
        else:
            is_latest = self.speculative_turns.is_latest(turn_id, turn_revision)

        if not is_latest:
            logger.info(
                "%s: dropping stale STT %s for turn=%s rev=%s",
                self.__class__.__name__,
                stage,
                turn_id,
                turn_revision,
            )
        return is_latest
