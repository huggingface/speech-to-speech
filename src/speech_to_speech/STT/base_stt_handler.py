from __future__ import annotations

import logging
from collections import Counter
from time import perf_counter
from typing import Any

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)


class BaseSTTHandler(BaseHandler[STTIn, STTOut]):
    """Base STT handler with speculative-turn stale input filtering."""

    speculative_turns: SpeculativeTurnTracker | None = None
    final_revision_settle_s: float = 0.6

    def should_process_input(self, item: STTIn) -> bool:
        mode = getattr(item, "mode", None)
        turn_id = getattr(item, "turn_id", None)
        turn_revision = getattr(item, "turn_revision", None)
        wait_for_stability = mode == "final"
        gate_start = perf_counter()
        is_latest = self._is_latest_turn_item(
            item,
            wait_for_pending_reopen=True,
            wait_for_stability=wait_for_stability,
        )
        gate_wait_s = perf_counter() - gate_start
        if gate_wait_s >= 0.05:
            logger.info(
                "%s: STT input gate waited %.3fs for turn=%s rev=%s mode=%s latest=%s age=%.3fs queue=%s",
                self.__class__.__name__,
                gate_wait_s,
                turn_id,
                turn_revision,
                mode,
                is_latest,
                self._item_age_s(item),
                self._safe_qsize(),
            )

        if not is_latest:
            queued_drops = self._drop_stale_queued_inputs()
            self._log_stale_turn_item(item, "input", queued_drops=queued_drops)
            return False
        return True

    def should_emit_output(self, output: STTOut) -> bool:
        if not self._is_latest_turn_item(output, wait_for_pending_reopen=True, wait_for_stability=False):
            self._log_stale_turn_item(output, "output")
            return False
        return True

    def _is_latest_turn_item(
        self,
        item: object,
        *,
        wait_for_pending_reopen: bool,
        wait_for_stability: bool,
    ) -> bool:
        if self.speculative_turns is None:
            return True
        turn_id = getattr(item, "turn_id", None)
        turn_revision = getattr(item, "turn_revision", None)
        if turn_id is None or turn_revision is None:
            return True

        if wait_for_stability:
            is_latest = self.speculative_turns.is_latest_after_stability_window(
                turn_id,
                turn_revision,
                self.final_revision_settle_s,
            )
        elif wait_for_pending_reopen:
            is_latest = self.speculative_turns.is_latest_after_pending_reopen(turn_id, turn_revision)
        else:
            is_latest = self.speculative_turns.is_latest(turn_id, turn_revision)
        return is_latest

    def _drop_stale_queued_inputs(self) -> int:
        if self.speculative_turns is None or not hasattr(self.queue_in, "mutex") or not hasattr(self.queue_in, "queue"):
            return 0

        dropped = 0
        with self.queue_in.mutex:
            kept: list[Any] = []
            while self.queue_in.queue:
                queued_item = self.queue_in.queue.popleft()
                if isinstance(queued_item, VADAudio) and not self._is_latest_turn_item(
                    queued_item,
                    wait_for_pending_reopen=False,
                    wait_for_stability=False,
                ):
                    dropped += 1
                else:
                    kept.append(queued_item)
            self.queue_in.queue.extend(kept)
            if dropped:
                self.queue_in.not_full.notify_all()
        return dropped

    def _log_stale_turn_item(self, item: object, stage: str, *, queued_drops: int = 0) -> None:
        turn_id = getattr(item, "turn_id", None)
        turn_revision = getattr(item, "turn_revision", None)
        if turn_id is None or turn_revision is None:
            return

        if not hasattr(self, "_stale_drop_counts"):
            self._stale_drop_counts: Counter[tuple[str, str, int]] = Counter()
        key = (stage, turn_id, turn_revision)
        self._stale_drop_counts[key] += 1

        message = "%s: dropping stale STT %s for turn=%s rev=%s age=%.3fs"
        args: tuple[object, ...] = (
            self.__class__.__name__,
            stage,
            turn_id,
            turn_revision,
            self._item_age_s(item),
        )
        if queued_drops:
            message += " (+%d queued)"
            args = (*args, queued_drops)

        if self._stale_drop_counts[key] == 1:
            logger.info(message, *args)
        else:
            logger.debug(message, *args)

    def _item_age_s(self, item: object) -> float:
        created_at_s = getattr(item, "created_at_s", None)
        if not isinstance(created_at_s, float):
            return 0.0
        return max(0.0, perf_counter() - created_at_s)

    def _safe_qsize(self) -> int | str:
        try:
            return self.queue_in.qsize()
        except NotImplementedError:
            return "unknown"
