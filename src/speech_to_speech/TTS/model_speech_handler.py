from __future__ import annotations

import logging
from threading import Event
from time import perf_counter
from typing import Any, Iterator

import numpy as np

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.console import console
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.pipeline.turn_latency import record_response_e2e

logger = logging.getLogger(__name__)


class ModelSpeechHandler(BaseHandler[TTSIn, TTSOut]):
    """Plays the speech a speaking LLM generated, in place of synthesis (``--tts none``)."""

    def setup(
        self,
        should_listen: Event,
        blocksize: int = 512,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.should_listen = should_listen
        self.blocksize = blocksize
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self._reset_response()

    def _reset_response(self) -> None:
        # Samples short of a full block wait for the next chunk of the same response.
        self._pending = np.empty(0, dtype="<i2")
        self._response_key: str | None = None

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = self.speculative_turns
        if isinstance(tts_input, EndOfResponse):
            tail = self._pending if tts_input.response_key == self._response_key else self._pending[:0]
            self._reset_response()
            if speculative_turns and not speculative_turns.wait_for_gate(tts_input.turn_id, tts_input.turn_revision):
                if tts_input.response_key is None:
                    return
                tts_input.cleanup_only = True
            elif tail.size:
                yield np.pad(tail, (0, self.blocksize - tail.size))
            yield AUDIO_RESPONSE_DONE
            return

        if tts_input.audio is None:
            if tts_input.text and (
                speculative_turns is None or speculative_turns.is_latest(tts_input.turn_id, tts_input.turn_revision)
            ):
                console.print(f"[green]ASSISTANT: {tts_input.text}")
            return
        if speculative_turns and not speculative_turns.wait_for_gate(
            tts_input.turn_id,
            tts_input.turn_revision,
            commit=True,
        ):
            logger.debug("Dropping stale model speech for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision)
            return
        gen = tts_input.cancel_generation
        if tts_input.response_key != self._response_key:
            self._reset_response()
            self._response_key = tts_input.response_key
            if tts_input.speech_stopped_at_s is not None:
                record_response_e2e(
                    getattr(self, "turn_latency_store", None),
                    tts_input.response_key,
                    max(0.0, perf_counter() - tts_input.speech_stopped_at_s),
                )
        pending = np.concatenate((self._pending, tts_input.audio))
        usable = pending.size - pending.size % self.blocksize
        for start in range(0, usable, self.blocksize):
            if gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen):
                self._reset_response()
                return
            yield pending[start : start + self.blocksize].copy()
        self._pending = pending[usable:]

    def on_session_end(self) -> None:
        self._reset_response()
