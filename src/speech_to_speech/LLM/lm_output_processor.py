"""
LLM Output Processor

Intercepts LLM output to:
1. Extract tool calls and send them via text_output_queue
2. Forward clean text to TTS pipeline
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from queue import Queue
from uuid import uuid4

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.events import (
    AssistantResponseDoneEvent,
    AssistantTextEvent,
    ResponseFailedEvent,
    TokenUsageEvent,
)
from speech_to_speech.pipeline.handler_types import LLMOut, TTSIn
from speech_to_speech.pipeline.messages import (
    AssistantTextPart,
    AssistantToolCallPart,
    EndOfResponse,
    LLMResponseChunk,
    TokenUsage,
    TTSInput,
)
from speech_to_speech.pipeline.queue_types import TextEventItem
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.utils.utils import response_wants_audio

logger = logging.getLogger(__name__)


class LMOutputProcessor(BaseHandler[LLMOut, TTSIn]):
    """
    Processes LLM output to extract tool calls and forward clean text to TTS.

    Input: :class:`LLMResponseChunk`, :class:`TokenUsage`, or :class:`EndOfResponse` from LLM
    Output: :class:`TTSInput` or :class:`EndOfResponse` to TTS
    Side effect: Sends :class:`AssistantTextEvent` / :class:`TokenUsageEvent` to text_output_queue
    """

    def setup(
        self,
        text_output_queue: Queue[TextEventItem] | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        """
        Initialize the processor.

        Args:
            text_output_queue: Queue to send text messages and tool calls
        """
        self.text_output_queue = text_output_queue
        self.speculative_turns = speculative_turns
        self._response_key: str | None = None
        self._assistant_output_ordinal: int | None = None
        self._next_assistant_output_ordinal = 0

    def _start_response(self, response_key: str | None) -> str:
        key = response_key or self._response_key or uuid4().hex
        if key != self._response_key:
            self._response_key = key
            self._assistant_output_ordinal = None
            self._next_assistant_output_ordinal = 0
        return key

    def _reset_response(self) -> None:
        self._response_key = None
        self._assistant_output_ordinal = None
        self._next_assistant_output_ordinal = 0

    def _turn_output_allowed(self, turn_id: str | None, turn_revision: int | None) -> bool:
        if self.speculative_turns is None:
            return True
        return self.speculative_turns.is_latest_after_reopen_grace(turn_id, turn_revision)

    def process(self, lm_output: LLMOut) -> Iterator[TTSIn]:
        """
        Process LLM output: send text/tools to WebSocket, forward clean text to TTS.

        Yields:
            :class:`TTSInput` or :class:`EndOfResponse` for TTS
        """
        if isinstance(lm_output, TokenUsage):
            if not self._turn_output_allowed(
                lm_output.turn_id,
                lm_output.turn_revision,
            ):
                logger.debug(
                    "Dropping stale token usage for turn=%s rev=%s", lm_output.turn_id, lm_output.turn_revision
                )
                return
            usage_response_key = self._start_response(lm_output.response_key)
            if self.text_output_queue is not None:
                self.text_output_queue.put(
                    TokenUsageEvent(
                        input_tokens=lm_output.input_tokens or 0,
                        output_tokens=lm_output.output_tokens or 0,
                        turn_id=lm_output.turn_id,
                        turn_revision=lm_output.turn_revision,
                        cancel_generation=lm_output.cancel_generation,
                        response_key=usage_response_key,
                    )
                )
            return

        if isinstance(lm_output, EndOfResponse):
            response_key = lm_output.response_key or self._response_key
            if not self._turn_output_allowed(
                lm_output.turn_id,
                lm_output.turn_revision,
            ):
                logger.debug(
                    "Dropping stale end-of-response for turn=%s rev=%s",
                    lm_output.turn_id,
                    lm_output.turn_revision,
                )
                self._reset_response()
                return
            # A failed generation (e.g. invalid out-of-band input) closes the response as
            # "failed" via the text side-channel, then falls through to emit the normal
            # EndOfResponse so the audio path still re-enables listening / releases the slot.
            if lm_output.error and self.text_output_queue is not None:
                self.text_output_queue.put(
                    ResponseFailedEvent(
                        message=lm_output.error,
                        turn_id=lm_output.turn_id,
                        turn_revision=lm_output.turn_revision,
                        cancel_generation=lm_output.cancel_generation,
                        response_key=response_key,
                    )
                )
            if not lm_output.error and self.text_output_queue is not None:
                self.text_output_queue.put(
                    AssistantResponseDoneEvent(
                        response_key=response_key,
                        turn_id=lm_output.turn_id,
                        turn_revision=lm_output.turn_revision,
                        cancel_generation=lm_output.cancel_generation,
                    )
                )
            yield EndOfResponse(
                turn_id=lm_output.turn_id,
                turn_revision=lm_output.turn_revision,
                cancel_generation=lm_output.cancel_generation,
                response_key=response_key,
            )
            self._reset_response()
            return

        if not isinstance(lm_output, LLMResponseChunk):
            logger.warning("LMOutputProcessor received unexpected type: %s", type(lm_output))
            return

        if not self._turn_output_allowed(
            lm_output.turn_id,
            lm_output.turn_revision,
        ):
            logger.debug("Dropping stale LLM chunk for turn=%s rev=%s", lm_output.turn_id, lm_output.turn_revision)
            return

        logger.debug("LM processor: parts=%s", lm_output.parts)

        response_key = self._start_response(lm_output.response_key)

        for part in lm_output.parts:
            if isinstance(part, AssistantTextPart):
                if self._assistant_output_ordinal is None:
                    self._assistant_output_ordinal = self._next_assistant_output_ordinal
                    self._next_assistant_output_ordinal += 1
                part.ordinal = self._assistant_output_ordinal
            elif isinstance(part, AssistantToolCallPart):
                self._assistant_output_ordinal = None

        if self.text_output_queue is not None:
            event = AssistantTextEvent(
                parts=lm_output.parts,
                turn_id=lm_output.turn_id,
                turn_revision=lm_output.turn_revision,
                cancel_generation=lm_output.cancel_generation,
                response_key=response_key,
            )
            if lm_output.tools:
                event.tools = lm_output.tools
                logger.info(f"Sending to clients: text='{lm_output.text}', tools={[t.name for t in lm_output.tools]}")
            else:
                logger.debug(f"Sending to clients: text='{lm_output.text}' (no tools)")
            self.text_output_queue.put(event)

        if response_wants_audio(lm_output.response):
            for part in lm_output.parts:
                if not isinstance(part, AssistantTextPart) or not part.text:
                    continue
                logger.debug("Forwarding to TTS: '%s'", part.text)
                yield TTSInput(
                    text=part.text,
                    language_code=lm_output.language_code,
                    runtime_config=lm_output.runtime_config,
                    response=lm_output.response,
                    turn_id=lm_output.turn_id,
                    turn_revision=lm_output.turn_revision,
                    speech_stopped_at_s=lm_output.speech_stopped_at_s,
                    cancel_generation=lm_output.cancel_generation,
                    response_key=response_key,
                    assistant_output_ordinal=part.ordinal,
                )

    def on_session_end(self) -> None:
        self._reset_response()
