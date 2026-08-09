"""
LLM Output Processor

Intercepts LLM output to:
1. Preserve text and tool events in model order
2. Forward clean text to TTS in the same queue
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
    PipelineEvent,
    ResponseFailedEvent,
    TokenUsageEvent,
)
from speech_to_speech.pipeline.handler_types import LLMOut, TTSIn
from speech_to_speech.pipeline.messages import (
    AssistantTextPart,
    EndOfResponse,
    LLMResponseChunk,
    TokenUsage,
    TTSInput,
)
from speech_to_speech.pipeline.queue_types import TextEventItem
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.utils.utils import response_wants_audio

logger = logging.getLogger(__name__)


class LMOutputProcessor(BaseHandler[LLMOut, TTSIn | PipelineEvent]):
    """
    Places ordered output events and their TTS inputs on one queue.

    Input: :class:`LLMResponseChunk`, :class:`TokenUsage`, or :class:`EndOfResponse` from LLM
    Output: assistant events, :class:`TTSInput`, or :class:`EndOfResponse` to TTS;
    token usage bypasses TTS through ``text_output_queue``.
    """

    def setup(
        self,
        text_output_queue: Queue[TextEventItem] | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        self.text_output_queue = text_output_queue
        self.speculative_turns = speculative_turns
        self._response_key: str | None = None

    def _start_response(self, response_key: str | None) -> str:
        key = response_key or self._response_key or uuid4().hex
        self._response_key = key
        return key

    def _reset_response(self) -> None:
        self._response_key = None

    def _turn_output_allowed(self, turn_id: str | None, turn_revision: int | None) -> bool:
        if self.speculative_turns is None:
            return True
        return self.speculative_turns.is_latest_after_reopen_grace(turn_id, turn_revision)

    def process(self, lm_output: LLMOut) -> Iterator[TTSIn | PipelineEvent]:
        """
        Forward response events and audio inputs in their original order.

        Yields:
            Response events, :class:`TTSInput`, or :class:`EndOfResponse`
        """
        if isinstance(lm_output, TokenUsage):
            usage_response_key = self._start_response(lm_output.response_key)
            usage_event = TokenUsageEvent(
                input_tokens=lm_output.input_tokens or 0,
                output_tokens=lm_output.output_tokens or 0,
                turn_id=lm_output.turn_id,
                turn_revision=lm_output.turn_revision,
                cancel_generation=lm_output.cancel_generation,
                response_key=usage_response_key,
            )
            if self.text_output_queue is not None:
                self.text_output_queue.put(usage_event)
            else:
                yield usage_event
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
                if response_key is not None:
                    # Bypass downstream speculative gates with a lifecycle-only
                    # terminal. The router uses its key to cancel an opened stale
                    # response or clear only that queued response.
                    yield EndOfResponse(
                        cancel_generation=lm_output.cancel_generation,
                        response_key=response_key,
                        cleanup_only=True,
                    )
                return
            if lm_output.error:
                yield ResponseFailedEvent(
                    message=lm_output.error,
                    turn_id=lm_output.turn_id,
                    turn_revision=lm_output.turn_revision,
                    cancel_generation=lm_output.cancel_generation,
                    response_key=response_key,
                )
            else:
                yield AssistantResponseDoneEvent(
                    response_key=response_key,
                    turn_id=lm_output.turn_id,
                    turn_revision=lm_output.turn_revision,
                    cancel_generation=lm_output.cancel_generation,
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
            event = AssistantTextEvent(
                parts=[part],
                turn_id=lm_output.turn_id,
                turn_revision=lm_output.turn_revision,
                cancel_generation=lm_output.cancel_generation,
                response_key=response_key,
            )
            yield event
            if (
                not isinstance(part, AssistantTextPart)
                or not part.text.strip()
                or not response_wants_audio(lm_output.response)
            ):
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
            )

    def on_session_end(self) -> None:
        self._reset_response()
