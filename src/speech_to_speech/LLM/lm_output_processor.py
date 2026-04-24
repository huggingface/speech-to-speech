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

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.events import AssistantTextEvent, TokenUsageEvent
from speech_to_speech.pipeline.handler_types import LLMOut, TTSIn
from speech_to_speech.pipeline.messages import EndOfResponse, LLMResponseChunk, TokenUsage, TTSInput
from speech_to_speech.pipeline.queue_types import TextEventItem

logger = logging.getLogger(__name__)


class LMOutputProcessor(BaseHandler[LLMOut, TTSIn]):
    """
    Processes LLM output to extract tool calls and forward clean text to TTS.

    Input: :class:`LLMResponseChunk`, :class:`TokenUsage`, or :class:`EndOfResponse` from LLM
    Output: :class:`TTSInput` or :class:`EndOfResponse` to TTS
    Side effect: Sends :class:`AssistantTextEvent` / :class:`TokenUsageEvent` to text_output_queue
    """

    def setup(self, text_output_queue: Queue[TextEventItem] | None = None) -> None:
        """
        Initialize the processor.

        Args:
            text_output_queue: Queue to send text messages and tool calls
        """
        self.text_output_queue = text_output_queue

    def process(self, lm_output: LLMOut) -> Iterator[TTSIn]:
        """
        Process LLM output: send text/tools to WebSocket, forward clean text to TTS.

        Yields:
            :class:`TTSInput` or :class:`EndOfResponse` for TTS
        """
        if isinstance(lm_output, TokenUsage):
            if self.text_output_queue is not None:
                self.text_output_queue.put(
                    TokenUsageEvent(
                        input_tokens=lm_output.input_tokens or 0,
                        output_tokens=lm_output.output_tokens or 0,
                    )
                )
            return

        if isinstance(lm_output, EndOfResponse):
            yield EndOfResponse()
            return

        if not isinstance(lm_output, LLMResponseChunk):
            logger.warning("LMOutputProcessor received unexpected type: %s", type(lm_output))
            return

        logger.debug(f"LM processor: text='{lm_output.text}', tools={lm_output.tools}")

        if self.text_output_queue is not None:
            event = AssistantTextEvent(text=lm_output.text)
            if lm_output.tools:
                event.tools = lm_output.tools
                logger.info(
                    f"Sending to clients: text='{lm_output.text}', tools={[t['name'] for t in lm_output.tools]}"
                )
            else:
                logger.debug(f"Sending to clients: text='{lm_output.text}' (no tools)")
            self.text_output_queue.put(event)

        if lm_output.text:
            logger.debug(f"Forwarding to TTS: '{lm_output.text}'")
            yield TTSInput(
                text=lm_output.text,
                language_code=lm_output.language_code,
                runtime_config=lm_output.runtime_config,
                response=lm_output.response,
            )
