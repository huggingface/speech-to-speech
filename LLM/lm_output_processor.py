"""
LLM Output Processor

Intercepts LLM output to:
1. Extract tool calls and send them via text_output_queue
2. Forward clean text to TTS pipeline
"""

import logging
from baseHandler import BaseHandler
from pipeline_messages import MessageTag

logger = logging.getLogger(__name__)


class LMOutputProcessor(BaseHandler):
    """
    Processes LLM output to extract tool calls and forward clean text to TTS.

    Input: (text, language_code, tools) tuples from LLM
    Output: (text, language_code) tuples to TTS
    Side effect: Sends {"type": "assistant_text", "text": ..., "tools": ...} to text_output_queue
    """

    def setup(self, text_output_queue=None):
        """
        Initialize the processor.

        Args:
            text_output_queue: Queue to send text messages and tool calls
        """
        self.text_output_queue = text_output_queue

    def process(self, lm_output):
        """
        Process LLM output: send text/tools to WebSocket, forward clean text to TTS.

        Args:
            lm_output: Tuple of (text, language_code, tools, runtime_config, response)

        Yields:
            Tuple of (text, language_code, runtime_config, response) for TTS
        """
        sentinel, *_ = lm_output

        if sentinel == MessageTag.TOKEN_USAGE:
            _, input_tokens, output_tokens = lm_output
            if self.text_output_queue is not None:
                self.text_output_queue.put({
                    "type": "token_usage",
                    "input_tokens": input_tokens or 0,
                    "output_tokens": output_tokens or 0,
                })
            return

        if sentinel == MessageTag.END_OF_RESPONSE:
            yield (MessageTag.END_OF_RESPONSE, None)
            return

        text_chunk, language_code, tools, runtime_config, response = lm_output

        logger.debug(f"LM processor: text='{text_chunk}', tools={tools}")

        if self.text_output_queue is not None:
            message = {
                "type": "assistant_text",
                "text": text_chunk
            }
            if tools:
                message["tools"] = tools
                logger.info(f"Sending to clients: text='{text_chunk}', tools={[t['name'] for t in tools]}")
            else:
                logger.debug(f"Sending to clients: text='{text_chunk}' (no tools)")
            self.text_output_queue.put(message)

        if text_chunk:
            logger.debug(f"Forwarding to TTS: '{text_chunk}'")
            yield (text_chunk, language_code, runtime_config, response)
