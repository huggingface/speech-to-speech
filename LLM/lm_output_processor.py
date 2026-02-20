"""
LLM Output Processor

Intercepts LLM output to:
1. Extract tool calls and send them via text_output_queue
2. Forward clean text to TTS pipeline
"""

import logging
from baseHandler import BaseHandler

logger = logging.getLogger(__name__)


class LMOutputProcessor(BaseHandler):
    """
    Processes LLM output to extract tool calls and forward clean text to TTS.

    Input: (text, language_code, tools) tuples from LLM
    Output: (text, language_code) tuples to TTS
    Side effect: Sends {"type": "assistant_text", "text": ..., "tools": ...} to text_output_queue
    """

    def setup(self, text_output_queue):
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
            lm_output: Tuple of (text, language_code, tools)

        Yields:
            Tuple of (text, language_code) for TTS
        """
        text_chunk, language_code, tools = lm_output
        logger.debug(f"LM processor: text='{text_chunk}', tools={tools}")

        # Send text + tools to WebSocket clients
        if tools:
            message = {
                "type": "assistant_text",
                "text": text_chunk,
                "tools": tools
            }
            logger.info(f"Sending to clients: text='{text_chunk}', tools={[t['name'] for t in tools]}")
            self.text_output_queue.put(message)
        else:
            message = {
                "type": "assistant_text",
                "text": text_chunk
            }
            logger.debug(f"Sending to clients: text='{text_chunk}' (no tools)")
            self.text_output_queue.put(message)

        # Forward clean text to TTS (yield to maintain streaming)
        logger.debug(f"Forwarding to TTS: '{text_chunk}'")
        yield (text_chunk, language_code)
