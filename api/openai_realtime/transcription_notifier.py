import logging

from baseHandler import BaseHandler

logger = logging.getLogger(__name__)


class TranscriptionNotifier(BaseHandler):
    """
    Sits between STT and LLM.  Passes transcription data through unchanged
    while emitting a ``transcription_completed`` event on ``text_output_queue``
    so the Realtime API can forward it to connected clients.
    """

    def setup(self, text_output_queue=None):
        self.text_output_queue = text_output_queue

    def process(self, transcription):
        text = transcription[0] if isinstance(transcription, tuple) else transcription
        if self.text_output_queue and text:
            self.text_output_queue.put(
                {"type": "transcription_completed", "transcript": str(text)}
            )
            logger.debug(f"Transcription notified: {text!r:.80}")
        yield transcription
