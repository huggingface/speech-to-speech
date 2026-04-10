import logging

from baseHandler import BaseHandler
from pipeline_messages import MessageTag

logger = logging.getLogger(__name__)


class TranscriptionNotifier(BaseHandler):
    """
    Sits between STT and LLM.  Intercepts partial and final transcriptions,
    emitting events on ``text_output_queue`` for connected clients (Realtime
    API or plain WebSocket) while only forwarding final transcripts to the LLM.
    """

    def setup(self, text_output_queue=None):
        self.text_output_queue = text_output_queue

    def process(self, transcription):
        if isinstance(transcription, tuple) and len(transcription) == 2 and transcription[0] == MessageTag.PARTIAL:
            _, text = transcription
            if self.text_output_queue and text:
                self.text_output_queue.put(
                    {"type": "partial_transcription", "delta": str(text)}
                )
                logger.debug("Partial transcription: %s", str(text)[:80])
            return

        text = transcription[0] if isinstance(transcription, tuple) else transcription
        if self.text_output_queue and text:
            self.text_output_queue.put(
                {"type": "transcription_completed", "transcript": str(text)}
            )
            logger.debug("Transcription completed: %s", str(text)[:80])
        yield transcription
