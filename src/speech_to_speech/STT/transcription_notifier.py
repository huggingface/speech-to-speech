from __future__ import annotations

import logging
from queue import Queue
from typing import Any, Iterator

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.events import PartialTranscriptionEvent, TranscriptionCompletedEvent
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription

logger = logging.getLogger(__name__)


class TranscriptionNotifier(BaseHandler[PartialTranscription | Transcription]):
    """
    Sits between STT and LLM.  Intercepts partial and final transcriptions,
    emitting events on ``text_output_queue`` for connected clients (Realtime
    API or plain WebSocket) while only forwarding final transcripts to the LLM.
    """

    def setup(self, text_output_queue: Queue[Any] | None = None, suppress_yield: bool = False) -> None:
        self.text_output_queue = text_output_queue
        self.suppress_yield = suppress_yield

    def process(self, transcription: PartialTranscription | Transcription) -> Iterator[PartialTranscription | Transcription]:
        if isinstance(transcription, PartialTranscription):
            if self.text_output_queue and transcription.text:
                self.text_output_queue.put(PartialTranscriptionEvent(delta=str(transcription.text)))
                logger.debug("Partial transcription: %s", str(transcription.text)[:80])
            return

        if isinstance(transcription, Transcription):
            text = transcription.text
            language_code = transcription.language_code
        else:
            text = transcription
            language_code = None

        if self.text_output_queue and text:
            self.text_output_queue.put(TranscriptionCompletedEvent(transcript=str(text), language_code=language_code))
            logger.debug("Transcription completed: %s", str(text)[:80])

        if not self.suppress_yield:
            yield transcription
