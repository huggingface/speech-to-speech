from __future__ import annotations

import logging
from queue import Queue
from threading import Event
from typing import Iterator

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.events import (
    AudioInputCompletedEvent,
    PartialTranscriptionEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
)
from speech_to_speech.pipeline.handler_types import LLMIn, STTOut
from speech_to_speech.pipeline.messages import (
    PartialTranscription,
    Transcription,
    TranscriptionFailure,
    VADAudio,
)
from speech_to_speech.pipeline.queue_types import TextEventItem
from speech_to_speech.pipeline.transcript_logging import transcript_for_log

logger = logging.getLogger(__name__)


class TranscriptionNotifier(BaseHandler[STTOut, LLMIn]):
    """Sits between STT and LLM.

    It emits protocol-neutral transcription events on ``text_output_queue``.
    ``RealtimeService`` consumes those events, updates conversation state, and
    creates LLM requests.
    """

    def setup(
        self,
        text_output_queue: Queue[TextEventItem] | None = None,
        should_listen: Event | None = None,
    ) -> None:
        self.text_output_queue = text_output_queue
        self.should_listen = should_listen

    def process(self, transcription: STTOut) -> Iterator[LLMIn]:
        if isinstance(transcription, VADAudio):
            if self.text_output_queue is not None:
                self.text_output_queue.put(
                    AudioInputCompletedEvent(
                        audio=transcription.audio,
                        audio_sample_rate=16000,
                        audio_duration_s=len(transcription.audio) / 16000,
                        turn_id=transcription.turn_id,
                        turn_revision=transcription.turn_revision,
                        speech_stopped_at_s=transcription.created_at_s,
                    )
                )
            return
        if isinstance(transcription, PartialTranscription):
            if self.text_output_queue and transcription.text:
                self.text_output_queue.put(
                    PartialTranscriptionEvent(
                        delta=str(transcription.text),
                        turn_id=transcription.turn_id,
                        turn_revision=transcription.turn_revision,
                    )
                )
                logger.debug("Partial transcription: %s", transcript_for_log(transcription.text))
            return

        if isinstance(transcription, TranscriptionFailure):
            if self.text_output_queue is not None:
                self.text_output_queue.put(
                    TranscriptionFailedEvent(
                        message=transcription.message,
                        turn_id=transcription.turn_id,
                        turn_revision=transcription.turn_revision,
                    )
                )
            return

        if isinstance(transcription, Transcription):
            text = transcription.text
            language_code = transcription.language_code
            turn_id = transcription.turn_id
            turn_revision = transcription.turn_revision
            speech_stopped_at_s = transcription.speech_stopped_at_s
        else:
            text = transcription
            language_code = None
            turn_id = None
            turn_revision = None
            speech_stopped_at_s = None

        transcript = str(text)
        # Always close the client-visible transcription item. Empty final STT
        # results should not trigger the LLM, but clients may already have
        # received partial deltas and still need a completed event.
        if self.text_output_queue is not None:
            self.text_output_queue.put(
                TranscriptionCompletedEvent(
                    transcript=transcript,
                    language_code=language_code,
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                    speech_stopped_at_s=speech_stopped_at_s,
                )
            )

        if not transcript:
            logger.debug("Transcription completed with empty transcript")
            if self.should_listen is not None:
                self.should_listen.set()
                logger.debug("Empty transcription completed; listening re-enabled")
            return

        if language_code:
            logger.info("Transcription completed (language=%s): %s", language_code, transcript_for_log(transcript))
        else:
            logger.info("Transcription completed: %s", transcript_for_log(transcript))

        yield from ()
