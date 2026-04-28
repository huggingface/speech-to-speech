from __future__ import annotations

import logging
from queue import Queue
from typing import Iterator, Union

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.LLM.chat import make_user_message
from speech_to_speech.pipeline.events import PartialTranscriptionEvent, TranscriptionCompletedEvent
from speech_to_speech.pipeline.handler_types import LLMIn, STTOut
from speech_to_speech.pipeline.messages import GenerateResponseRequest, PartialTranscription, Transcription
from speech_to_speech.pipeline.queue_types import TextEventItem

logger = logging.getLogger(__name__)


class TranscriptionNotifier(BaseHandler[STTOut, Union[STTOut, LLMIn]]):
    """Sits between STT and LLM.

    For **realtime mode** (no ``runtime_config``): emits transcription events
    on ``text_output_queue`` for protocol translation but yields nothing -- the
    ``RealtimeService`` builds ``GenerateResponseRequest`` directly.

    For **legacy mode** (``runtime_config`` provided): appends the user
    message to ``runtime_config.chat`` and yields a
    ``GenerateResponseRequest`` so the LLM handler receives a uniform input
    type regardless of pipeline mode.
    """

    def setup(
        self,
        text_output_queue: Queue[TextEventItem] | None = None,
        runtime_config: RuntimeConfig | None = None,
    ) -> None:
        self.text_output_queue = text_output_queue
        self.runtime_config = runtime_config

    def process(self, transcription: STTOut) -> Iterator[Union[STTOut, LLMIn]]:
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

        if self.runtime_config is not None and text:
            self.runtime_config.chat.add_item(make_user_message(str(text)))
            yield GenerateResponseRequest(
                runtime_config=self.runtime_config,
                language_code=language_code,
            )
