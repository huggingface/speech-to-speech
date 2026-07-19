"""STT → LLM bridge for the s2mlt (speech → multi-language text) pipeline."""

from __future__ import annotations

import logging
from queue import Queue
from typing import Iterator

from openai.types.realtime import RealtimeSessionCreateRequest
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.LLM.chat import Chat, make_system_message, make_user_message
from speech_to_speech.LLM.translation import build_translation_system_prompt
from speech_to_speech.LLM.utils import resolve_auto_language
from speech_to_speech.pipeline.events import InputTranscriptionDeltaEvent, InputTranscriptionDoneEvent
from speech_to_speech.pipeline.handler_types import LLMIn, STTOut
from speech_to_speech.pipeline.messages import GenerateResponseRequest, PartialTranscription, Transcription
from speech_to_speech.pipeline.queue_types import TextEventItem

logger = logging.getLogger(__name__)


class TranslationNotifier(BaseHandler[STTOut, LLMIn]):
    """Sits between STT and the LLM in the s2mlt pipeline.

    Publishes live transcription snapshots (``input.transcription.delta``) and
    finals (``input.transcription.done``) on ``text_output_queue``, and for
    each non-empty final transcription yields a *stateless* per-segment
    :class:`GenerateResponseRequest`: a fresh chat holding only the translation
    system prompt and the transcript, with ``output_modalities=["text"]`` so
    the LLM handlers stream raw text without any TTS-oriented munging.

    Translation is per segment by design - no conversation history accumulates
    across segments, so a bad segment can never poison later ones.
    """

    def setup(
        self,
        target_languages: list[str],
        text_output_queue: Queue[TextEventItem] | None = None,
    ) -> None:
        self.text_output_queue = text_output_queue
        self.system_prompt = build_translation_system_prompt(target_languages)

    def process(self, transcription: STTOut) -> Iterator[LLMIn]:
        if isinstance(transcription, PartialTranscription):
            if self.text_output_queue is not None and transcription.text:
                self.text_output_queue.put(
                    InputTranscriptionDeltaEvent(
                        text=transcription.text,
                        turn_id=transcription.turn_id,
                        turn_revision=transcription.turn_revision,
                    )
                )
            return

        if not isinstance(transcription, Transcription):
            logger.warning("TranslationNotifier received unexpected type: %s", type(transcription))
            return

        transcript = transcription.text.strip()
        clean_language_code, _ = resolve_auto_language(transcription.language_code)
        if self.text_output_queue is not None:
            self.text_output_queue.put(
                InputTranscriptionDoneEvent(
                    text=transcript,
                    language_code=clean_language_code,
                    turn_id=transcription.turn_id,
                    turn_revision=transcription.turn_revision,
                )
            )

        if not transcript:
            logger.debug("Empty final transcription for turn=%s; no translation requested", transcription.turn_id)
            return

        logger.info(
            "Requesting translation (language=%s, turn=%s rev=%s): %s",
            clean_language_code,
            transcription.turn_id,
            transcription.turn_revision,
            transcript,
        )

        chat = Chat(2)
        chat.add_item(make_system_message(self.system_prompt))
        chat.add_item(make_user_message(transcript))
        yield GenerateResponseRequest(
            runtime_config=RuntimeConfig(chat=chat, session=RealtimeSessionCreateRequest(type="realtime")),
            response=RealtimeResponseCreateParams(output_modalities=["text"]),
            language_code=transcription.language_code,
            turn_id=transcription.turn_id,
            turn_revision=transcription.turn_revision,
            speech_stopped_at_s=transcription.speech_stopped_at_s,
        )
