from __future__ import annotations

import logging
from collections.abc import Iterator
from queue import Queue
from time import perf_counter

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.events import AudioInputCompletedEvent
from speech_to_speech.pipeline.handler_types import LLMIn
from speech_to_speech.pipeline.messages import VADAudio
from speech_to_speech.pipeline.queue_types import TextEventItem
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)


class AudioInputNotifier(BaseHandler[VADAudio, LLMIn]):
    """Bridge final VAD audio directly into the LLM stage."""

    def setup(
        self,
        speculative_turns: SpeculativeTurnTracker,
        text_output_queue: Queue[TextEventItem],
        sample_rate: int = 16000,
    ) -> None:
        self.sample_rate = sample_rate
        self.speculative_turns = speculative_turns
        self.text_output_queue = text_output_queue

    def should_process_input(self, item: VADAudio) -> bool:
        if item.mode == "progressive":
            return False
        if item.turn_id is None or item.turn_revision is None:
            return True
        remaining_delay_s = max(0.0, item.processing_delay_s - (perf_counter() - item.created_at_s))
        return self.speculative_turns.is_latest_after_stability_window(
            item.turn_id,
            item.turn_revision,
            remaining_delay_s,
        )

    def process(self, vad_audio: VADAudio) -> Iterator[LLMIn]:
        audio_duration_s = len(vad_audio.audio) / self.sample_rate if self.sample_rate else 0.0
        logger.info(
            "Audio input completed: %.3fs turn=%s rev=%s",
            audio_duration_s,
            vad_audio.turn_id,
            vad_audio.turn_revision,
        )

        self.text_output_queue.put(
            AudioInputCompletedEvent(
                audio=vad_audio.audio,
                audio_sample_rate=self.sample_rate,
                audio_duration_s=audio_duration_s,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
                speech_stopped_at_s=vad_audio.created_at_s,
            )
        )
        # RealtimeService owns conversation state and constructs the request.
        yield from ()
