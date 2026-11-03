"""Parakeet inference in the application's shared ZeroGPU worker."""

import logging
from time import perf_counter

import numpy as np

from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, TranscriptionFailure
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler
from speech_to_speech.zerogpu import runtime

logger = logging.getLogger(__name__)


class ZeroGPUSTTHandler(BaseSTTHandler):
    def setup(self, model_name="nvidia/parakeet-tdt-0.6b-v3", **kwargs):
        if model_name != runtime.get_model_ids()["stt"]:
            raise ValueError("ZeroGPU STT model must match runtime.initialize(stt_model=...).")
        self.start_language = self.last_language = "en"

    def process(self, item):
        started = perf_counter()
        try:
            text = runtime.transcribe(np.asarray(item.audio, dtype=np.float32))
        except Exception:
            logger.exception("ZeroGPU transcription failed")
            if item.mode != "progressive":
                yield TranscriptionFailure(
                    message="ZeroGPU transcription failed",
                    turn_id=item.turn_id,
                    turn_revision=item.turn_revision,
                    speech_stopped_at_s=item.speech_end_at_s,
                )
            return
        if item.mode == "progressive":
            yield PartialTranscription(text=text, turn_id=item.turn_id, turn_revision=item.turn_revision)
        else:
            self._record_final_stt(item, perf_counter() - started)
            yield Transcription(
                text=text,
                language_code="en",
                turn_id=item.turn_id,
                turn_revision=item.turn_revision,
                speech_stopped_at_s=item.speech_end_at_s,
            )
