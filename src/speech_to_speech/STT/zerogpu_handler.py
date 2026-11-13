"""Parakeet inference in the application's shared ZeroGPU worker."""

import logging
from time import perf_counter

import numpy as np

from speech_to_speech.pipeline.language_detection import (
    PARAKEET_TDT_LANGUAGES,
    detect_language_from_text,
    warm_language_detector,
)
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, TranscriptionFailure
from speech_to_speech.speech_languages import SpeechLanguageCapabilities
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler
from speech_to_speech.zerogpu import runtime

logger = logging.getLogger(__name__)


class ZeroGPUSTTHandler(BaseSTTHandler):
    def setup(self, model_name="nvidia/parakeet-tdt-0.6b-v3", **kwargs):
        if model_name != runtime.get_model_ids()["stt"]:
            raise ValueError("ZeroGPU STT model must match runtime.initialize(stt_model=...).")
        self.start_language = self.last_language = None
        self.language_capabilities = SpeechLanguageCapabilities(
            languages=frozenset({"en"} if model_name.endswith("-v2") else PARAKEET_TDT_LANGUAGES),
            # Selection controls the conversation language; Parakeet itself
            # always decodes automatically, just as it does in the native handler.
            supports_language_selection=True,
            supports_auto=not model_name.endswith("-v2"),
        )
        self._language_detector = (
            warm_language_detector(tuple(sorted(self.language_capabilities.languages)))
            if len(self.language_capabilities.languages) > 1
            else None
        )

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
            try:
                detected = (
                    (
                        "en"
                        if self.language_capabilities.languages == frozenset({"en"})
                        else detect_language_from_text(text, self._language_detector)
                    )
                    if text
                    else None
                )
            except Exception:
                logger.exception("Parakeet language detection failed; leaving language unset")
                detected = None
            if detected:
                self.last_language = detected
            yield Transcription(
                text=text,
                language_code=f"{detected}-auto" if detected else None,
                turn_id=item.turn_id,
                turn_revision=item.turn_revision,
                speech_stopped_at_s=item.speech_end_at_s,
            )

    def on_session_end(self):
        super().on_session_end()
        self.last_language = None
