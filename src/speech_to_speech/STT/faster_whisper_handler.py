from __future__ import annotations

import logging
import os
from typing import Any, Iterator

from faster_whisper import WhisperModel
from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

console = Console()

logger = logging.getLogger(__name__)

# Values that mean "detect the language" rather than naming one. faster-whisper expects
# ``language=None`` for auto-detection and rejects anything it does not recognise.
_AUTO_LANGUAGE_VALUES = {"", "auto", "none", "null"}


class FasterWhisperSTTHandler(BaseSTTHandler):
    """
    Handles the Speech To Text generation using a Whisper model.
    """

    def setup(
        self,
        model_name: str = "tiny.en",
        device: str = "auto",
        compute_type: str = "auto",
        gen_kwargs: dict[str, Any] = {},
    ) -> None:
        self.gen_kwargs = self.adapt_gen_kwargs(gen_kwargs)
        self.start_language = self._normalize_language(self.gen_kwargs.get("language"))
        if self.start_language is None:
            # faster-whisper auto-detects only when no language is passed at all; leaving a
            # sentinel like "auto" in place makes transcribe() raise on an unknown language.
            self.gen_kwargs.pop("language", None)
        else:
            self.gen_kwargs["language"] = self.start_language

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    @staticmethod
    def _normalize_language(language: Any) -> str | None:
        """Return the pinned language code, or ``None`` to auto-detect."""
        if not isinstance(language, str):
            return None
        stripped = language.strip()
        if stripped.lower() in _AUTO_LANGUAGE_VALUES:
            return None
        return stripped

    def _resolve_language(self, info: Any) -> str | None:
        """The language code to report for this turn.

        A pinned language is authoritative because ``transcribe()`` ran with it. Otherwise
        report what faster-whisper detected, tagged ``-auto`` so downstream
        ``resolve_auto_language()`` knows it may change between turns.
        """
        if self.start_language is not None:
            return self.start_language

        detected = getattr(info, "language", None)
        if isinstance(detected, str) and detected:
            probability = getattr(info, "language_probability", None)
            if isinstance(probability, float):
                logger.debug("Faster Whisper detected language %s (p=%.2f)", detected, probability)
            return f"{detected}-auto"
        return None

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        logger.debug("infering faster whisper...")

        segments, info = self.model.transcribe(vad_audio.audio, **self.gen_kwargs)
        output_text = []

        for segment in segments:
            logger.debug("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            output_text.append(segment.text)

        pred_text = " ".join(output_text).strip()
        language_code = self._resolve_language(info)

        logger.debug("finished whisper inference")
        if pred_text:
            console.print(f"[yellow]USER: {pred_text}")

            yield Transcription(
                text=pred_text,
                language_code=language_code,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
                speech_stopped_at_s=vad_audio.created_at_s,
            )
        else:
            logger.debug("no text detected. skipping...")

    def cleanup(self) -> None:
        logger.info("Stopping FasterWhisperSTTHandler")
        del self.model

    def adapt_gen_kwargs(self, gen_kwargs: dict[str, Any]) -> dict[str, Any]:
        gen_kwargs["without_timestamps"] = not gen_kwargs.pop("return_timestamps", True)

        return gen_kwargs
