from __future__ import annotations

import logging
from typing import Any, Iterator, Optional

import numpy as np
import torch
from lightning_whisper_mlx import LightningWhisperMLX
from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler
from speech_to_speech.utils.mlx_lock import MLXLockContext

logger = logging.getLogger(__name__)

console = Console()

SUPPORTED_LANGUAGES = [
    "en",
    "fr",
    "es",
    "zh",
    "ja",
    "ko",
    "hi",
    "de",
    "pt",
    "pl",
    "it",
    "nl",
]

DEFAULT_LANGUAGE = "en"


class LightningWhisperSTTHandler(BaseSTTHandler):
    """
    Handles the Speech To Text generation using a Whisper model.
    """

    def setup(
        self,
        model_name: str = "distil-large-v3",
        device: str = "mps",
        torch_dtype: str = "float16",
        compile_mode: Optional[str] = None,
        language: Optional[str] = None,
        gen_kwargs: dict[str, Any] = {},
    ) -> None:
        if len(model_name.split("/")) > 1:
            model_name = model_name.split("/")[-1]
        self.device = device
        self.model = LightningWhisperMLX(model=model_name, batch_size=6, quant=None)
        self.start_language = language
        # "auto" is a request to detect, not a language code, so it must never leak into
        # last_language -- it would fail every SUPPORTED_LANGUAGES check downstream.
        self.last_language = language if language != "auto" else None

        self.warmup()

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}")

        # 2 warmup steps for no compile or compile mode with CUDA graphs capture
        n_steps = 1
        dummy_input = np.array([0] * 512)

        for _ in range(n_steps):
            with MLXLockContext(handler_name=self.__class__.__name__):
                _ = self.model.transcribe(dummy_input)["text"].strip()

    def _forced_language(self) -> Optional[str]:
        """The language explicitly requested by the user, if any."""
        if self.start_language and self.start_language != "auto":
            return self.start_language
        return None

    def _resolve_language(self, transcription_dict: dict[str, Any], forced_language: Optional[str]) -> str:
        """Report the language that was actually transcribed.

        A detected language outside ``SUPPORTED_LANGUAGES`` is still reported as-is: the
        transcription is correct, and that list only decides what becomes the sticky
        fallback. Re-transcribing in a different language, or dropping the turn, throws away
        a good result because of a downstream allowlist.
        """
        if forced_language is not None:
            # transcribe() ran with this language, so it is authoritative.
            self.last_language = forced_language
            return forced_language

        detected = transcription_dict.get("language")
        if isinstance(detected, str) and detected:
            if detected in SUPPORTED_LANGUAGES:
                self.last_language = detected
            else:
                logger.warning("Whisper detected unsupported language: %s", detected)
            return detected

        # Only when no language could be determined at all.
        return self.last_language or DEFAULT_LANGUAGE

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        logger.debug("infering whisper...")

        audio = vad_audio.audio
        forced_language = self._forced_language()

        with MLXLockContext(handler_name=self.__class__.__name__):
            if forced_language is not None:
                transcription_dict = self.model.transcribe(audio, language=forced_language)
            else:
                transcription_dict = self.model.transcribe(audio)

        pred_text = (transcription_dict.get("text") or "").strip()
        language_code = self._resolve_language(transcription_dict, forced_language)
        torch.mps.empty_cache()

        logger.debug("finished whisper inference")
        console.print(f"[yellow]USER: {pred_text}")
        logger.debug(f"Language Code Whisper: {language_code}")

        if self.start_language == "auto":
            language_code += "-auto"

        yield Transcription(
            text=pred_text,
            language_code=language_code,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )
