from __future__ import annotations

import logging
from typing import Any, Iterator, Optional

import numpy as np
from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler
from speech_to_speech.utils.mlx_lock import MLXLockContext

logger = logging.getLogger(__name__)

console = Console()

DEFAULT_LANGUAGE = "en"

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


class MLXAudioWhisperSTTHandler(BaseSTTHandler):
    """
    Handles the Speech To Text generation using MLX Audio's Whisper implementation.
    Optimized for Apple Silicon using the MLX framework.
    """

    def setup(
        self,
        model_name: str = "mlx-community/whisper-large-v3-turbo",
        language: Optional[str] = None,
        gen_kwargs: dict[str, Any] = {},
    ) -> None:
        from mlx_audio.stt.generate import load_model
        from transformers import WhisperProcessor

        self.model_name = model_name
        self.start_language = language
        # "auto" is a request to detect, not a language code, so it must never leak into
        # last_language -- it would fail every SUPPORTED_LANGUAGES check downstream.
        self.last_language = language if language != "auto" else None
        self.gen_kwargs = gen_kwargs

        # Load the model directly
        logger.info(f"Loading model {model_name}...")
        self.model = load_model(model_name)

        # Check if processor was loaded, if not, load it manually from original model
        if self.model._processor is None:
            logger.info("Processor not found in MLX model, loading from original Whisper model...")
            # Map MLX model names to their original Whisper counterparts
            processor_model_map = {
                "mlx-community/whisper-large-v3-turbo": "openai/whisper-large-v3",
                "mlx-community/whisper-large-v3": "openai/whisper-large-v3",
                "mlx-community/whisper-medium": "openai/whisper-medium",
                "mlx-community/whisper-small": "openai/whisper-small",
                "mlx-community/whisper-base": "openai/whisper-base",
                "mlx-community/whisper-tiny": "openai/whisper-tiny",
            }

            # Get the appropriate processor model name
            processor_model = processor_model_map.get(model_name, "openai/whisper-large-v3")
            logger.info(f"Loading processor from {processor_model}...")

            try:
                self.model._processor = WhisperProcessor.from_pretrained(processor_model)
                logger.info("Processor loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load processor: {e}")
                raise

        logger.info(f"Model {model_name} loaded successfully")

        self.warmup()

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}")

        # Warmup with a dummy input
        dummy_audio = np.zeros(16000, dtype=np.float32)

        try:
            # Pre-warm the model by running a transcription
            with MLXLockContext(handler_name=self.__class__.__name__):
                _ = self.model.generate(dummy_audio, verbose=False)
            logger.info("Model warmed up and ready")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def _forced_language(self) -> Optional[str]:
        """The language explicitly requested by the user, if any."""
        if self.start_language and self.start_language != "auto":
            return self.start_language
        return None

    def _resolve_language(self, result: Any) -> str:
        """Pick the language code to report, updating the sticky fallback on success."""
        forced = self._forced_language()
        if forced is not None:
            # generate() ran with this language, so it is authoritative.
            self.last_language = forced
            return forced

        detected = getattr(result, "language", None)
        if isinstance(detected, str) and detected:
            if detected in SUPPORTED_LANGUAGES:
                self.last_language = detected
                return detected
            logger.warning("Detected unsupported language: %s", detected)

        last_language = self.last_language
        if last_language is not None and last_language in SUPPORTED_LANGUAGES:
            return last_language
        return DEFAULT_LANGUAGE

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        logger.debug("inferring mlx-audio whisper...")

        assert isinstance(vad_audio.audio, np.ndarray), "Audio must be a numpy array"
        audio_input = vad_audio.audio.astype(np.float32)

        # Prepare generation kwargs - only pass valid parameters
        gen_kwargs = {}

        # Add language if specified
        forced_language = self._forced_language()
        if forced_language is not None:
            gen_kwargs["language"] = forced_language

        try:
            # MLX models share a single Metal command queue, so concurrent inference from
            # the STT/LLM/TTS threads aborts the process with
            # "Completed handler provided after commit call". Every other MLX path in the
            # pipeline serializes through this lock; this one must too.
            with MLXLockContext(handler_name=self.__class__.__name__):
                result = self.model.generate(audio_input, verbose=False, **gen_kwargs)

            # Extract text from result
            pred_text = result.text.strip() if hasattr(result, "text") else str(result).strip()
            language_code = self._resolve_language(result)

        except Exception as e:
            logger.error(f"MLX Audio Whisper inference failed: {e}")
            pred_text = ""
            language_code = self.last_language if self.last_language else DEFAULT_LANGUAGE

        logger.debug("finished mlx-audio whisper inference")
        console.print(f"[yellow]USER: {pred_text}")
        logger.debug(f"Language Code: {language_code}")

        if self.start_language == "auto":
            language_code += "-auto"

        yield Transcription(
            text=pred_text,
            language_code=language_code,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )
