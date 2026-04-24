"""
Kokoro TTS Handler

Supports NVIDIA Kokoro TTS model for high-quality multilingual speech synthesis.
- On Apple Silicon (MPS): Uses mlx-audio with mlx-community/Kokoro-82M-bf16
- On CUDA/CPU: Uses native kokoro library with hexgrad/Kokoro-82M

Model supports 8 languages with multiple voices per language.
"""

from __future__ import annotations

import logging
from sys import platform
from threading import Event
from typing import Any, Iterator, Optional

import numpy as np
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.utils.mlx_lock import MLXLockContext

logger = logging.getLogger(__name__)
console = Console()

# Language code mapping from Whisper/langdetect language codes to Kokoro lang codes
WHISPER_LANGUAGE_TO_KOKORO_LANG = {
    "en": "b",  # British English
    "ja": "j",  # Japanese
    "zh": "z",  # Chinese
    "fr": "f",  # French
    "es": "e",  # Spanish
    "it": "i",  # Italian
    "pt": "p",  # Portuguese
    "hi": "h",  # Hindi
    # Additional European languages from Parakeet - map to closest Kokoro language
    "de": "b",  # German -> British English (no German voice available)
    "nl": "b",  # Dutch -> British English
    "pl": "b",  # Polish -> British English
    "ru": "b",  # Russian -> British English
    "uk": "b",  # Ukrainian -> British English
}

# Default voices for each Kokoro language code
# These are native voices that sound natural for each language
# Voice naming: first letter = language, second letter = gender (f=female, m=male)
# Available voices per language:
#   a (American): af_alloy, af_aoede, af_bella, af_heart, af_jessica, af_kore, af_nicole, af_nova, af_river, af_sarah, af_sky
#                 am_adam, am_echo, am_eric, am_fenrir, am_liam, am_michael, am_onyx, am_puck, am_santa
#   b (British):  bf_alice, bf_emma, bf_isabella, bf_lily, bm_daniel, bm_fable, bm_george, bm_lewis
#   e (Spanish):  ef_dora, em_alex, em_santa
#   f (French):   ff_siwis
#   h (Hindi):    hf_alpha, hf_beta, hm_omega, hm_psi
#   i (Italian):  if_sara, im_nicola
#   j (Japanese): jf_alpha, jf_gongitsune, jf_nezumi, jf_tebukuro, jm_kumo
#   p (Portuguese): pf_dora, pm_alex, pm_santa
#   z (Chinese):  zf_xiaobei, zf_xiaoni, zf_xiaoxiao, zf_xiaoyi, zm_yunjian, zm_yunxi, zm_yunxia, zm_yunyang
KOKORO_LANG_DEFAULT_VOICES = {
    "a": "af_heart",  # American English female
    "b": "bm_fable",  # British English male
    "e": "ef_dora",  # Spanish female
    "f": "ff_siwis",  # French female
    "h": "hf_alpha",  # Hindi female
    "i": "if_sara",  # Italian female
    "j": "jf_alpha",  # Japanese female
    "p": "pf_dora",  # Portuguese female
    "z": "zf_xiaobei",  # Chinese female
}


class KokoroTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """
    Handles Text-to-Speech using Kokoro TTS model.

    On Apple Silicon (MPS): Uses mlx-audio with the MLX-converted model.
    On CUDA/CPU: Uses native kokoro library for optimal performance.

    Kokoro 82M is a 82M parameter multilingual TTS model
    supporting 8 languages with multiple voices per language.
    """

    def setup(
        self,
        should_listen: Event,
        model_name: Optional[str] = None,
        device: str = "auto",
        voice: str = "bm_fable",
        lang_code: str = "b",
        speed: float = 1.0,
        blocksize: int = 512,
        gen_kwargs: dict[str, Any] | None = None,
        cancel_scope: CancelScope | None = None,
    ) -> None:
        """
        Initialize the Kokoro TTS model.

        Args:
            model_name: Model identifier. Defaults are:
                - MPS: "mlx-community/Kokoro-82M-bf16"
                - CUDA/CPU: "hexgrad/Kokoro-82M"
            device: Device to use ("auto", "cuda", "mps", "cpu")
            voice: Voice identifier (e.g. "bm_fable")
            lang_code: Language code (e.g. "b" for British English)
            speed: Speech speed multiplier
            blocksize: Audio chunk size for streaming
            gen_kwargs: Unused, for pipeline compatibility
        """
        self.should_listen = should_listen
        self.voice = voice
        self.lang_code = lang_code
        self.speed = speed
        self.blocksize = blocksize
        self.cancel_scope = cancel_scope

        # Determine device
        if device == "auto":
            if platform == "darwin":
                self.device = "mps"
            else:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Set default model based on device
        if model_name is None:
            if self.device == "mps":
                model_name = "mlx-community/Kokoro-82M-bf16"
            else:
                model_name = "hexgrad/Kokoro-82M"

        self.model_name = model_name

        logger.info(f"Loading Kokoro model: {model_name} on {self.device}")

        if self.device == "mps":
            self._setup_mlx(model_name)
        else:
            self._setup_kokoro(model_name)

        self._initial_voice = self.voice
        self._initial_lang_code = self.lang_code

        self.warmup()

    def _setup_mlx(self, model_name: str) -> None:
        """Setup for Apple Silicon using mlx-audio."""
        try:
            from mlx_audio.tts.utils import load_model

            self.backend = "mlx"
            self.model = load_model(model_name)
            logger.info("MLX Audio Kokoro model loaded successfully")

            # Get or create the pipeline for our language and preload the voice
            # This avoids the voice being reloaded on every generate() call
            self._pipeline = self.model._get_pipeline(self.lang_code)
            self._voice_tensor = self._pipeline.load_voice(self.voice)
            logger.info(f"Preloaded voice: {self.voice}")

            # Preload voices for common languages to avoid download delays during inference
            self._preload_multilingual_voices()
        except ImportError as e:
            message = str(e)
            if "misaki" in message or "espeakng_loader" in message:
                raise ImportError(
                    "Kokoro TTS on Apple Silicon requires additional mlx-audio TTS dependencies. "
                    f"Missing dependency: {message}. "
                    "Install with: pip install misaki espeakng-loader"
                ) from e
            raise ImportError(
                "mlx-audio is required for Kokoro TTS on Apple Silicon. Install with: pip install mlx-audio"
            ) from e

    def _setup_kokoro(self, model_name: str) -> None:
        """Setup for CUDA/CPU using native kokoro library."""
        try:
            from kokoro import KPipeline

            self.backend = "kokoro"
            self.pipeline = KPipeline(lang_code=self.lang_code)
            logger.info("Native Kokoro pipeline loaded successfully")
        except ImportError as e:
            raise ImportError(
                "kokoro is required for Kokoro TTS on CUDA/CPU. "
                "Install with: pip install kokoro>=0.9.2 soundfile\n"
                "Also ensure espeak-ng is installed: apt-get install espeak-ng (Linux) or brew install espeak-ng (macOS)"
            ) from e

    def _preload_multilingual_voices(self) -> None:
        """Preload voices for common languages to avoid download delays during inference."""
        # Only preload a few commonly used language voices to avoid excessive startup time
        # Users speaking other languages will experience a one-time download delay
        # preload_langs = ["a", "b", "e", "f", "h", "i", "j", "p", "z"] # English, Spanish, French, Hindi, Italian, Japanese, Portuguese
        preload_langs = ["a", "e", "f"]  # English, Spanish, French

        for lang_code in preload_langs:
            voice = KOKORO_LANG_DEFAULT_VOICES.get(lang_code)
            if voice:
                try:
                    logger.info(f"Preloading voice for language '{lang_code}': {voice}")
                    pipeline = self.model._get_pipeline(lang_code)
                    pipeline.load_voice(voice)
                except Exception as e:
                    logger.warning(f"Failed to preload voice {voice}: {e}")

    def warmup(self) -> None:
        """Warm up the model with a dummy inference."""
        logger.info(f"Warming up {self.__class__.__name__}")

        # Run a short dummy inference to warm up the model
        if self.backend == "mlx":
            for _ in self.model.generate(
                text="Hello",
                voice=self.voice,
                speed=self.speed,
                lang_code=self.lang_code,
            ):
                pass
        else:
            for _ in self.pipeline("Hello", voice=self.voice, speed=self.speed):
                pass

        logger.info(f"{self.__class__.__name__} warmed up")

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        """
        Process text input and generate audio output.

        Yields:
            Audio chunks as numpy int16 arrays
        """
        if isinstance(tts_input, EndOfResponse):
            yield AUDIO_RESPONSE_DONE
            return

        runtime_config = tts_input.runtime_config
        response = tts_input.response
        language_code = tts_input.language_code
        text = tts_input.text

        voice: Optional[str] = None
        if response and response.audio and response.audio.output and response.audio.output.voice:
            voice = str(response.audio.output.voice)
        if not voice and runtime_config:
            audio_cfg = runtime_config.session.audio
            audio_output = audio_cfg.output if audio_cfg is not None else None
            voice = str(audio_output.voice) if audio_output is not None and audio_output.voice else None
        if voice:
            self.voice = voice

        if self.backend == "mlx":
            yield from self._process_mlx(text, language_code)
        else:
            yield from self._process_kokoro(text, language_code)

        if not runtime_config:
            self.should_listen.set()

    def _process_mlx(self, llm_sentence: str, language_code: Optional[str] = None) -> Iterator[np.ndarray]:
        """Process using MLX backend with Apple Silicon optimizations."""
        from scipy.signal import resample_poly

        gen = self.cancel_scope.generation if self.cancel_scope else None
        with MLXLockContext(handler_name="KokoroTTS", timeout=10.0):
            if language_code is not None:
                new_lang_code = WHISPER_LANGUAGE_TO_KOKORO_LANG.get(language_code, self.lang_code)
                if new_lang_code != self.lang_code:
                    new_voice = KOKORO_LANG_DEFAULT_VOICES.get(new_lang_code, self.voice)
                    logger.info(
                        f"Language change detected: {self.lang_code} -> {new_lang_code}, voice: {self.voice} -> {new_voice}"
                    )
                    try:
                        new_pipeline = self.model._get_pipeline(new_lang_code)
                        new_voice_tensor = new_pipeline.load_voice(new_voice)
                        self.lang_code = new_lang_code
                        self.voice = new_voice
                        self._pipeline = new_pipeline
                        self._voice_tensor = new_voice_tensor
                    except Exception as e:
                        logger.warning(
                            f"Failed to switch language/voice: {e}. Keeping current language: {self.lang_code}"
                        )

            console.print(f"[green]ASSISTANT: {llm_sentence}")

            # Generate audio using the preloaded pipeline directly
            # This avoids the voice reload that happens in model.generate()
            for result in self._pipeline(
                text=llm_sentence,
                voice=self.voice,
                speed=self.speed,
            ):
                if result.audio is None:
                    continue
                # result.audio is an mx.array with shape (1, samples), convert to numpy and squeeze
                audio = np.array(result.audio, dtype=np.float32).squeeze(0)

                # Trim silence from start and end of audio
                # Kokoro generates ~250ms of silence at the start and variable silence at the end
                threshold = 0.01
                abs_audio = np.abs(audio)
                # Find first and last samples above threshold
                above_threshold = abs_audio > threshold
                if np.any(above_threshold):
                    start_idx = int(np.argmax(above_threshold))
                    end_idx = len(audio) - int(np.argmax(above_threshold[::-1]))
                    # Add small padding (5ms = 120 samples at 24kHz) to avoid cutting speech
                    padding = 120
                    start_idx = max(0, start_idx - padding)
                    end_idx = min(len(audio), end_idx + padding)
                    audio = audio[start_idx:end_idx]

                # Kokoro outputs at 24kHz, resample to 16kHz for the pipeline
                # Using scipy's polyphase resampling (fast and high quality)
                # 16000/24000 = 2/3, so up=2, down=3
                audio = resample_poly(audio, up=2, down=3)

                # Convert to int16 format
                audio = (audio * 32768).astype(np.int16)

                # Yield audio in fixed-size chunks
                for i in range(0, len(audio), self.blocksize):
                    if gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen):
                        logger.info("TTS generation cancelled (interruption)")
                        return
                    chunk = audio[i : i + self.blocksize]
                    # Pad the last chunk if necessary
                    if len(chunk) < self.blocksize:
                        chunk = np.pad(chunk, (0, self.blocksize - len(chunk)))
                    logger.debug(f"TTS yielding audio chunk: {len(chunk)} samples")
                    yield chunk

    def _process_kokoro(self, llm_sentence: str, language_code: Optional[str] = None) -> Iterator[np.ndarray]:
        """Process using native kokoro library."""
        from scipy.signal import resample_poly

        gen = self.cancel_scope.generation if self.cancel_scope else None
        if language_code is not None:
            new_lang_code = WHISPER_LANGUAGE_TO_KOKORO_LANG.get(language_code, self.lang_code)
            if new_lang_code != self.lang_code:
                new_voice = KOKORO_LANG_DEFAULT_VOICES.get(new_lang_code, self.voice)
                logger.info(
                    f"Language change detected: {self.lang_code} -> {new_lang_code}, voice: {self.voice} -> {new_voice}"
                )
                self.lang_code = new_lang_code
                self.voice = new_voice
                from kokoro import KPipeline

                self.pipeline = KPipeline(lang_code=self.lang_code)

        console.print(f"[green]ASSISTANT: {llm_sentence}")

        # Generate audio using Kokoro
        # The pipeline yields tuples of (graphemes, phonemes, audio)
        for gs, ps, audio in self.pipeline(llm_sentence, voice=self.voice, speed=self.speed):
            if audio is None:
                continue

            # Ensure audio is numpy array
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio, dtype=np.float32)
            else:
                audio = audio.astype(np.float32)

            # Kokoro outputs at 24kHz, resample to 16kHz for the pipeline
            # Using scipy's polyphase resampling (fast and high quality)
            # 16000/24000 = 2/3, so up=2, down=3
            audio = resample_poly(audio, up=2, down=3)

            # Convert to int16 format
            audio = (audio * 32768).astype(np.int16)

            # Yield audio in fixed-size chunks
            for i in range(0, len(audio), self.blocksize):
                if gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen):
                    logger.info("TTS generation cancelled (interruption)")
                    return
                chunk = audio[i : i + self.blocksize]
                # Pad the last chunk if necessary
                if len(chunk) < self.blocksize:
                    chunk = np.pad(chunk, (0, self.blocksize - len(chunk)))
                yield chunk

    def on_session_end(self) -> None:
        self.voice = self._initial_voice
        self.lang_code = self._initial_lang_code
        if self.backend == "mlx":
            try:
                self._pipeline = self.model._get_pipeline(self.lang_code)
                self._voice_tensor = self._pipeline.load_voice(self.voice)
            except Exception as e:
                logger.warning(f"Failed to restore initial voice/language on session end: {e}")
        else:
            from kokoro import KPipeline

            self.pipeline = KPipeline(lang_code=self.lang_code)
        logger.debug("Kokoro TTS session state reset")
