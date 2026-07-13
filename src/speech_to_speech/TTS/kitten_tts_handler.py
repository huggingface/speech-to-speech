"""
KittenTTS Handler

Supports KittenML TTS models for high-quality speech synthesis.
"""

from __future__ import annotations

import logging
import os
import sys
from threading import Event
from typing import Iterator, Optional

import numpy as np
from rich.console import Console
from scipy.signal import resample_poly

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)
console = Console()


class KittenTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """
    Handles Text-to-Speech using KittenTTS model.
    """

    def setup(
        self,
        should_listen: Event,
        model_name: Optional[str] = "KittenML/kitten-tts-mini-0.8",
        device: str = "cpu",
        voice: str = "Bruno",
        blocksize: int = 512,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        **kwargs,
    ) -> None:
        self.should_listen = should_listen
        self.voice = voice
        self.device = device
        self.model_name = model_name
        self.blocksize = blocksize
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns

        try:
            # Auto-configure espeak-ng path for Windows if installed in default location
            if sys.platform == "win32" and "PHONEMIZER_ESPEAK_LIBRARY" not in os.environ:
                default_espeak_path = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
                if os.path.exists(default_espeak_path):
                    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = default_espeak_path

            from kittentts import KittenTTS
        except ImportError as e:
            raise ImportError(
                "KittenTTS is required. Install with: pip install kittentts"
            ) from e

        logger.info(f"Loading KittenTTS model: {model_name} on {self.device}")

        # Download model files from HuggingFace if a repo ID is given
        model_path = None
        voices_path = None
        if "/" in model_name:
            try:
                from huggingface_hub import hf_hub_download, list_repo_files

                logger.info(f"Downloading KittenTTS weights from {model_name}...")
                repo_files = list_repo_files(model_name)
                onnx_files = [f for f in repo_files if f.endswith(".onnx")]
                if not onnx_files:
                    raise ValueError(f"No .onnx file found in repo {model_name}")
                model_path = hf_hub_download(repo_id=model_name, filename=onnx_files[0])
                try:
                    voices_path = hf_hub_download(repo_id=model_name, filename="voices.npz")
                except Exception:
                    voices_path = None
            except ImportError:
                logger.warning(
                    "huggingface_hub not installed; passing model_name directly to KittenTTS"
                )

        # Load the model
        try:
            if model_path:
                self.model = KittenTTS(model_path=model_path, voices_path=voices_path)
            else:
                self.model = KittenTTS(model_path=model_name)
        except RuntimeError as e:
            if "espeak" in str(e).lower():
                raise RuntimeError(
                    "KittenTTS relies on `phonemizer`, which requires `espeak-ng` to be installed on your system.\n"
                    "Installation instructions:\n"
                    "  - Windows: Install the .msi from https://github.com/espeak-ng/espeak-ng/releases and restart your terminal.\n"
                    "  - macOS: brew install espeak-ng\n"
                    "  - Linux (Ubuntu/Debian): sudo apt-get install espeak-ng\n"
                ) from e
            raise
        logger.info("KittenTTS model loaded")

        # The ONNX model expects style input with shape [1, 256] but voices.npz
        # stores [400, 256]. Slice to the first row for each voice.
        if hasattr(self.model, "_voices") and hasattr(self.model._voices, "files"):
            voices_dict = {}
            for k in self.model._voices.files:
                v = self.model._voices[k]
                if v.ndim == 2 and v.shape[0] > 1:
                    voices_dict[k] = v[0:1, :]
                else:
                    voices_dict[k] = v
            self.model._voices = voices_dict

        # Map friendly voice names to internal names if needed
        self._voice_map = {
            "Bella": "expr-voice-2-f",
            "Jasper": "expr-voice-2-m",
            "Luna": "expr-voice-3-f",
            "Bruno": "expr-voice-3-m",
            "Rosie": "expr-voice-4-f",
            "Hugo": "expr-voice-4-m",
            "Kiki": "expr-voice-5-f",
            "Leo": "expr-voice-5-m",
        }
        internal_voice = self._voice_map.get(self.voice, self.voice)
        if internal_voice not in self.model.available_voices:
            logger.warning(
                f"Voice '{self.voice}' not found. Falling back to 'Bruno'. "
                f"Available: {self.model.available_voices}"
            )
            internal_voice = "expr-voice-3-m"
        self._internal_voice = internal_voice

        self.warmup()

    def warmup(self) -> None:
        """Warm up the TTS engine to ensure it's loaded into memory."""
        logger.info("Warming up KittenTTSHandler")
        try:
            _ = self.model.generate(text="Hello", voice=self._internal_voice)
        except Exception as e:
            logger.error(f"Failed to warmup KittenTTS: {e}")
        logger.info("KittenTTSHandler warmed up")

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        """Synthesize speech from text chunks."""
        speculative_turns = getattr(self, "speculative_turns", None)

        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id, tts_input.turn_revision
            ):
                return
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id, tts_input.turn_revision
        ):
            logger.debug(
                "Dropping stale TTS input for turn=%s rev=%s",
                tts_input.turn_id, tts_input.turn_revision,
            )
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        cancel_gen = self.cancel_scope.generation if self.cancel_scope else None

        text = tts_input.text
        if not text.strip():
            return

        console.print(f"[green]ASSISTANT: {text}")
        logger.debug(f"KittenTTS synthesizing: {text}")

        try:
            # Generate audio (blocking — KittenTTS PyPI package has no streaming API)
            audio_chunk = self.model.generate(text=text, voice=self._internal_voice)

            # Check for interruption after the blocking call
            if cancel_gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(cancel_gen):
                logger.info("KittenTTS output cancelled (interruption)")
                return

            # Ensure 1D float32
            audio = audio_chunk.squeeze().astype(np.float32)

            # Resample from 24kHz to 16kHz (24000 -> 16000 = 2/3)
            audio = resample_poly(audio, up=2, down=3)

            # Convert to int16 with clipping
            audio_int16 = np.clip(audio * 32768, -32768, 32767).astype(np.int16)

            # Yield in block-aligned chunks
            n = (len(audio_int16) // self.blocksize) * self.blocksize
            for i in range(0, n, self.blocksize):
                if cancel_gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(cancel_gen):
                    logger.info("KittenTTS output cancelled (interruption)")
                    return
                yield audio_int16[i : i + self.blocksize]

            # Pad the tail
            if n < len(audio_int16):
                tail = audio_int16[n:]
                yield np.pad(tail, (0, self.blocksize - len(tail)))

        except Exception as e:
            logger.error(f"Error in KittenTTS generation: {e}")
