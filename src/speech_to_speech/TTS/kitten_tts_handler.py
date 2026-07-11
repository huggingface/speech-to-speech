"""
KittenTTS Handler

Supports KittenML TTS models for high-quality speech synthesis.
"""

from __future__ import annotations

import logging
from threading import Event
from typing import Any, Iterator, Optional
import numpy as np

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, AudioOutput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)


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
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        **kwargs,
    ) -> None:
        self.should_listen = should_listen
        self.voice = voice
        self.device = device
        self.model_name = model_name
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns

        try:
            import sys
            import os
            # Auto-configure espeak-ng path for Windows if installed in default location
            if sys.platform == "win32" and "PHONEMIZER_ESPEAK_LIBRARY" not in os.environ:
                default_espeak_path = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
                if os.path.exists(default_espeak_path):
                    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = default_espeak_path
                    
            from kittentts import KittenTTS
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "KittenTTS is required. Install with: pip install kittentts huggingface_hub"
            ) from e

        logger.info(f"Loading KittenTTS model: {model_name} on {self.device}")
        
        # Download from HF if the model_name contains a slash (indicating a repo)
        if "/" in model_name:
            logger.info(f"Downloading KittenTTS weights from {model_name}...")
            # Automatically resolve paths to required files
            from huggingface_hub import list_repo_files
            repo_files = list_repo_files(model_name)
            onnx_files = [f for f in repo_files if f.endswith('.onnx')]
            if not onnx_files:
                raise ValueError(f"No .onnx file found in repo {model_name}")
            model_path = hf_hub_download(repo_id=model_name, filename=onnx_files[0])
            try:
                voices_path = hf_hub_download(repo_id=model_name, filename="voices.npz")
            except Exception:
                voices_path = None
        else:
            model_path = model_name
            voices_path = None

        # Load the model
        try:
            self.model = KittenTTS(model_path, voices_path=voices_path)
        except RuntimeError as e:
            if "espeak" in str(e).lower():
                raise RuntimeError(
                    "KittenTTS relies on `phonemizer`, which requires `espeak-ng` to be installed on your system.\n"
                    "Installation instructions:\n"
                    "  - Windows: Install the .msi from https://github.com/espeak-ng/espeak-ng/releases and restart your terminal.\n"
                    "  - macOS: brew install espeak-ng\n"
                    "  - Linux (Ubuntu/Debian): sudo apt-get install espeak-ng\n"
                ) from e
            raise e
        logger.info("KittenTTS model loaded")

        if self.voice not in self.model.available_voices:
            logger.warning(f"Voice '{self.voice}' not found. Falling back to default 'expr-voice-2-m'. Available voices: {self.model.available_voices}")
            self.voice = "expr-voice-2-m"

        # Monkey-patch the voices array shapes to fit ONNX inputs (KittenTTS mini-0.8 expects [1, 256] instead of [400, 256])
        if hasattr(self.model, "_voices"):
            self.model._voices = dict(self.model._voices)
            for k in self.model._voices:
                if len(self.model._voices[k].shape) == 2 and self.model._voices[k].shape[0] > 1:
                    self.model._voices[k] = self.model._voices[k][0:1, :]

        self.warmup()

    def warmup(self) -> None:
        """Warm up the TTS engine to ensure it's loaded into memory."""
        logger.info("Warming up KittenTTSHandler")
        # generate a short silent output
        try:
            _ = self.model.generate(text="Hello", voice=self.voice)
        except Exception as e:
            logger.error(f"Failed to warmup KittenTTS: {e}")
        logger.info("KittenTTSHandler warmed up")

    def process(self, input_chunk: TTSIn) -> Iterator[TTSOut]:
        """
        Synthesize speech from text chunks.
        """
        if isinstance(input_chunk, EndOfResponse):
            yield AUDIO_RESPONSE_DONE
            return

        text = input_chunk.text
        turn_id = input_chunk.turn_id
        revision_id = input_chunk.turn_revision if hasattr(input_chunk, 'turn_revision') else input_chunk.revision_id
        
        if not text or text.strip() == "":
            return

        from rich.console import Console
        console = Console()
        console.print(f"[green]ASSISTANT: {text}")

        logger.debug(f"KittenTTS synthesizing: {text}")

        # Generate full audio for the chunk
        try:
            audio_chunk = self.model.generate(text=text, voice=self.voice)
            if self.cancel_scope and self.cancel_scope.is_cancelled(turn_id, revision_id):
                logger.debug("TTS stream cancelled")
                return

            from scipy.signal import resample_poly
            
            # audio_chunk is a numpy array. Ensure it is a 1D float32 array
            audio = audio_chunk.squeeze().astype(np.float32)
            
            # KittenTTS outputs at 24kHz, resample to 16kHz for the pipeline
            # Using scipy's polyphase resampling (fast and high quality)
            # 16000/24000 = 2/3, so up=2, down=3
            audio = resample_poly(audio, up=2, down=3)

            # Convert to int16 format
            audio = (audio * 32767).astype(np.int16)
            
            # Yield audio in fixed-size chunks (default 512)
            blocksize = 512
            for i in range(0, len(audio), blocksize):
                if self.cancel_scope and self.cancel_scope.is_cancelled(turn_id, revision_id):
                    logger.debug("TTS generation cancelled (interruption)")
                    return
                chunk = audio[i : i + blocksize]
                # Pad the last chunk if necessary
                if len(chunk) < blocksize:
                    chunk = np.pad(chunk, (0, blocksize - len(chunk)))
                yield chunk

        except Exception as e:
            logger.error(f"Error in KittenTTS generation: {e}")
