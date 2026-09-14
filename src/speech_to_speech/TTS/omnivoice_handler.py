from __future__ import annotations

import logging
from math import gcd
from threading import Event
from typing import Any, Iterator

import numpy as np
import scipy.signal
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)
console = Console()

PIPELINE_SAMPLE_RATE = 16000


class OmniVoiceTTSHandler(BaseHandler[TTSIn, TTSOut]):
    def setup(
        self,
        should_listen: Event,
        model_name: str = "k2-fsa/OmniVoice",
        device: str = "auto",
        dtype: str = "float16",
        ref_audio: str | None = None,
        ref_text: str | None = None,
        voice_clone_prompt: str | None = None,
        instruct: str | None = None,
        language: str | None = None,
        num_steps: int = 32,
        speed: float = 1.0,
        blocksize: int = 512,
        gen_kwargs: dict[str, Any] | None = None,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        if blocksize <= 0:
            raise ValueError(f"blocksize must be positive, got {blocksize}")
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if speed <= 0:
            raise ValueError(f"speed must be positive, got {speed}")
        if ref_audio is not None and not ref_text:
            raise ValueError("ref_text is required when ref_audio is configured")
        if ref_text is not None and ref_audio is None:
            raise ValueError("ref_audio is required when ref_text is configured")
        if voice_clone_prompt is not None and ref_audio is not None:
            raise ValueError("voice_clone_prompt and ref_audio are mutually exclusive")
        if instruct is not None and (voice_clone_prompt is not None or ref_audio is not None):
            raise ValueError("instruct cannot be combined with voice-cloning configuration")

        import torch
        from omnivoice import OmniVoice, VoiceClonePrompt

        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype)
        if torch_dtype is None:
            raise ValueError(f"Unsupported OmniVoice dtype {dtype!r}; choose float16, bfloat16, or float32")

        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.instruct = instruct
        self.language = language
        self.num_steps = num_steps
        self.speed = speed
        self.blocksize = blocksize

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                device = "xpu"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        logger.info("Loading OmniVoice model %r on %s", model_name, device)
        self.model = OmniVoice.from_pretrained(model_name, device_map=device, dtype=torch_dtype)

        self.voice_clone_prompt = None
        if voice_clone_prompt is not None:
            self.voice_clone_prompt = VoiceClonePrompt.load(voice_clone_prompt)
            logger.info("Loaded OmniVoice clone prompt from %r", voice_clone_prompt)
        elif ref_audio is not None:
            self.voice_clone_prompt = self.model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
            logger.info("Prepared OmniVoice clone prompt from %r", ref_audio)

    def _is_cancelled(self, generation: int | None) -> bool:
        return generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation)

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = getattr(self, "speculative_turns", None)
        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id, tts_input.turn_revision
            ):
                if tts_input.response_key is None:
                    return
                tts_input.cleanup_only = True
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id, tts_input.turn_revision
        ):
            logger.debug(
                "Dropping stale TTS input for turn=%s rev=%s",
                tts_input.turn_id,
                tts_input.turn_revision,
            )
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        text = tts_input.text
        if not text.strip():
            return

        cancel_generation = self.cancel_scope.generation if self.cancel_scope else None
        generation_kwargs: dict[str, Any] = {
            "text": text,
            "num_step": self.num_steps,
            "speed": self.speed,
        }
        language = self.language or tts_input.language_code
        if language:
            generation_kwargs["language"] = language
        if self.voice_clone_prompt is not None:
            generation_kwargs["voice_clone_prompt"] = self.voice_clone_prompt
        elif self.instruct is not None:
            generation_kwargs["instruct"] = self.instruct

        console.print(f"[green]ASSISTANT: {text}")
        audios = self.model.generate(**generation_kwargs)

        # OmniVoice generation is blocking. If the user interrupted while it ran,
        # discard the completed buffer before it reaches the output client.
        if self._is_cancelled(cancel_generation):
            logger.info("OmniVoice TTS output cancelled (interruption)")
            return
        if not audios:
            return

        audio = np.asarray(audios[0], dtype=np.float32).reshape(-1)
        source_sample_rate = int(getattr(self.model, "sampling_rate", 24000))
        if source_sample_rate != PIPELINE_SAMPLE_RATE:
            divisor = gcd(source_sample_rate, PIPELINE_SAMPLE_RATE)
            audio = scipy.signal.resample_poly(
                audio,
                PIPELINE_SAMPLE_RATE // divisor,
                source_sample_rate // divisor,
            )

        audio_int16 = np.clip(audio * 32768, -32768, 32767).astype(np.int16)
        full_block_samples = (len(audio_int16) // self.blocksize) * self.blocksize
        for offset in range(0, full_block_samples, self.blocksize):
            if self._is_cancelled(cancel_generation):
                logger.info("OmniVoice TTS output cancelled (interruption)")
                return
            yield audio_int16[offset : offset + self.blocksize]

        if full_block_samples < len(audio_int16):
            if self._is_cancelled(cancel_generation):
                logger.info("OmniVoice TTS output cancelled (interruption)")
                return
            tail = audio_int16[full_block_samples:]
            yield np.pad(tail, (0, self.blocksize - len(tail)))
