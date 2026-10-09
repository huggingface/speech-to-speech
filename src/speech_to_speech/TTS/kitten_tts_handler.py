"""CPU synthesis with Kitten's English ONNX models."""

from __future__ import annotations

import logging
from threading import Event
from typing import Iterator

import numpy as np
from rich.console import Console
from scipy.signal import resample_poly

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.pipeline.transcript_logging import transcript_for_log

logger = logging.getLogger(__name__)
console = Console()


class KittenTTSHandler(BaseHandler[TTSIn, TTSOut]):
    def setup(
        self,
        should_listen: Event,
        model_name: str = "KittenML/kitten-tts-mini-0.8",
        device: str = "cpu",
        voice: str = "Bruno",
        blocksize: int = 512,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        **_kwargs: object,
    ) -> None:
        if device != "cpu":
            raise ValueError("KittenTTS supports only device='cpu'")
        if blocksize <= 0:
            raise ValueError(f"blocksize must be positive, got {blocksize}")

        self.should_listen = should_listen
        self.voice = voice
        self.blocksize = blocksize
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns

        try:
            from kittenml.kittentts_legacy import download_from_huggingface
        except ImportError as exc:
            raise ImportError("Install KittenTTS with: pip install 'speech-to-speech[kitten]'") from exc

        # The supported ONNX loader follows config.json, retains all voice rows,
        # and supplies checkpoint-compatible text chunking and style selection.
        logger.info("Loading KittenTTS model: %s on cpu", model_name)
        self.model = download_from_huggingface(model_name, backend="cpu")
        if not self._voice_available(voice):
            raise ValueError(f"Unsupported KittenTTS voice {voice!r}")
        self.warmup()

    def _voice_available(self, voice: str) -> bool:
        internal_voice = self.model.voice_aliases.get(voice, voice)
        return internal_voice in self.model.available_voices

    def _resolve_voice(self, tts_input: TTSInput) -> str:
        voice: str | None = None
        response = tts_input.response
        if response is not None and response.audio is not None and response.audio.output is not None:
            response_voice = response.audio.output.voice
            voice = str(response_voice) if response_voice else None
        if not voice and tts_input.runtime_config is not None:
            audio = tts_input.runtime_config.session.audio
            output = audio.output if audio is not None else None
            session_voice = output.voice if output is not None else None
            voice = str(session_voice) if session_voice else None
        if not voice:
            return self.voice
        if self._voice_available(voice):
            return voice
        logger.warning("Unsupported KittenTTS voice override: %s; using configured voice", transcript_for_log(voice))
        return self.voice

    def warmup(self) -> None:
        logger.info("Warming up KittenTTSHandler")
        self.model.generate(text="Hello", voice=self.voice)
        logger.info("KittenTTSHandler warmed up")

    def _is_cancelled(self, generation: int | None) -> bool:
        return generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation)

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = self.speculative_turns
        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.wait_for_gate(tts_input.turn_id, tts_input.turn_revision):
                if tts_input.response_key is None:
                    return
                tts_input.cleanup_only = True
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.wait_for_gate(
            tts_input.turn_id, tts_input.turn_revision, commit=True
        ):
            logger.debug("Dropping stale TTS input for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision)
            return

        text = tts_input.text
        if not text.strip():
            return
        cancel_gen = self.cancel_scope.generation if self.cancel_scope else None
        voice = self._resolve_voice(tts_input)
        console.print(f"[green]ASSISTANT: {text}")
        logger.debug("KittenTTS synthesizing: %s", transcript_for_log(text))

        # Synthesis completes before delivery; upstream handles bounded text chunks.
        # Let BaseHandler report failures through its privacy-safe exception logger.
        audio_chunk = self.model.generate(text=text, voice=voice)
        if self._is_cancelled(cancel_gen):
            return

        audio = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        audio = resample_poly(audio, up=2, down=3)
        audio_int16 = np.clip(audio * 32768, -32768, 32767).astype(np.int16)

        n = (len(audio_int16) // self.blocksize) * self.blocksize
        for i in range(0, n, self.blocksize):
            if self._is_cancelled(cancel_gen):
                return
            yield audio_int16[i : i + self.blocksize]
        if n < len(audio_int16):
            if self._is_cancelled(cancel_gen):
                return
            yield np.pad(audio_int16[n:], (0, self.blocksize - (len(audio_int16) - n)))
