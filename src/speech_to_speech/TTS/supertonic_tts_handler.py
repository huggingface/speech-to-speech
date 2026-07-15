from __future__ import annotations

import logging
from threading import Event
from typing import Iterator

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


class SupertonicTTSHandler(BaseHandler[TTSIn, TTSOut]):
    def setup(
        self,
        should_listen: Event,
        voice: str = "M1",
        lang: str = "na",
        speed: float = 1.0,
        blocksize: int = 512,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        **kwargs,
    ) -> None:
        self.should_listen = should_listen
        self.voice = voice
        self.lang = lang
        self.speed = speed
        self.blocksize = blocksize
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns

        try:
            from supertonic import TTS
        except ImportError:
            logger.error(
                "Supertonic package is not installed. Please install it using "
                "`pip install supertonic` or `pip install speech-to-speech[supertonic]`"
            )
            raise

        self.tts = TTS(auto_download=True)
        self.voice_style = self.tts.get_voice_style(voice_name=self.voice)
        logger.info(f"Loaded Supertonic TTS with voice '{self.voice}'")
        self.warmup()

    def warmup(self) -> None:
        logger.info("Warming up Supertonic TTS...")
        _ = self.tts.synthesize(
            text="Warmup",
            lang=self.lang,
            voice_style=self.voice_style,
            speed=self.speed,
        )

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
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

        # Prefer per-utterance language from the pipeline; fall back to CLI arg.
        lang = tts_input.language_code if tts_input.language_code else self.lang

        console.print(f"[green]ASSISTANT: {text}")

        # Supertonic returns (1, num_samples) shaped array at 44.1kHz float32
        wav, _duration = self.tts.synthesize(
            text=text,
            lang=lang,
            voice_style=self.voice_style,
            speed=self.speed,
        )

        # If the user interrupted during the blocking synthesize(), drop the buffer.
        if cancel_gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(cancel_gen):
            logger.info("Supertonic TTS output cancelled (interruption)")
            return

        # Squeeze down to 1D
        audio_44k = wav.squeeze()

        # Resample from 44100 to 16000
        audio_16k_float = scipy.signal.resample_poly(audio_44k, 160, 441)

        # Convert to int16 format expected by the audio pipeline
        audio_int16 = np.clip(audio_16k_float * 32768, -32768, 32767).astype(np.int16)

        # Yield in block-aligned chunks so the streamer can handle it smoothly
        n = (len(audio_int16) // self.blocksize) * self.blocksize
        for i in range(0, n, self.blocksize):
            if cancel_gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(cancel_gen):
                logger.info("Supertonic TTS output cancelled (interruption)")
                return
            yield audio_int16[i : i + self.blocksize]

        # Pad the tail so the audio streamer's fixed-blocksize callback doesn't crash
        if n < len(audio_int16):
            tail = audio_int16[n:]
            yield np.pad(tail, (0, self.blocksize - len(tail)))
