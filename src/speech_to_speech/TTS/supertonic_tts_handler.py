from __future__ import annotations

import logging
from math import gcd
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

PIPELINE_SAMPLE_RATE = 16000
SUPERTONIC_LANGUAGE_CODES = frozenset(
    {
        "ar",
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fi",
        "fr",
        "hi",
        "hr",
        "hu",
        "id",
        "it",
        "ja",
        "ko",
        "lt",
        "lv",
        "na",
        "nl",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "sv",
        "tr",
        "uk",
        "vi",
    }
)


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
        **_kwargs: object,
    ) -> None:
        if blocksize <= 0:
            raise ValueError(f"blocksize must be positive, got {blocksize}")

        normalized_lang = self._normalize_language_code(lang)
        if normalized_lang not in SUPERTONIC_LANGUAGE_CODES:
            raise ValueError(
                f"Unsupported Supertonic language code {lang!r}; "
                f"choose one of {', '.join(sorted(SUPERTONIC_LANGUAGE_CODES))}"
            )

        self.should_listen = should_listen
        self.voice = voice
        self.lang = normalized_lang
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
        logger.info("Loaded Supertonic TTS with voice %r", self.voice)
        self.warmup()

    def warmup(self) -> None:
        logger.info("Warming up Supertonic TTS...")
        _ = self.tts.synthesize(
            text="Warmup",
            lang=self.lang,
            voice_style=self.voice_style,
            speed=self.speed,
        )

    @staticmethod
    def _normalize_language_code(language_code: str) -> str:
        normalized = language_code.strip().lower()
        if normalized.endswith("-auto"):
            normalized = normalized.removesuffix("-auto")
        return normalized.split("-", 1)[0]

    def _resolve_language(self, language_code: str | None) -> str:
        if not language_code:
            return self.lang

        normalized = self._normalize_language_code(language_code)
        if normalized in SUPERTONIC_LANGUAGE_CODES:
            return normalized

        logger.warning(
            "Supertonic does not support language code %r; falling back to %r",
            language_code,
            self.lang,
        )
        return self.lang

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

        cancel_gen = self.cancel_scope.generation if self.cancel_scope else None

        text = tts_input.text
        if not text.strip():
            return

        lang = self._resolve_language(tts_input.language_code)

        console.print(f"[green]ASSISTANT: {text}")

        # Supertonic returns (1, num_samples) shaped array at 44.1kHz float32
        wav, _duration = self.tts.synthesize(
            text=text,
            lang=lang,
            voice_style=self.voice_style,
            speed=self.speed,
        )

        # If the user interrupted during the blocking synthesize(), drop the buffer.
        if self._is_cancelled(cancel_gen):
            logger.info("Supertonic TTS output cancelled (interruption)")
            return

        audio = np.asarray(wav, dtype=np.float32).squeeze()
        source_sample_rate = int(getattr(self.tts, "sample_rate", 44100))
        divisor = gcd(source_sample_rate, PIPELINE_SAMPLE_RATE)
        audio_16k_float = scipy.signal.resample_poly(
            audio,
            PIPELINE_SAMPLE_RATE // divisor,
            source_sample_rate // divisor,
        )

        # Convert to int16 format expected by the audio pipeline
        audio_int16 = np.clip(audio_16k_float * 32768, -32768, 32767).astype(np.int16)

        # Yield in block-aligned chunks so the streamer can handle it smoothly
        n = (len(audio_int16) // self.blocksize) * self.blocksize
        for i in range(0, n, self.blocksize):
            if self._is_cancelled(cancel_gen):
                logger.info("Supertonic TTS output cancelled (interruption)")
                return
            yield audio_int16[i : i + self.blocksize]

        # Pad the tail so the audio streamer's fixed-blocksize callback doesn't crash
        if n < len(audio_int16):
            if self._is_cancelled(cancel_gen):
                logger.info("Supertonic TTS output cancelled (interruption)")
                return
            tail = audio_int16[n:]
            yield np.pad(tail, (0, self.blocksize - len(tail)))
