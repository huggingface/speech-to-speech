from __future__ import annotations

import base64
import binascii
import logging
import os
from collections.abc import Iterator
from threading import Event
from time import perf_counter
from typing import Any

import httpx
import numpy as np
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from scipy.signal import resample_poly

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.arguments_classes.gemini_tts_arguments import DEFAULT_GEMINI_TTS_PROMPT
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import ResponseFailedEvent
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

genai: Any
try:
    from google import genai
except ImportError:
    genai = None

logger = logging.getLogger(__name__)

GEMINI_TTS_SAMPLE_RATE = 24000
PIPELINE_SAMPLE_RATE = 16000
_RESAMPLER_CONTEXT_SAMPLES = 96

GEMINI_TTS_VOICES = (
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
)


class _StreamingResampler:
    """Resample 24 kHz PCM to 16 kHz while retaining FIR context."""

    def __init__(self) -> None:
        self._buffer = np.zeros(_RESAMPLER_CONTEXT_SAMPLES, dtype=np.float64)
        self._pending_samples = 0

    def push(self, samples: np.ndarray, *, final: bool = False) -> np.ndarray:
        if samples.size:
            self._buffer = np.concatenate((self._buffer, samples.astype(np.float64, copy=False)))
            self._pending_samples += int(samples.size)

        if final:
            real_samples = self._pending_samples
            padded_samples = ((real_samples + 2) // 3) * 3
            self._buffer = np.pad(
                self._buffer,
                (0, padded_samples - real_samples + _RESAMPLER_CONTEXT_SAMPLES),
            )
            process_samples = padded_samples
            expected_output = (real_samples * 2 + 2) // 3
        else:
            process_samples = max(0, self._pending_samples - _RESAMPLER_CONTEXT_SAMPLES)
            process_samples -= process_samples % 3
            expected_output = process_samples * 2 // 3

        if process_samples == 0:
            return np.empty(0, dtype=np.int16)

        resampled = resample_poly(self._buffer, up=2, down=3)
        context_output = _RESAMPLER_CONTEXT_SAMPLES * 2 // 3
        output = resampled[context_output : context_output + expected_output]

        if final:
            self._buffer = np.zeros(_RESAMPLER_CONTEXT_SAMPLES, dtype=np.float64)
            self._pending_samples = 0
        else:
            self._buffer = self._buffer[process_samples:]
            self._pending_samples -= process_samples

        return np.clip(np.rint(output), -32768, 32767).astype(np.int16)


class GeminiTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """Stream Gemini TTS audio into the 16 kHz realtime pipeline."""

    def setup(
        self,
        should_listen: Event,
        model_name: str = "gemini-3.1-flash-tts-preview",
        voice: str = "Kore",
        api_key: str | None = None,
        prompt: str = DEFAULT_GEMINI_TTS_PROMPT,
        timeout_s: float = 20.0,
        blocksize: int = 512,
        gen_kwargs: dict[str, Any] | None = None,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        if genai is None:
            raise ImportError("google-genai is required for Gemini TTS")
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError("Gemini TTS requires --gemini_tts_api_key or GEMINI_API_KEY.")
        if timeout_s <= 0:
            raise ValueError("gemini_tts_timeout_s must be greater than zero.")
        if blocksize <= 0:
            raise ValueError("gemini_tts_blocksize must be greater than zero.")

        self.should_listen = should_listen
        self.model_name = model_name
        self.voice = self._canonical_voice(voice)
        self.prompt = prompt.strip()
        self.timeout_s = float(timeout_s)
        self.blocksize = int(blocksize)
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.gen_kwargs = gen_kwargs or {}
        self.client = genai.Client(
            api_key=resolved_key,
            http_options={"timeout": int(self.timeout_s * 1000)},
        )

    @staticmethod
    def _canonical_voice(voice: str) -> str:
        voices_by_lower = {candidate.lower(): candidate for candidate in GEMINI_TTS_VOICES}
        canonical = voices_by_lower.get(voice.strip().lower())
        if canonical is None:
            raise ValueError(
                f"Unsupported Gemini TTS voice {voice!r}. Supported voices: {', '.join(GEMINI_TTS_VOICES)}"
            )
        return canonical

    def _session_voice(
        self,
        runtime_config: RuntimeConfig | None,
        response: RealtimeResponseCreateParams | None,
    ) -> str:
        requested: str | None = None
        if response and response.audio and response.audio.output and response.audio.output.voice:
            requested = str(response.audio.output.voice)
        elif runtime_config is not None:
            audio = runtime_config.session.audio
            output = audio.output if audio is not None else None
            if output is not None and output.voice:
                requested = str(output.voice)
        if not requested:
            return self.voice
        try:
            return self._canonical_voice(requested)
        except ValueError:
            logger.error("Rejecting unsupported Gemini TTS session voice %r; using Kore", requested)
            return "Kore"

    def _input_prompt(self, text: str) -> str:
        if not self.prompt:
            return text
        return f"{self.prompt}\n\nTekst do przeczytania:\n{text}"

    def _create_stream(self, text: str, voice: str) -> Any:
        return self.client.interactions.create(
            model=self.model_name,
            input=self._input_prompt(text),
            response_format={"type": "audio"},
            generation_config={"speech_config": [{"voice": voice}]},
            stream=True,
            **self.gen_kwargs,
        )

    @staticmethod
    def _close_stream(stream: Any) -> None:
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception("Failed to close Gemini TTS stream")

    @staticmethod
    def _status_code(exc: BaseException) -> int | None:
        for candidate in (getattr(exc, "status_code", None), getattr(exc, "code", None)):
            if isinstance(candidate, int):
                return candidate
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        return status_code if isinstance(status_code, int) else None

    @classmethod
    def _is_retryable(cls, exc: BaseException) -> bool:
        if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
            return True
        status_code = cls._status_code(exc)
        return status_code == 429 or status_code is not None and 500 <= status_code < 600

    def _is_cancelled(self, generation: int | None) -> bool:
        return generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation)

    def _stream_audio(self, text: str, voice: str, generation: int | None) -> Iterator[np.ndarray]:
        stream: Any = None
        byte_remainder = b""
        block_remainder = np.empty(0, dtype=np.int16)
        resampler = _StreamingResampler()
        try:
            stream = self._create_stream(text, voice)
            for event in stream:
                if self._is_cancelled(generation):
                    logger.info("Gemini TTS generation cancelled (interruption)")
                    return
                if getattr(event, "event_type", None) != "step.delta":
                    continue
                delta = getattr(event, "delta", None)
                if getattr(delta, "type", None) != "audio":
                    continue
                encoded = getattr(delta, "data", None)
                if not encoded:
                    continue
                raw = byte_remainder + base64.b64decode(encoded, validate=True)
                even_length = len(raw) - len(raw) % 2
                byte_remainder = raw[even_length:]
                if even_length == 0:
                    continue
                pcm = np.frombuffer(raw[:even_length], dtype="<i2")
                converted = resampler.push(pcm)
                if converted.size == 0:
                    continue
                block_remainder = np.concatenate((block_remainder, converted))
                complete = len(block_remainder) // self.blocksize * self.blocksize
                for offset in range(0, complete, self.blocksize):
                    if self._is_cancelled(generation):
                        logger.info("Gemini TTS generation cancelled (interruption)")
                        return
                    yield block_remainder[offset : offset + self.blocksize]
                block_remainder = block_remainder[complete:]

            if byte_remainder:
                raise ValueError("Gemini TTS returned an incomplete PCM16 sample.")
            converted = resampler.push(np.empty(0, dtype=np.int16), final=True)
            if converted.size:
                block_remainder = np.concatenate((block_remainder, converted))
            if block_remainder.size:
                padding = (-len(block_remainder)) % self.blocksize
                if padding:
                    block_remainder = np.pad(block_remainder, (0, padding))
                for offset in range(0, len(block_remainder), self.blocksize):
                    if self._is_cancelled(generation):
                        logger.info("Gemini TTS generation cancelled (interruption)")
                        return
                    yield block_remainder[offset : offset + self.blocksize]
        except binascii.Error as exc:
            raise ValueError("Gemini TTS returned invalid base64 audio.") from exc
        finally:
            self._close_stream(stream)

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = self.speculative_turns
        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id,
                tts_input.turn_revision,
            ):
                if tts_input.response_key is None:
                    return
                tts_input.cleanup_only = True
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id,
            tts_input.turn_revision,
        ):
            logger.debug(
                "Dropping stale Gemini TTS input for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision
            )
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        text = tts_input.text.strip()
        if not text:
            return
        voice = self._session_voice(tts_input.runtime_config, tts_input.response)
        generation = tts_input.cancel_generation
        if generation is None and self.cancel_scope is not None:
            generation = self.cancel_scope.generation

        emitted_audio = False
        started_at = perf_counter()
        for attempt in range(2):
            try:
                for audio_chunk in self._stream_audio(text, voice, generation):
                    if not emitted_audio:
                        emitted_audio = True
                        if tts_input.speech_stopped_at_s is not None:
                            latency_s = perf_counter() - tts_input.speech_stopped_at_s
                            if latency_s >= 0:
                                logger.info(
                                    "Last speech detected to first speech out: %.3fs (turn=%s rev=%s)",
                                    latency_s,
                                    tts_input.turn_id,
                                    tts_input.turn_revision,
                                )
                        logger.info("Gemini TTS first audio in %.3fs", perf_counter() - started_at)
                    yield audio_chunk
                return
            except Exception as exc:
                if self._is_cancelled(generation):
                    return
                if attempt == 0 and not emitted_audio and self._is_retryable(exc):
                    logger.warning("Retrying Gemini TTS once after transient error: %s", exc)
                    continue
                logger.error("Gemini TTS generation failed: %s", exc, exc_info=True)
                yield ResponseFailedEvent(
                    message="Gemini TTS generation failed.",
                    turn_id=tts_input.turn_id,
                    turn_revision=tts_input.turn_revision,
                    cancel_generation=tts_input.cancel_generation,
                    response_key=tts_input.response_key,
                )
                return
