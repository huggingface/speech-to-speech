from __future__ import annotations

import io
import logging
import os
from collections.abc import Callable
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Any, Iterator

import httpx
import numpy as np

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.control import SESSION_END, is_control_message
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = 16000
LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}


class SpeechRequestCancelled(RuntimeError):
    pass


class SpeechRequestError(RuntimeError):
    """Sanitized HTTP/protocol failure safe to log or surface."""


class HttpSpeechOperation:
    """Exactly one speech request and its streaming transport lifecycle."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        api_key: str | None,
        payload: dict[str, Any],
        timeout_s: float,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.payload = payload
        self.timeout_s = timeout_s
        self._cancelled = Event()
        self._transport_lock = Lock()
        self._client: httpx.Client | None = None
        self._response: httpx.Response | None = None

    def iter_bytes(self, cancel_check: Callable[[], bool]) -> Iterator[bytes]:
        if cancel_check():
            self._cancelled.set()
            raise SpeechRequestCancelled
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        monitor_stop = Event()
        monitor = Thread(
            target=self._monitor_cancellation,
            args=(cancel_check, monitor_stop),
            name="tts-http-cancel",
            daemon=True,
        )
        monitor.start()
        client = httpx.Client(timeout=self.timeout_s)
        with self._transport_lock:
            self._client = client

        try:
            with client.stream("POST", self.endpoint_url, headers=headers, json=self.payload) as response:
                with self._transport_lock:
                    self._response = response
                if self._cancelled.is_set():
                    raise SpeechRequestCancelled
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise SpeechRequestError(f"speech server returned HTTP {exc.response.status_code}") from exc
                for chunk in response.iter_bytes():
                    if self._cancelled.is_set() or cancel_check():
                        self.cancel()
                        raise SpeechRequestCancelled
                    if chunk:
                        yield chunk
        except SpeechRequestCancelled:
            raise
        except SpeechRequestError:
            raise
        except httpx.TimeoutException as exc:
            raise SpeechRequestError("speech request timed out") from exc
        except httpx.HTTPError as exc:
            if self._cancelled.is_set():
                raise SpeechRequestCancelled from exc
            raise SpeechRequestError(f"speech transport failed: {type(exc).__name__}") from exc
        finally:
            monitor_stop.set()
            monitor.join(timeout=0.2)
            with self._transport_lock:
                self._response = None
                self._client = None
            client.close()

    def cancel(self) -> None:
        self._cancelled.set()
        with self._transport_lock:
            response = self._response
            client = self._client
        if response is not None:
            try:
                response.close()
            except Exception:
                logger.debug("Error closing speech response", exc_info=True)
        if client is not None:
            try:
                client.close()
            except Exception:
                logger.debug("Error closing speech client", exc_info=True)

    def _monitor_cancellation(self, cancel_check: Callable[[], bool], stop: Event) -> None:
        while not stop.wait(0.025):
            if cancel_check():
                self.cancel()
                return


class _StreamingLinearResampler:
    """Small stateful PCM resampler that preserves continuity across HTTP chunks."""

    def __init__(self, source_rate: int, target_rate: int) -> None:
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("sample rates must be positive")
        self.source_rate = source_rate
        self.target_rate = target_rate
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start = 0
        self._next_output_index = 0

    def push(self, samples: np.ndarray, *, final: bool = False) -> np.ndarray:
        incoming = np.asarray(samples, dtype=np.float32).reshape(-1)
        if incoming.size:
            self._buffer = np.concatenate((self._buffer, incoming))
        if self._buffer.size == 0:
            return np.empty(0, dtype=np.int16)
        if self.source_rate == self.target_rate:
            same_rate_output = self._buffer
            self._buffer = np.empty(0, dtype=np.float32)
            self._buffer_start = 0
            return np.clip(np.round(same_rate_output), -32768, 32767).astype(np.int16)

        last_source_index = self._buffer_start + self._buffer.size - 1
        resampled_output: list[float] = []
        while True:
            numerator = self._next_output_index * self.source_rate
            left = numerator // self.target_rate
            remainder = numerator % self.target_rate
            right = left + (1 if remainder else 0)
            if right > last_source_index:
                if not final or left > last_source_index:
                    break
                right = left
                remainder = 0
            left_offset = int(left - self._buffer_start)
            right_offset = int(right - self._buffer_start)
            fraction = remainder / self.target_rate
            value = self._buffer[left_offset] * (1.0 - fraction) + self._buffer[right_offset] * fraction
            resampled_output.append(float(value))
            self._next_output_index += 1

        next_source = (self._next_output_index * self.source_rate) // self.target_rate
        keep_from = max(self._buffer_start, int(next_source) - 1)
        drop = min(self._buffer.size, keep_from - self._buffer_start)
        if drop > 0:
            self._buffer = self._buffer[drop:]
            self._buffer_start += drop

        return np.clip(np.round(resampled_output), -32768, 32767).astype(np.int16)


class OpenAICompatibleTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """Client handler for POST /v1/audio/speech with cancellable PCM streaming."""

    def setup(
        self,
        should_listen: Event,
        base_url: str = "http://localhost:8091/v1",
        api_key: str | None = None,
        model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        voice: str = "aiden",
        language: str | None = None,
        task_type: str | None = None,
        instructions: str | None = None,
        response_format: str = "pcm",
        sample_rate: int = 24000,
        speed: float = 1.0,
        stream: bool = True,
        timeout: float = 300.0,
        blocksize: int = 512,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if response_format not in {"pcm", "wav"}:
            raise ValueError("OpenAI-compatible TTS currently supports response_format 'pcm' or 'wav'")
        if stream and response_format != "pcm":
            raise ValueError("Streaming OpenAI-compatible TTS requires response_format='pcm'")
        if timeout <= 0:
            raise ValueError("OpenAI-compatible TTS timeout must be > 0")
        if blocksize < 1:
            raise ValueError("OpenAI-compatible TTS blocksize must be >= 1")
        if stream and speed != 1.0:
            raise ValueError("Streaming OpenAI-compatible TTS requires speed=1.0")

        self.should_listen = should_listen
        self.base_url = base_url.rstrip("/")
        self.endpoint_url = f"{self.base_url}/audio/speech"
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.model = model
        self.voice = voice
        self.language = language
        self.task_type = task_type
        self.instructions = instructions
        self.response_format = response_format
        self.sample_rate = sample_rate
        self.speed = speed
        self.stream = stream
        self.timeout = timeout
        self.blocksize = blocksize
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.gen_kwargs = gen_kwargs or {}
        self._operation_lock = Lock()
        self._active_operation: HttpSpeechOperation | None = None

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        if isinstance(tts_input, EndOfResponse):
            if self.speculative_turns and not self.speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id,
                tts_input.turn_revision,
            ):
                return
            yield AUDIO_RESPONSE_DONE
            return

        if self.speculative_turns and not self.speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id,
            tts_input.turn_revision,
        ):
            logger.debug("Dropping stale remote TTS input for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision)
            return

        text, input_language = self._coalesce_pending_tts_input(tts_input)
        if not text:
            return
        voice = self._resolve_voice(tts_input.runtime_config, tts_input.response)
        language = self._resolve_language(input_language)
        payload = self._request_payload(text=text, voice=voice, language=language)
        operation = HttpSpeechOperation(
            endpoint_url=self.endpoint_url,
            api_key=self.api_key,
            payload=payload,
            timeout_s=self.timeout,
        )
        with self._operation_lock:
            self._active_operation = operation

        cancel_generation = tts_input.cancel_generation
        if cancel_generation is None and self.cancel_scope is not None:
            cancel_generation = self.cancel_scope.generation

        def cancel_check() -> bool:
            cancelled = self.stop_event.is_set() or (
                cancel_generation is not None
                and self.cancel_scope is not None
                and self.cancel_scope.is_stale(cancel_generation)
            )
            if cancelled:
                return True
            return self.speculative_turns is not None and not self.speculative_turns.is_latest(
                tts_input.turn_id,
                tts_input.turn_revision,
            )

        first_audio = True
        started_at_s = perf_counter()
        try:
            if self.response_format == "pcm":
                source_chunks = operation.iter_bytes(cancel_check)
                for chunk in self._decode_pcm_stream(source_chunks):
                    if cancel_check():
                        operation.cancel()
                        return
                    if first_audio:
                        if not self._commit_first_audio(tts_input):
                            operation.cancel()
                            return
                        self._log_first_audio_latency(tts_input, started_at_s)
                        first_audio = False
                    yield chunk
            else:
                encoded = b"".join(operation.iter_bytes(cancel_check))
                for chunk in self._decode_wav(encoded):
                    if cancel_check():
                        operation.cancel()
                        return
                    if first_audio:
                        if not self._commit_first_audio(tts_input):
                            operation.cancel()
                            return
                        self._log_first_audio_latency(tts_input, started_at_s)
                        first_audio = False
                    yield chunk
        except SpeechRequestCancelled:
            logger.info("OpenAI-compatible TTS request cancelled")
        except Exception as exc:
            message = str(exc) if isinstance(exc, SpeechRequestError) else "speech request failed"
            logger.error("OpenAI-compatible TTS failed: %s", message, exc_info=True)
        finally:
            with self._operation_lock:
                if self._active_operation is operation:
                    self._active_operation = None

    def _commit_first_audio(self, tts_input: TTSInput) -> bool:
        tracker = self.speculative_turns
        if tracker is None or tts_input.turn_id is None or tts_input.turn_revision is None:
            return True
        return tracker.commit_if_latest_after_reopen_grace(
            tts_input.turn_id,
            tts_input.turn_revision,
        )

    def _request_payload(self, *, text: str, voice: str, language: str | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": text,
            "voice": voice,
            "response_format": self.response_format,
            **self.gen_kwargs,
        }
        if self.stream:
            payload.update({"stream": True, "stream_format": "audio"})
        elif self.speed != 1.0:
            payload["speed"] = self.speed
        if language:
            payload["language"] = LANGUAGE_NAMES.get(language.lower(), language)
        if self.task_type:
            payload["task_type"] = self.task_type
        if self.instructions:
            payload["instructions"] = self.instructions
        return payload

    def _coalesce_pending_tts_input(self, current: TTSInput) -> tuple[str, str | None]:
        parts = [current.text.strip()] if current.text.strip() else []
        language = current.language_code
        if not hasattr(self.queue_in, "mutex") or not hasattr(self.queue_in, "queue"):
            return " ".join(parts), language

        with self.queue_in.mutex:
            while self.queue_in.queue:
                next_item = self.queue_in.queue[0]
                if (
                    is_control_message(next_item, SESSION_END.kind)
                    or (isinstance(next_item, bytes) and next_item == PIPELINE_END)
                    or isinstance(next_item, EndOfResponse)
                    or not isinstance(next_item, TTSInput)
                ):
                    break
                if current.turn_id != next_item.turn_id or current.turn_revision != next_item.turn_revision:
                    break
                if language and next_item.language_code and language != next_item.language_code:
                    break
                self.queue_in.queue.popleft()
                if next_item.text.strip():
                    parts.append(next_item.text.strip())
                language = language or next_item.language_code
        return " ".join(parts).strip(), language

    def _resolve_language(self, input_language: str | None) -> str | None:
        if self.language is None:
            return None
        if self.language.strip().lower() == "auto":
            return input_language or "Auto"
        return self.language

    def _resolve_voice(self, runtime_config: RuntimeConfig | None, response: Any) -> str:
        if response and response.audio and response.audio.output and response.audio.output.voice:
            return str(response.audio.output.voice)
        if runtime_config is not None:
            audio = runtime_config.session.audio
            output = audio.output if audio is not None else None
            if output is not None and output.voice:
                return str(output.voice)
        return self.voice

    def _decode_pcm_stream(self, encoded_chunks: Iterator[bytes]) -> Iterator[np.ndarray]:
        resampler = _StreamingLinearResampler(self.sample_rate, PIPELINE_SAMPLE_RATE)
        byte_remainder = b""
        sample_remainder = np.empty(0, dtype=np.int16)
        for encoded in encoded_chunks:
            encoded = byte_remainder + encoded
            usable = len(encoded) - (len(encoded) % 2)
            byte_remainder = encoded[usable:]
            if usable == 0:
                continue
            samples = np.frombuffer(encoded[:usable], dtype="<i2")
            converted = resampler.push(samples)
            sample_remainder = np.concatenate((sample_remainder, converted))
            while sample_remainder.size >= self.blocksize:
                yield sample_remainder[: self.blocksize].copy()
                sample_remainder = sample_remainder[self.blocksize :]

        converted = resampler.push(np.empty(0, dtype=np.int16), final=True)
        sample_remainder = np.concatenate((sample_remainder, converted))
        if byte_remainder:
            logger.warning("Speech endpoint returned an incomplete PCM16 sample")
        while sample_remainder.size >= self.blocksize:
            yield sample_remainder[: self.blocksize].copy()
            sample_remainder = sample_remainder[self.blocksize :]
        if sample_remainder.size:
            yield np.pad(sample_remainder, (0, self.blocksize - sample_remainder.size))

    def _decode_wav(self, encoded: bytes) -> Iterator[np.ndarray]:
        from scipy.io import wavfile
        from scipy.signal import resample_poly

        sample_rate, samples = wavfile.read(io.BytesIO(encoded))
        waveform = np.asarray(samples)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if np.issubdtype(waveform.dtype, np.floating):
            waveform = np.clip(waveform, -1.0, 1.0) * 32767.0
        waveform = waveform.astype(np.float32)
        if sample_rate != PIPELINE_SAMPLE_RATE:
            gcd = int(np.gcd(sample_rate, PIPELINE_SAMPLE_RATE))
            waveform = resample_poly(
                waveform,
                up=PIPELINE_SAMPLE_RATE // gcd,
                down=sample_rate // gcd,
            )
        pcm = np.clip(np.round(waveform), -32768, 32767).astype(np.int16)
        for offset in range(0, pcm.size, self.blocksize):
            chunk = pcm[offset : offset + self.blocksize]
            if chunk.size < self.blocksize:
                chunk = np.pad(chunk, (0, self.blocksize - chunk.size))
            yield chunk

    def _log_first_audio_latency(self, tts_input: TTSInput, request_started_at_s: float) -> None:
        logger.info("OpenAI-compatible TTS time to first audio: %.3fs", perf_counter() - request_started_at_s)
        if tts_input.speech_stopped_at_s is not None:
            logger.info(
                "Last speech detected to first speech out: %.3fs (turn=%s rev=%s)",
                max(0.0, perf_counter() - tts_input.speech_stopped_at_s),
                tts_input.turn_id,
                tts_input.turn_revision,
            )

    def on_session_end(self) -> None:
        with self._operation_lock:
            operation = self._active_operation
        if operation is not None:
            operation.cancel()

    def cleanup(self) -> None:
        self.on_session_end()
