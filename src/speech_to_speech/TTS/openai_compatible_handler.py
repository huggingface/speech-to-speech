from __future__ import annotations

import asyncio
import io
import logging
import os
from collections.abc import Callable
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Any, Iterator, cast

import httpx
import numpy as np

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import ResponseFailedEvent
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
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


_SPEECH_STREAM_DONE = object()
_SPEECH_STREAM_QUEUE_MAXSIZE = 2
_SPEECH_STREAM_POLL_INTERVAL_S = 0.025


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
        self._worker_loop: asyncio.AbstractEventLoop | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._deadline_exceeded = Event()

    def iter_bytes(self, cancel_check: Callable[[], bool]) -> Iterator[bytes]:
        deadline_at_s = perf_counter() + self.timeout_s
        self._raise_if_stopped(cancel_check)
        results: Queue[tuple[bool, object]] = Queue(maxsize=_SPEECH_STREAM_QUEUE_MAXSIZE)
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        worker = Thread(
            target=self._read_stream,
            args=(headers, results, cancel_check),
            name="tts-http-reader",
            daemon=True,
        )
        worker.start()

        completed = False
        try:
            while True:
                self._raise_if_stopped(cancel_check)
                remaining_s = deadline_at_s - perf_counter()
                if remaining_s <= 0:
                    self._deadline_exceeded.set()
                    self.cancel()
                    raise SpeechRequestError("speech request timed out")
                try:
                    succeeded, value = results.get(timeout=min(_SPEECH_STREAM_POLL_INTERVAL_S, remaining_s))
                except Empty:
                    continue
                self._raise_if_stopped(cancel_check)
                if not succeeded:
                    raise cast(BaseException, value)
                if value is _SPEECH_STREAM_DONE:
                    completed = True
                    return
                yield cast(bytes, value)
        finally:
            if not completed:
                self.cancel()
            worker.join()

    def _read_stream(
        self,
        headers: dict[str, str],
        results: Queue[tuple[bool, object]],
        cancel_check: Callable[[], bool],
    ) -> None:
        result: tuple[bool, object] = (True, _SPEECH_STREAM_DONE)
        loop = asyncio.new_event_loop()
        task = loop.create_task(self._read_stream_async(headers, results, cancel_check))
        with self._transport_lock:
            self._worker_loop = loop
            self._worker_task = task
            cancelled_before_start = self._cancelled.is_set()
        if cancelled_before_start:
            task.cancel()
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            result = (False, SpeechRequestCancelled())
        except Exception as exc:
            result = (False, self._normalize_error(exc))
        finally:
            with self._transport_lock:
                if self._worker_task is task:
                    self._worker_task = None
                if self._worker_loop is loop:
                    self._worker_loop = None
            loop.close()

        self._publish(results, result)

    async def _read_stream_async(
        self,
        headers: dict[str, str],
        results: Queue[tuple[bool, object]],
        cancel_check: Callable[[], bool],
    ) -> None:
        client = httpx.AsyncClient(timeout=self.timeout_s)
        try:
            if self._cancelled.is_set() or cancel_check():
                self._cancelled.set()
                raise SpeechRequestCancelled

            async with client.stream("POST", self.endpoint_url, headers=headers, json=self.payload) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise SpeechRequestError(f"speech server returned HTTP {exc.response.status_code}") from exc
                self._validate_content_type(response)
                async for chunk in response.aiter_bytes():
                    if self._cancelled.is_set():
                        return
                    if chunk and not self._publish(results, (True, chunk)):
                        return
        finally:
            close_task = asyncio.create_task(client.aclose())
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError:
                await close_task
                raise

    def _publish(self, results: Queue[tuple[bool, object]], result: tuple[bool, object]) -> bool:
        while not self._cancelled.is_set():
            try:
                results.put(result, timeout=_SPEECH_STREAM_POLL_INTERVAL_S)
            except Full:
                continue
            return True
        return False

    @staticmethod
    def _validate_content_type(response: httpx.Response) -> None:
        headers = getattr(response, "headers", None)
        if headers is None:
            return
        media_type = headers.get("content-type", "").partition(";")[0].strip().lower()
        if media_type.startswith("text/") or media_type == "application/json" or media_type.endswith("+json"):
            raise SpeechRequestError("speech endpoint returned a non-audio response")

    def _normalize_error(self, exc: Exception) -> BaseException:
        if isinstance(exc, (SpeechRequestCancelled, SpeechRequestError)):
            return exc
        if isinstance(exc, httpx.TimeoutException):
            return SpeechRequestError("speech request timed out")
        if isinstance(exc, httpx.HTTPError):
            if self._deadline_exceeded.is_set():
                return SpeechRequestError("speech request timed out")
            if self._cancelled.is_set():
                return SpeechRequestCancelled()
            return SpeechRequestError(f"speech transport failed: {type(exc).__name__}")
        if self._deadline_exceeded.is_set():
            return SpeechRequestError("speech request timed out")
        if self._cancelled.is_set():
            return SpeechRequestCancelled()
        return exc

    def cancel(self) -> None:
        with self._transport_lock:
            if self._cancelled.is_set():
                return
            self._cancelled.set()
            loop = self._worker_loop
            task = self._worker_task
        if loop is not None and task is not None:
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                # The worker may have completed and closed its loop between the
                # lock snapshot and this cancellation request.
                pass

    def _raise_if_stopped(self, cancel_check: Callable[[], bool]) -> None:
        if self._deadline_exceeded.is_set():
            raise SpeechRequestError("speech request timed out")
        if self._cancelled.is_set() or cancel_check():
            self.cancel()
            raise SpeechRequestCancelled


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
        self._failed_responses: set[tuple[int | None, str | None, str | None, int | None]] = set()

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        if isinstance(tts_input, EndOfResponse):
            if self.speculative_turns and not self.speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id,
                tts_input.turn_revision,
            ):
                if tts_input.response_key is None:
                    return
                tts_input.cleanup_only = True
            with self._operation_lock:
                self._failed_responses.discard(self._response_identity(tts_input))
            yield AUDIO_RESPONSE_DONE
            return

        if self.speculative_turns and not self.speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id,
            tts_input.turn_revision,
        ):
            logger.debug(
                "Dropping stale remote TTS input for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision
            )
            return

        cancel_generation = tts_input.cancel_generation
        if cancel_generation is None and self.cancel_scope is not None:
            cancel_generation = self.cancel_scope.generation
        response_identity = self._response_identity(tts_input, cancel_generation)
        with self._operation_lock:
            if response_identity in self._failed_responses:
                logger.debug(
                    "Dropping remote TTS input after response failure turn=%s rev=%s",
                    tts_input.turn_id,
                    tts_input.turn_revision,
                )
                return

        text = tts_input.text.strip()
        if not text:
            return
        voice = self._resolve_voice(tts_input.runtime_config, tts_input.response)
        language = self._resolve_language(tts_input.language_code)
        payload = self._request_payload(text=text, voice=voice, language=language)
        operation = HttpSpeechOperation(
            endpoint_url=self.endpoint_url,
            api_key=self.api_key,
            payload=payload,
            timeout_s=self.timeout,
        )
        with self._operation_lock:
            self._active_operation = operation

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
            if first_audio and not cancel_check():
                raise SpeechRequestError("speech endpoint returned no audio")
        except SpeechRequestCancelled:
            logger.info("OpenAI-compatible TTS request cancelled")
        except Exception as exc:
            message = str(exc) if isinstance(exc, SpeechRequestError) else "speech request failed"
            logger.error("OpenAI-compatible TTS failed: %s", message, exc_info=True)
            if not cancel_check():
                with self._operation_lock:
                    self._failed_responses.add(response_identity)
                self.queue_out.put(
                    cast(
                        TTSOut,
                        ResponseFailedEvent(
                            message=message,
                            turn_id=tts_input.turn_id,
                            turn_revision=tts_input.turn_revision,
                            cancel_generation=cancel_generation,
                            response_key=tts_input.response_key,
                        ),
                    )
                )
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

    @staticmethod
    def _response_identity(
        message: TTSInput | EndOfResponse,
        cancel_generation: int | None = None,
    ) -> tuple[int | None, str | None, str | None, int | None]:
        if cancel_generation is None:
            cancel_generation = message.cancel_generation
        return (cancel_generation, message.response_key, message.turn_id, message.turn_revision)

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
            # Preserve valid audio: a trailing byte cannot form a PCM16 sample, so
            # discard it with a warning instead of failing the whole response.
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
            self._failed_responses.clear()
        if operation is not None:
            operation.cancel()

    def cleanup(self) -> None:
        self.on_session_end()
