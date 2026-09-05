from __future__ import annotations

import asyncio
import io
import logging
import os
import wave
from collections.abc import Callable
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Any, Iterator, cast

import httpx
import numpy as np
from scipy.signal import firwin, lfilter

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import ResponseFailedEvent
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = 16000


class SpeechRequestCancelled(RuntimeError):
    pass


class SpeechRequestError(RuntimeError):
    """Sanitized HTTP/protocol failure safe to log or surface."""


_SPEECH_STREAM_DONE = object()
_SPEECH_STREAM_QUEUE_MAXSIZE = 2
_SPEECH_STREAM_POLL_INTERVAL_S = 0.025
_AUDIO_MEDIA_TYPES_BY_FORMAT = {
    "pcm": frozenset({"audio/pcm", "audio/l16", "audio/x-pcm"}),
    "wav": frozenset({"application/x-wav", "audio/vnd.wave", "audio/wav", "audio/wave", "audio/x-wav"}),
    "mp3": frozenset({"audio/mp3", "audio/mpeg"}),
    "opus": frozenset({"audio/ogg", "audio/opus"}),
    "aac": frozenset({"audio/aac"}),
    "flac": frozenset({"audio/flac", "audio/x-flac"}),
}


class HttpSpeechOperation:
    """Exactly one speech request and its streaming transport lifecycle."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        api_key: str | None,
        payload: dict[str, Any],
        timeout_s: float,
        response_format: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.payload = payload
        self.extra_headers = dict(extra_headers or {})
        self.timeout_s = timeout_s
        self.response_format = response_format if response_format is not None else payload.get("response_format")
        self._cancelled = Event()
        self._transport_lock = Lock()
        self._worker_loop: asyncio.AbstractEventLoop | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._deadline_exceeded = Event()

    def iter_bytes(self, cancel_check: Callable[[], bool]) -> Iterator[bytes]:
        deadline_at_s = perf_counter() + self.timeout_s
        self._raise_if_stopped(cancel_check)
        results: Queue[tuple[bool, object]] = Queue(maxsize=_SPEECH_STREAM_QUEUE_MAXSIZE)
        headers = dict(self.extra_headers)
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

    def _validate_content_type(self, response: httpx.Response) -> None:
        headers = getattr(response, "headers", None)
        if headers is None:
            return
        media_type = headers.get("content-type", "").partition(";")[0].strip().lower()
        if media_type.startswith("text/") or media_type == "application/json" or media_type.endswith("+json"):
            raise SpeechRequestError("speech endpoint returned a non-audio response")
        response_format = self.response_format
        if response_format not in {"pcm", "wav"}:
            return
        for actual_format, media_types in _AUDIO_MEDIA_TYPES_BY_FORMAT.items():
            if media_type not in media_types:
                continue
            if actual_format != response_format:
                raise SpeechRequestError(
                    f"speech endpoint returned {actual_format} audio for requested {response_format} format"
                )
            return

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


class _StreamingFIRResampler:
    """Stateful anti-aliased resampler with stable output across input chunks."""

    def __init__(self, source_rate: int, target_rate: int) -> None:
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("sample rates must be positive")
        self.source_rate = source_rate
        self.target_rate = target_rate
        gcd = int(np.gcd(source_rate, target_rate))
        self._up = target_rate // gcd
        self._down = source_rate // gcd
        self._input_samples = 0
        self._upsampled_samples = 0
        self._output_samples = 0
        self._finalized = False

        if self._up == self._down:
            self._taps = np.ones(1, dtype=np.float64)
            self._filter_state = np.empty(0, dtype=np.float64)
            self._delay = 0
            return

        max_rate = max(self._up, self._down)
        half_length = 10 * max_rate
        self._taps = (
            firwin(
                2 * half_length + 1,
                cutoff=1.0 / max_rate,
                window=("kaiser", 5.0),
            )
            * self._up
        )
        self._filter_state = np.zeros(self._taps.size - 1, dtype=np.float64)
        self._delay = (self._taps.size - 1) // 2

    def push(self, samples: np.ndarray, *, final: bool = False) -> np.ndarray:
        if self._finalized:
            raise RuntimeError("resampler has already been finalized")

        incoming = np.asarray(samples, dtype=np.float64).reshape(-1)
        self._input_samples += incoming.size
        if self._up == self._down:
            if final:
                self._finalized = True
            return self._to_int16(incoming)

        output_parts: list[np.ndarray] = []
        if incoming.size:
            upsampled = np.zeros(incoming.size * self._up, dtype=np.float64)
            upsampled[:: self._up] = incoming
            output_parts.append(self._filter_and_decimate(upsampled))

        if final:
            self._finalized = True
            # Flush the FIR tail so the centered final samples are not truncated.
            output_parts.append(self._filter_and_decimate(np.zeros(self._taps.size - 1, dtype=np.float64)))

        output = np.concatenate(output_parts) if output_parts else np.empty(0, dtype=np.float64)
        target_total = (self._input_samples * self.target_rate + self.source_rate - 1) // self.source_rate
        remaining = max(0, target_total - self._output_samples)
        if output.size > remaining:
            output = output[:remaining]
        self._output_samples += output.size
        return self._to_int16(output)

    def _filter_and_decimate(self, upsampled: np.ndarray) -> np.ndarray:
        filtered, self._filter_state = lfilter(
            self._taps,
            np.ones(1, dtype=np.float64),
            upsampled,
            zi=self._filter_state,
        )
        global_start = self._upsampled_samples
        self._upsampled_samples += upsampled.size
        minimum_index = max(global_start, self._delay)
        phase = (minimum_index - self._delay) % self._down
        first_index = minimum_index if phase == 0 else minimum_index + self._down - phase
        local_start = first_index - global_start
        if local_start >= filtered.size:
            return np.empty(0, dtype=np.float64)
        return filtered[local_start :: self._down]

    @staticmethod
    def _to_int16(samples: np.ndarray) -> np.ndarray:
        return np.clip(np.round(samples), -32768, 32767).astype(np.int16)


class _StreamingByteReader(io.RawIOBase):
    """Expose an iterator of HTTP bytes as the file API used by ``wave``."""

    def __init__(self, chunks: Iterator[bytes]) -> None:
        super().__init__()
        self._chunks = iter(chunks)
        self._buffer = bytearray()
        self._eof = False

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            for chunk in self._chunks:
                self._buffer.extend(chunk)
            self._eof = True
            size = len(self._buffer)
        while len(self._buffer) < size and not self._eof:
            try:
                self._buffer.extend(next(self._chunks))
            except StopIteration:
                self._eof = True
        result = bytes(self._buffer[:size])
        del self._buffer[:size]
        return result

    def readable(self) -> bool:
        return True


class OpenAICompatibleTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """Client handler for POST /v1/audio/speech with cancellable audio streaming."""

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
        stream: bool = False,
        timeout: float = 300.0,
        blocksize: int = 512,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        gen_kwargs: dict[str, Any] | None = None,
        warmup_enabled: bool = True,
    ) -> None:
        if response_format not in {"pcm", "wav"}:
            raise ValueError("OpenAI-compatible TTS currently supports response_format 'pcm' or 'wav'")
        if stream and response_format != "pcm":
            raise ValueError("The vLLM streaming extension requires response_format='pcm'")
        if timeout <= 0:
            raise ValueError("OpenAI-compatible TTS timeout must be > 0")
        if blocksize < 1:
            raise ValueError("OpenAI-compatible TTS blocksize must be >= 1")
        if stream and speed != 1.0:
            raise ValueError("The vLLM streaming extension requires speed=1.0")

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
        if warmup_enabled:
            self.warmup()

    def warmup(self) -> None:
        """Validate the configured speech endpoint before accepting sessions."""
        logger.info("Warming up %s", self.__class__.__name__)
        started_at_s = perf_counter()
        operation = self._make_operation(
            text="Warmup",
            voice=self.voice,
        )

        source_chunks = operation.iter_bytes(self.stop_event.is_set)
        decoded_chunks = (
            self._decode_pcm_stream(source_chunks)
            if self.response_format == "pcm"
            else self._decode_wav_stream(source_chunks)
        )

        received_audio = False
        for chunk in decoded_chunks:
            received_audio = received_audio or chunk.size > 0
        if not received_audio:
            raise SpeechRequestError("speech endpoint returned no audio")

        logger.info(
            "%s warmed up in %.3fs",
            self.__class__.__name__,
            perf_counter() - started_at_s,
        )

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
        operation: HttpSpeechOperation | None = None
        try:
            voice = self._resolve_voice(tts_input.runtime_config, tts_input.response)
            operation = self._make_operation(text=text, voice=voice, runtime_config=tts_input.runtime_config)
            with self._operation_lock:
                self._active_operation = operation
            source_chunks = operation.iter_bytes(cancel_check)
            decoded_chunks = (
                self._decode_pcm_stream(source_chunks)
                if self.response_format == "pcm"
                else self._decode_wav_stream(source_chunks)
            )
            for chunk in decoded_chunks:
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
            if operation is not None:
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

    def _make_operation(
        self,
        *,
        text: str,
        voice: str | dict[str, str],
        runtime_config: RuntimeConfig | None = None,
    ) -> HttpSpeechOperation:
        routing = runtime_config.routing if runtime_config is not None else None
        payload = self._request_payload(text=text, voice=voice)
        if routing is not None:
            payload["model"] = routing.routes.tts.model
        return HttpSpeechOperation(
            endpoint_url=self.endpoint_url,
            api_key=self.api_key,
            payload=payload,
            timeout_s=self.timeout,
            response_format=self.response_format,
            extra_headers=routing.headers("tts") if routing else None,
        )

    def _request_payload(
        self,
        *,
        text: str,
        voice: str | dict[str, str],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": text,
            "voice": voice,
            "response_format": self.response_format,
            **self.gen_kwargs,
        }
        if self.response_format == "pcm":
            payload["stream_format"] = "audio"
        if self.stream:
            payload["stream"] = True
        elif self.speed != 1.0:
            payload["speed"] = self.speed
        if self.language:
            payload["language"] = self.language
        if self.task_type:
            payload["task_type"] = self.task_type
        if self.instructions:
            payload["instructions"] = self.instructions
        return payload

    def _resolve_voice(
        self,
        runtime_config: RuntimeConfig | None,
        response: Any,
    ) -> str | dict[str, str]:
        if response and response.audio and response.audio.output and response.audio.output.voice:
            return self._serialize_voice(response.audio.output.voice)
        if runtime_config is not None:
            audio = runtime_config.session.audio
            output = audio.output if audio is not None else None
            if output is not None and output.voice:
                return self._serialize_voice(output.voice)
        return self.voice

    @staticmethod
    def _serialize_voice(voice: Any) -> str | dict[str, str]:
        if isinstance(voice, str):
            return voice
        voice_id = voice.get("id") if isinstance(voice, dict) else getattr(voice, "id", None)
        if isinstance(voice_id, str) and voice_id:
            return {"id": voice_id}
        raise ValueError("Realtime voice overrides must be a voice name or custom voice ID")

    def _decode_pcm_stream(self, encoded_chunks: Iterator[bytes]) -> Iterator[np.ndarray]:
        byte_remainder = b""

        def sample_chunks() -> Iterator[np.ndarray]:
            nonlocal byte_remainder
            for encoded in encoded_chunks:
                encoded = byte_remainder + encoded
                usable = len(encoded) - (len(encoded) % 2)
                byte_remainder = encoded[usable:]
                if usable:
                    yield np.frombuffer(encoded[:usable], dtype="<i2")
            if byte_remainder:
                # Preserve valid audio: a trailing byte cannot form a PCM16 sample, so
                # discard it with a warning instead of failing the whole response.
                logger.warning("Speech endpoint returned an incomplete PCM16 sample")

        yield from self._resample_to_blocks(sample_chunks(), self.sample_rate)

    def _decode_wav_stream(self, encoded_chunks: Iterator[bytes]) -> Iterator[np.ndarray]:
        stream = _StreamingByteReader(encoded_chunks)
        try:
            wav_reader = wave.open(cast(Any, stream), "rb")
        except (EOFError, wave.Error) as exc:
            raise SpeechRequestError("speech endpoint returned an invalid WAV stream") from exc

        try:
            channels = wav_reader.getnchannels()
            sample_width = wav_reader.getsampwidth()
            sample_rate = wav_reader.getframerate()
            if wav_reader.getcomptype() != "NONE" or channels < 1 or sample_width not in {1, 2, 3, 4}:
                raise SpeechRequestError("speech endpoint returned an unsupported WAV format")
            source_frames_per_read = max(
                1024,
                (self.blocksize * sample_rate + PIPELINE_SAMPLE_RATE - 1) // PIPELINE_SAMPLE_RATE + 64,
            )

            def sample_chunks() -> Iterator[np.ndarray]:
                byte_remainder = b""
                frame_size = channels * sample_width
                while encoded := wav_reader.readframes(source_frames_per_read):
                    encoded = byte_remainder + encoded
                    usable = len(encoded) - (len(encoded) % frame_size)
                    byte_remainder = encoded[usable:]
                    if usable:
                        yield self._decode_wav_frames(encoded[:usable], channels, sample_width)
                if byte_remainder:
                    logger.warning("Speech endpoint returned an incomplete WAV audio frame")

            yield from self._resample_to_blocks(sample_chunks(), sample_rate)
        except wave.Error as exc:
            raise SpeechRequestError("speech endpoint returned an invalid WAV stream") from exc
        finally:
            wav_reader.close()

    @staticmethod
    def _decode_wav_frames(encoded: bytes, channels: int, sample_width: int) -> np.ndarray:
        if sample_width == 1:
            waveform = (np.frombuffer(encoded, dtype=np.uint8).astype(np.float64) - 128.0) * 256.0
        elif sample_width == 2:
            waveform = np.frombuffer(encoded, dtype="<i2").astype(np.float64)
        elif sample_width == 3:
            octets = np.frombuffer(encoded, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            values = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
            values = np.where(values & 0x800000, values - 0x1000000, values)
            waveform = values.astype(np.float64) / 256.0
        else:
            waveform = np.frombuffer(encoded, dtype="<i4").astype(np.float64) / 65536.0
        frames = waveform.reshape(-1, channels)
        return frames.mean(axis=1) if channels > 1 else frames[:, 0]

    def _resample_to_blocks(
        self,
        sample_chunks: Iterator[np.ndarray],
        source_rate: int,
    ) -> Iterator[np.ndarray]:
        resampler = _StreamingFIRResampler(source_rate, PIPELINE_SAMPLE_RATE)
        sample_remainder = np.empty(0, dtype=np.int16)
        for samples in sample_chunks:
            converted = resampler.push(samples)
            sample_remainder = np.concatenate((sample_remainder, converted))
            while sample_remainder.size >= self.blocksize:
                yield sample_remainder[: self.blocksize].copy()
                sample_remainder = sample_remainder[self.blocksize :]

        converted = resampler.push(np.empty(0, dtype=np.float64), final=True)
        sample_remainder = np.concatenate((sample_remainder, converted))
        while sample_remainder.size >= self.blocksize:
            yield sample_remainder[: self.blocksize].copy()
            sample_remainder = sample_remainder[self.blocksize :]
        if sample_remainder.size:
            yield np.pad(sample_remainder, (0, self.blocksize - sample_remainder.size))

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
