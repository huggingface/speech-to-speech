"""Headless, stateful Qwen3-ASR inference through the qwen-asr vLLM backend.

This module owns one shared model and its independently mutable per-stream
states.  It does not own a microphone, VAD, conversation, or network protocol.
Applications decide which audio to admit and when to commit a stream.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


class QwenVllmUnhealthyError(RuntimeError):
    """The shared model may still be executing an abandoned inference call."""


class QwenVllmAdmissionError(RuntimeError):
    """Stream capacity is full, or a call's deadline expires before inference."""


@dataclass(frozen=True)
class QwenVllmSettings:
    """Backend settings and explicit limits for one shared engine."""

    model: str
    language: str | None
    gpu_memory_utilization: float
    max_new_tokens: int
    unfixed_chunk_num: int
    unfixed_token_num: int
    chunk_size_sec: float
    stream_step_ms: int
    max_segment_sec: float
    inference_timeout_s: float
    inference_rt_margin: float
    max_concurrency: int
    max_streams: int
    quantization: str | None = None
    kv_cache_dtype: str | None = None
    mm_processor_cache_gb: float = 0.0
    enable_prefix_caching: bool = False
    sample_rate: int = 16000

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.sample_rate != 16000:
            raise ValueError("Qwen3-ASR streaming requires 16000 Hz audio")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        positive_counts = (self.max_new_tokens, self.stream_step_ms, self.max_concurrency, self.max_streams)
        if any(type(value) is not int or value <= 0 for value in positive_counts):
            raise ValueError("token, step, concurrency, and stream limits must be positive integers")
        counts = (self.unfixed_chunk_num, self.unfixed_token_num)
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("unfixed chunk and token settings must be nonnegative integers")
        durations = (
            self.chunk_size_sec,
            self.max_segment_sec,
            self.inference_timeout_s,
            self.inference_rt_margin,
            self.mm_processor_cache_gb,
        )
        if not all(math.isfinite(value) for value in durations):
            raise ValueError("duration, timeout, and cache settings must be finite")
        if self.chunk_size_sec <= 0 or self.max_segment_sec < 0:
            raise ValueError("chunk_size_sec must be positive and max_segment_sec nonnegative")
        if self.inference_timeout_s < 0 or self.inference_rt_margin < 0 or self.mm_processor_cache_gb < 0:
            raise ValueError("timeout, margin, and cache settings must be nonnegative")
        if self.max_segment_sec and round(self.max_segment_sec * self.sample_rate) < 1:
            raise ValueError("max_segment_sec must allow at least one sample")


@dataclass(frozen=True)
class TranscriptUpdate:
    text: str
    language: str | None
    final: bool
    forced: bool = False


class QwenVllmEngine:
    """One shared model with bounded inference and an explicit health signal.

    A blocking qwen-asr call can outlive its awaiter. Cancelled work retains
    its admission slot until it drains; timed-out work poisons the engine so
    the hosting service can restart rather than admit work into uncertain state.
    """

    def __init__(
        self,
        settings: QwenVllmSettings,
        *,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings
        self._model_factory = model_factory
        self._model: Any = None
        self._startup: asyncio.Task[None] | None = None
        self._admission = asyncio.Semaphore(settings.max_concurrency)
        # Reserve capacity for the stream's lifetime, including queued inference.
        self._active_streams = 0
        self._poison_reason: str | None = None
        self._drainers: set[asyncio.Task[None]] = set()

    def health(self) -> tuple[bool, str]:
        """Report whether the engine can safely accept work; start() establishes readiness."""
        return (False, self._poison_reason) if self._poison_reason else (True, "ok")

    async def start(self) -> None:
        """Load and warm one model, even if a waiting caller is cancelled."""
        self._check_health()
        if self._startup is None:
            self._startup = asyncio.create_task(self._initialize())
            self._startup.add_done_callback(self._startup_finished)
        await asyncio.shield(self._startup)

    async def _initialize(self) -> None:
        factory = self._model_factory
        if factory is None:
            try:
                from qwen_asr import Qwen3ASRModel
            except ImportError as exc:
                raise RuntimeError("Install qwen-asr[vllm] to use stateful Qwen3-ASR") from exc
            factory = Qwen3ASRModel.LLM
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "gpu_memory_utilization": self.settings.gpu_memory_utilization,
            "max_new_tokens": self.settings.max_new_tokens,
            "mm_processor_cache_gb": self.settings.mm_processor_cache_gb,
            "enable_prefix_caching": self.settings.enable_prefix_caching,
        }
        if self.settings.quantization:
            kwargs["quantization"] = self.settings.quantization
        if self.settings.kv_cache_dtype:
            kwargs["kv_cache_dtype"] = self.settings.kv_cache_dtype
        self._model = await asyncio.to_thread(factory, **kwargs)
        state = await self._new_state(timeout=0)
        silence = np.zeros(round(self.settings.sample_rate * self.settings.chunk_size_sec), dtype=np.float32)
        await self._call("warmup_step", self._model.streaming_transcribe, silence, state, timeout=0)
        await self._call("warmup_finish", self._model.finish_streaming_transcribe, state, timeout=0)

    def _startup_finished(self, task: asyncio.Task[None]) -> None:
        if task.cancelled() or task.exception() is not None:
            self._poison_reason = self._poison_reason or "startup_failed"

    def _check_health(self) -> None:
        if self._poison_reason:
            raise QwenVllmUnhealthyError(self._poison_reason)

    async def new_stream(self) -> QwenVllmStream:
        """Reserve a stream slot and return independent, warmed inference state."""
        if self._active_streams >= self.settings.max_streams:
            raise QwenVllmAdmissionError("Qwen stream capacity is full")
        self._active_streams += 1
        try:
            await self.start()
            return QwenVllmStream(self, await self._new_state())
        except BaseException:
            self._release_stream()
            raise

    def _release_stream(self) -> None:
        if self._active_streams <= 0:
            raise RuntimeError("Qwen stream admission released more than once")
        self._active_streams -= 1

    async def _new_state(self, *, timeout: float | None = None) -> Any:
        return await self._call(
            "init_state",
            self._model.init_streaming_state,
            timeout=timeout,
            language=self.settings.language,
            unfixed_chunk_num=self.settings.unfixed_chunk_num,
            unfixed_token_num=self.settings.unfixed_token_num,
            chunk_size_sec=self.settings.chunk_size_sec,
        )

    async def _advance(self, state: Any, pcm: bytes, segment_bytes: int) -> None:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        await self._call(
            "streaming_transcribe",
            self._model.streaming_transcribe,
            samples,
            state,
            timeout=self._timeout(segment_bytes),
        )

    async def _finish(self, state: Any, segment_bytes: int) -> None:
        await self._call(
            "finish_streaming_transcribe",
            self._model.finish_streaming_transcribe,
            state,
            timeout=self._timeout(segment_bytes),
        )

    def _timeout(self, segment_bytes: int) -> float:
        base = self.settings.inference_timeout_s
        return base + self.settings.inference_rt_margin * segment_bytes / (self.settings.sample_rate * 2) if base else 0

    async def _call(
        self, label: str, fn: Callable[..., Any], *args: Any, timeout: float | None = None, **kwargs: Any
    ) -> Any:
        self._check_health()
        budget = self.settings.inference_timeout_s if timeout is None else timeout
        deadline = asyncio.get_running_loop().time() + budget if budget else None
        acquired = False
        work: asyncio.Task[Any] | None = None
        try:
            async with asyncio.timeout_at(deadline):
                await self._admission.acquire()
                acquired = True
                self._check_health()
                work = asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
                return await asyncio.shield(work)
        except TimeoutError as exc:
            if work is None:
                raise QwenVllmAdmissionError(f"{label} deadline expired while waiting for inference") from exc
            self._poison_reason = f"{label}_timeout"
            raise QwenVllmUnhealthyError(self._poison_reason) from exc
        finally:
            if acquired:
                if work is not None and not work.done():
                    self._drain(work, label, deadline=deadline)
                else:
                    self._admission.release()

    def _drain(self, work: asyncio.Task[Any], label: str, *, deadline: float | None) -> None:
        async def finish() -> None:
            try:
                if deadline is None:
                    await work
                else:
                    remaining = max(0, deadline - asyncio.get_running_loop().time())
                    await asyncio.wait_for(asyncio.shield(work), remaining)
            except TimeoutError:
                self._poison_reason = f"{label}_timeout"
                # Poisoning rejects new work, but the worker still owns its slot.
                try:
                    await work
                except Exception:
                    pass
            except Exception:
                # The abandoned call's result cannot be delivered to a closed stream.
                pass
            finally:
                self._admission.release()

        task = asyncio.create_task(finish())
        self._drainers.add(task)
        task.add_done_callback(self._drainers.discard)


class QwenVllmStream:
    """Per-session PCM16 stream; ``commit`` flushes the current state exactly once."""

    def __init__(self, engine: QwenVllmEngine, state: Any) -> None:
        self._engine = engine
        self._state = state
        self._pending = bytearray()
        self._segment_bytes = 0
        self._last_partial = ""
        self._closed = False
        self._released = False
        self._needs_reset = False
        self._lock = asyncio.Lock()

    async def push(self, pcm16: bytes) -> list[TranscriptUpdate]:
        """Accept ordered PCM16 samples and return changed partials or forced finals."""
        async with self._lock:
            self._check_open()
            if len(pcm16) % 2:
                raise ValueError("PCM16 input must contain complete samples")
            cap = round(self._engine.settings.max_segment_sec * self._engine.settings.sample_rate) * 2
            if cap and len(pcm16) > cap:
                raise ValueError("PCM16 input exceeds the configured segment cap")
            try:
                updates: list[TranscriptUpdate] = []
                step_bytes = round(self._engine.settings.sample_rate * self._engine.settings.stream_step_ms / 1000) * 2
                while pcm16:
                    await self._reset_after_final()
                    remaining = cap - self._segment_bytes - len(self._pending) if cap else len(pcm16)
                    count = min(len(pcm16), remaining)
                    self._pending.extend(pcm16[:count])
                    pcm16 = pcm16[count:]
                    at_cap = bool(cap and self._segment_bytes + len(self._pending) == cap)
                    if len(self._pending) >= step_bytes or at_cap:
                        updates.extend(await self._advance_pending())
                    if at_cap:
                        updates.append(await self._finalize(forced=True))
                return updates
            except BaseException:
                self._discard()
                raise

    async def commit(self) -> list[TranscriptUpdate]:
        """Flush this utterance's pending samples and return one explicit final."""
        async with self._lock:
            self._check_open()
            try:
                await self._reset_after_final()
                updates = await self._advance_pending() if self._pending else []
                updates.append(await self._finalize(forced=False))
                return updates
            except BaseException:
                self._discard()
                raise

    async def close(self) -> None:
        """Reject late output and release capacity after the current operation finishes."""
        self._closed = True
        async with self._lock:
            self._discard()

    def _discard(self) -> None:
        if self._released:
            return
        self._closed = True
        self._pending.clear()
        self._engine._release_stream()
        self._released = True

    async def _advance_pending(self) -> list[TranscriptUpdate]:
        pcm = bytes(self._pending)
        self._pending.clear()
        next_size = self._segment_bytes + len(pcm)
        await self._engine._advance(self._state, pcm, next_size)
        self._check_open()
        self._segment_bytes = next_size
        text = self._state.text
        if text and text != self._last_partial:
            self._last_partial = text
            return [TranscriptUpdate(text, self._language(), final=False)]
        return []

    async def _finalize(self, *, forced: bool) -> TranscriptUpdate:
        await self._engine._finish(self._state, self._segment_bytes)
        self._check_open()
        update = TranscriptUpdate(self._state.text, self._language(), final=True, forced=forced)
        self._needs_reset = True
        self._segment_bytes = 0
        self._last_partial = ""
        return update

    async def _reset_after_final(self) -> None:
        if self._needs_reset:
            self._state = await self._engine._new_state()
            self._check_open()
            self._needs_reset = False

    def _language(self) -> str | None:
        return self._state.language or self._engine.settings.language

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Qwen stream is closed")
