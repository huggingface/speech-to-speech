"""Stream Big Bench Audio questions into a running Realtime server and collect answers.

One websocket connection per question. That is deliberate: the server builds a
fresh :class:`~speech_to_speech.LLM.chat.Chat` per connection, so reconnecting is
what keeps question *n* out of question *n+1*'s context.

The recordings run 13-31 seconds and the narration pauses mid-sentence, so the
default 64 ms VAD hangover would chop one question into several turns. The runner
raises ``turn_detection.silence_duration_ms`` for the session, and when a split
happens anyway it grades the last response of the item and flags it, rather than
silently scoring an answer to half a question.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

from speech_to_speech.evals.big_bench_audio.dataset import (
    SAMPLE_RATE_HZ,
    EvalItem,
    Subset,
    load_pcm16_mono,
    resolve_audio,
)
from speech_to_speech.evals.big_bench_audio.report import ItemResult
from speech_to_speech.evals.big_bench_audio.scoring import grade

logger = logging.getLogger(__name__)

BYTES_PER_SAMPLE = 2
# How often the consumer re-checks whether the question has finished streaming.
_PRE_SEND_POLL_S = 0.25

DEFAULT_INSTRUCTIONS = (
    "You are taking a spoken reasoning test. The user reads one self-contained question aloud. "
    "Work through it briefly out loud, then close with a sentence of exactly the form "
    "'Final answer: X'. Choose X like this: if the question asks whether an argument is "
    "deductively valid, X is the single word valid or invalid; if the question can be answered "
    "yes or no, X is the single word Yes or No; if the question asks how many, X is a single "
    "whole number. Never end without that final sentence. Keep the whole reply under 120 words."
)


@dataclass
class RunnerConfig:
    """Connection and turn-taking settings for one eval run."""

    url: str = "ws://127.0.0.1:8765/v1/realtime"
    api_key: Optional[str] = None
    instructions: str = DEFAULT_INSTRUCTIONS
    voice: Optional[str] = None
    # 1.0 streams at real time, as a microphone would. Higher is faster but leans
    # on the server's input queue rather than its VAD timing.
    speed: float = 1.0
    chunk_ms: int = 20
    trailing_silence_ms: int = 2000
    # Silero's 64 ms default splits these long, pause-heavy recordings.
    silence_duration_ms: int = 900
    vad_threshold: Optional[float] = None
    response_timeout_s: float = 180.0
    connect_timeout_s: float = 30.0
    # Quiet time on the socket after a response.done before the turn is settled.
    grace_s: float = 2.0
    settle_s: float = 0.5
    retries: int = 1

    def __post_init__(self) -> None:
        for name in ("speed", "chunk_ms", "response_timeout_s", "connect_timeout_s", "grace_s"):
            value = getattr(self, name)
            if not 0 < value < float("inf"):
                raise ValueError(f"{name} must be finite and positive")
        if self.retries < 0 or self.trailing_silence_ms < 0:
            raise ValueError("retries and trailing_silence_ms must be nonnegative")

    @property
    def chunk_bytes(self) -> int:
        return SAMPLE_RATE_HZ * BYTES_PER_SAMPLE * self.chunk_ms // 1000


@dataclass
class _ResponseCapture:
    """Transcript and timing for a single assistant response."""

    transcript: str = ""
    audio_bytes: int = 0
    first_audio_at: Optional[float] = None
    done_at: Optional[float] = None


@dataclass
class _TurnCapture:
    """Everything one question's connection produced."""

    input_transcript: str = ""
    responses: list[_ResponseCapture] = field(default_factory=list)
    audio_end_at: Optional[float] = None
    error: Optional[str] = None

    @property
    def last(self) -> Optional[_ResponseCapture]:
        return self.responses[-1] if self.responses else None


def build_session_update(config: RunnerConfig) -> dict[str, Any]:
    """The ``session.update`` that puts the server into eval mode."""
    turn_detection: dict[str, Any] = {
        "type": "server_vad",
        "interrupt_response": True,
        "silence_duration_ms": config.silence_duration_ms,
    }
    if config.vad_threshold is not None:
        turn_detection["threshold"] = config.vad_threshold

    session: dict[str, Any] = {
        "type": "realtime",
        "instructions": config.instructions,
        "audio": {"input": {"turn_detection": turn_detection}, "output": {}},
    }
    if config.voice:
        session["audio"]["output"]["voice"] = config.voice
    return {"type": "session.update", "session": session}


def iter_chunks(pcm: bytes, chunk_bytes: int) -> Iterator[bytes]:
    """Split *pcm* into fixed-size chunks, zero-padding the last one."""
    for offset in range(0, len(pcm), chunk_bytes):
        chunk = pcm[offset : offset + chunk_bytes]
        if len(chunk) < chunk_bytes:
            chunk = chunk + b"\x00" * (chunk_bytes - len(chunk))
        yield chunk


def _append_event(chunk: bytes) -> str:
    return json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode("ascii")})


async def _stream_audio(
    ws: Any,
    pcm: bytes,
    config: RunnerConfig,
    capture: _TurnCapture,
    audio_sent: asyncio.Event,
) -> None:
    """Send the question then trailing silence, pacing at ``speed`` x real time.

    ``capture.audio_end_at`` is stamped when the *question* ends, not when the
    trailing silence does, so latency is measured from the moment the user stopped
    speaking -- the VAD hangover a real caller waits through is part of it.
    ``audio_sent`` is set in a ``finally`` so a send that dies mid-question still
    starts the consumer's deadline instead of leaving it waiting forever.
    """
    pause = config.chunk_ms / 1000.0 / max(config.speed, 0.01)
    try:
        for chunk in iter_chunks(pcm, config.chunk_bytes):
            await ws.send(_append_event(chunk))
            await asyncio.sleep(pause)
        capture.audio_end_at = time.monotonic()

        silence = _append_event(b"\x00" * config.chunk_bytes)
        for _ in range(config.trailing_silence_ms // config.chunk_ms):
            await ws.send(silence)
            await asyncio.sleep(pause)
    finally:
        if capture.audio_end_at is None:
            capture.audio_end_at = time.monotonic()
        audio_sent.set()


async def _consume(ws: Any, config: RunnerConfig, capture: _TurnCapture, audio_sent: asyncio.Event) -> None:
    """Read server events into *capture* until the turn settles, times out, or errors."""
    current: Optional[_ResponseCapture] = None
    deadline: Optional[float] = None

    def ensure_current() -> _ResponseCapture:
        nonlocal current
        if current is None:
            current = _ResponseCapture()
            capture.responses.append(current)
        return current

    while True:
        if audio_sent.is_set() and deadline is None:
            deadline = time.monotonic() + config.response_timeout_s

        settled = bool(capture.responses) and capture.responses[-1].done_at is not None and audio_sent.is_set()
        if deadline is None:
            # Still streaming: poll so the response deadline starts promptly once
            # the last chunk goes out, instead of after a long blocking recv.
            timeout = _PRE_SEND_POLL_S
        elif settled:
            timeout = config.grace_s
        else:
            timeout = max(0.1, deadline - time.monotonic())

        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            if settled:
                return
            if deadline is not None and time.monotonic() >= deadline:
                capture.error = "response_timeout"
                return
            continue
        except Exception as exc:  # noqa: BLE001 — surfaced as a per-item error
            capture.error = f"{type(exc).__name__}: {exc}"
            return

        event = json.loads(raw)
        kind = event.get("type", "")

        if kind == "conversation.item.input_audio_transcription.completed":
            capture.input_transcript = " ".join(
                part for part in (capture.input_transcript, event.get("transcript", "")) if part
            ).strip()
        elif kind == "response.created":
            current = _ResponseCapture()
            capture.responses.append(current)
        elif kind in ("response.audio.delta", "response.output_audio.delta"):
            delta = event.get("delta", "")
            if delta:
                response = ensure_current()
                response.audio_bytes += len(base64.b64decode(delta))
                if response.first_audio_at is None:
                    response.first_audio_at = time.monotonic()
        elif kind in ("response.audio_transcript.delta", "response.output_audio_transcript.delta"):
            response = ensure_current()
            response.transcript += event.get("delta", "")
        elif kind in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
            transcript = event.get("transcript")
            if transcript:
                ensure_current().transcript = transcript
        elif kind in ("response.output_text.delta", "response.text.delta"):
            response = ensure_current()
            response.transcript += event.get("delta", "")
        elif kind == "response.done":
            ensure_current().done_at = time.monotonic()
            status = event.get("response", {}).get("status", "completed")
            if status != "completed":
                capture.error = f"response_{status}"
            current = None
        elif kind == "error":
            capture.error = str(event.get("error", {}).get("message") or "error event")
            return


@asynccontextmanager
async def open_session(config: RunnerConfig) -> Any:
    """Open one Realtime connection, wait for ``session.created``, and apply eval settings."""
    import websockets

    headers = [("Authorization", f"Bearer {config.api_key}")] if config.api_key else None
    async with websockets.connect(
        config.url,
        max_size=2**24,
        max_queue=2048,
        additional_headers=headers,
        open_timeout=config.connect_timeout_s,
    ) as ws:
        first = json.loads(await asyncio.wait_for(ws.recv(), timeout=config.connect_timeout_s))
        if first.get("type") == "error":
            raise RuntimeError(str(first.get("error", {}).get("message") or "connection rejected"))
        if first.get("type") != "session.created":
            raise RuntimeError(f"unexpected first event: {first.get('type')!r}")
        await ws.send(json.dumps(build_session_update(config)))
        updated = json.loads(await asyncio.wait_for(ws.recv(), timeout=config.connect_timeout_s))
        if updated.get("type") != "session.updated":
            raise RuntimeError(f"session configuration rejected: {updated.get('error') or updated.get('type')}")
        yield ws


async def run_item(config: RunnerConfig, item: EvalItem, pcm: bytes) -> ItemResult:
    """Ask one question over a fresh connection and grade the answer."""
    result = ItemResult(id=item.id, category=item.category, official_answer=item.official_answer)
    capture = _TurnCapture()
    try:
        async with open_session(config) as ws:
            audio_sent = asyncio.Event()
            sender = asyncio.create_task(_stream_audio(ws, pcm, config, capture, audio_sent))
            try:
                await _consume(ws, config, capture, audio_sent)
            finally:
                sender.cancel()
                outcomes = await asyncio.gather(sender, return_exceptions=True)
                if isinstance(outcomes[0], Exception):
                    capture.error = f"audio_send_failed: {outcomes[0]}"
    except Exception as exc:  # noqa: BLE001 — every failure is a comparable data point
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    result.input_transcript = capture.input_transcript
    last = capture.last
    if capture.error:
        result.error = capture.error
    elif last is None:
        result.error = "no_response"

    if last is not None:
        result.reply = last.transcript
        result.audio_out_bytes = last.audio_bytes
        if not last.audio_bytes:
            result.error = result.error or "no_audio_output"
        if capture.audio_end_at is not None:
            if last.first_audio_at is not None and last.first_audio_at >= capture.audio_end_at:
                result.ttfb_s = last.first_audio_at - capture.audio_end_at
            if last.done_at is not None and last.done_at >= capture.audio_end_at:
                result.turn_s = last.done_at - capture.audio_end_at
        if len(capture.responses) > 1:
            # The recording was chopped into several turns; the last one saw the
            # whole question, but its context is polluted by the earlier answers.
            result.error = result.error or f"split_turn ({len(capture.responses)} responses)"

    verdict = grade(item.category, item.official_answer, result.reply)
    result.extracted = verdict.extracted
    result.correct = verdict.correct
    result.method = verdict.method
    return result


ProgressCallback = Callable[[int, int, ItemResult], None]


async def run_subset(
    subset: Subset,
    config: RunnerConfig,
    *,
    progress: Optional[ProgressCallback] = None,
) -> list[ItemResult]:
    """Run every item in *subset* sequentially and return one result per item."""
    results: list[ItemResult] = []
    total = len(subset)
    for index, item in enumerate(subset.items, start=1):
        pcm = load_pcm16_mono(resolve_audio(item, subset))

        result = await run_item(config, item, pcm)
        for attempt in range(config.retries):
            if not result.error or result.error.startswith("split_turn"):
                break
            logger.warning("item %s failed (%s); retry %s", item.id, result.error, attempt + 1)
            await asyncio.sleep(config.settle_s)
            result = await run_item(config, item, pcm)

        results.append(result)
        if progress is not None:
            progress(index, total, result)
        await asyncio.sleep(config.settle_s)
    return results
