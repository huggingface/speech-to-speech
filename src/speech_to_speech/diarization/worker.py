from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from threading import Event
from time import perf_counter

import numpy as np

from speech_to_speech.pipeline.speaker_metadata import (
    PendingSpeakerAttribution,
    SpeakerAttribution,
    SpeakerAttributionFuture,
    SpeakerInterval,
    SpeakerSession,
)
from speech_to_speech.pipeline.transcript_logging import log_exception

from .streaming import SpeakerSegment, StreamingDiarizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Work:
    session: SpeakerSession
    audio: np.ndarray | None = None
    start: int = 0
    end: int = 0
    result: SpeakerAttributionFuture | None = None
    enqueued_at: float = field(default_factory=lambda: perf_counter())


class DiarizationWorker:
    """Ordered speech-only model worker. Producers never perform inference or wait.

    Absolute sample offsets preserve skipped silence and let duplicate pre-roll
    be trimmed. A session whose bounded queue overflows is invalidated rather
    than silently dropping audio and corrupting speaker identity.
    """

    def __init__(self, diarizer: StreamingDiarizer, stop_event: Event, *, max_queue_items: int = 128):
        if max_queue_items < 1:
            raise ValueError("The diarization queue must have a positive bound")
        self.diarizer = diarizer
        self.stop_event = stop_event
        self.sample_rate = diarizer.sample_rate
        self.queue: Queue[_Work] = Queue(maxsize=max_queue_items)
        self._session = SpeakerSession()
        self._active_session: SpeakerSession | None = None
        self._clear_timeline()

    def _clear_timeline(self) -> None:
        self._origin: int | None = None
        self._last_end = 0
        self._segments: deque[SpeakerSegment] = deque()
        self._coverage: deque[tuple[int, int]] = deque()
        self._history_start = 0
        self._processing_seconds = 0.0

    def reset_session(self) -> None:
        # The VAD thread invalidates old promises immediately. Only run() ever
        # touches the model, including a reset after an old forward finishes.
        self._session.invalid.set()
        self._session = SpeakerSession()
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break

    def _enqueue(self, work: _Work) -> None:
        if self.stop_event.is_set() or work.session.invalid.is_set():
            work.session.invalid.set()
            return
        try:
            self.queue.put_nowait(work)
        except Full:
            work.session.invalid.set()
            logger.warning("Diarization fell behind; speaker labels unavailable until the next session")

    def append_audio(self, audio: np.ndarray, start_sample: int) -> None:
        # Each work item is bounded even if a caller supplies an unusually large
        # onset buffer. With the normal VAD, blocks are just 32 ms.
        for offset in range(0, len(audio), self.sample_rate // 10):
            if self._session.invalid.is_set() or self.stop_event.is_set():
                return
            block = audio[offset : offset + self.sample_rate // 10].astype(np.float32, copy=True)
            self._enqueue(_Work(self._session, audio=block, start=start_sample + offset))

    def finalize(self, start_sample: int, end_sample: int) -> PendingSpeakerAttribution:
        result = SpeakerAttributionFuture()
        session = self._session
        self._enqueue(_Work(session, start=start_sample, end=end_sample, result=result))
        return PendingSpeakerAttribution(((result, session, 0.0),))

    def discard_utterance(self) -> None:
        self._enqueue(_Work(self._session))

    def _store(self, segments: list[SpeakerSegment]) -> None:
        assert self._origin is not None
        offset = self._origin / self.sample_rate
        self._segments.extend(SpeakerSegment(s.speaker, s.start + offset, s.end + offset) for s in segments)

    def _flush(self) -> None:
        if self._origin is None:
            return
        self._store(self.diarizer.finish_utterance())
        self._coverage.append((self._origin, self._last_end))
        self._origin = None

    def _append(self, audio: np.ndarray, start: int) -> None:
        # VAD pre-roll may include the previous utterance's trailing silence.
        trim = max(0, self._last_end - start)
        audio = audio[trim:]
        start += trim
        if not len(audio):
            return
        if self._origin is not None and start != self._last_end:
            self._flush()
        if self._origin is None:
            self._origin = start
        self._store(self.diarizer.push(audio, sample_rate=self.sample_rate))
        self._last_end = start + len(audio)
        self._history_start = max(0, self._last_end - 120 * self.sample_rate)
        cutoff = self._history_start / self.sample_rate
        self._segments = deque(s for s in self._segments if s.end > cutoff)
        self._coverage = deque(span for span in self._coverage if span[1] > self._history_start)

    def _snapshot(self, start: int, end: int, *, include_open: bool = False) -> SpeakerAttribution:
        left, right = start / self.sample_rate, end / self.sample_rate
        segments = list(self._segments)
        coverage = list(self._coverage)
        if include_open and self._origin is not None:
            offset = self._origin / self.sample_rate
            segments.extend(
                SpeakerSegment(s.speaker, s.start + offset, s.end + offset) for s in self.diarizer.active_segments
            )
            coverage.append((self._origin, self._origin + round(self.diarizer.processed_seconds * self.sample_rate)))
        intervals = [
            SpeakerInterval(speaker=s.speaker, start=max(left, s.start) - left, end=min(right, s.end) - left)
            for s in segments
            if s.end > left and s.start < right
        ]
        cursor = max(start, self._history_start)
        for begin, finish in coverage:
            if begin > cursor:
                break
            if finish > cursor:
                cursor = finish
        return SpeakerAttribution(intervals=intervals, complete=start >= self._history_start and cursor >= end)

    def run(self) -> None:
        logger.info("Diarization worker running; waiting for VAD-selected speech")
        try:
            while not self.stop_event.is_set():
                try:
                    work = self.queue.get(timeout=0.1)
                except Empty:
                    if self._active_session is not None and self._active_session.invalid.is_set():
                        self.diarizer.reset()
                        self._clear_timeline()
                        self._active_session = None
                    continue
                if work.session.invalid.is_set():
                    continue
                try:
                    if work.session is not self._active_session:
                        self.diarizer.reset()
                        self._clear_timeline()
                        self._active_session = work.session
                    if work.audio is not None:
                        first_audio = self._origin is None and self._last_end == 0
                        started = perf_counter()
                        self._append(work.audio, work.start)
                        self._processing_seconds += perf_counter() - started
                        if first_audio:
                            logger.info("Diarization processing speech; first audio chunk processed")
                    else:
                        started = perf_counter()
                        queue_seconds = started - work.enqueued_at
                        if work.result is not None:
                            work.result.partial = self._snapshot(work.start, work.end, include_open=True)
                        flush_started = perf_counter()
                        self._flush()
                        flush_seconds = perf_counter() - flush_started
                        if work.result is not None:
                            result = self._snapshot(work.start, work.end)
                            finished = perf_counter()
                            processing_seconds = self._processing_seconds + finished - started
                            work.result.set_result(result)
                            logger.info(
                                "Diarization utterance processed: audio=%.2fs speakers=%s complete=%s queued_chunks=%d "
                                "durations=[%s] processing=%.3fs flush=%.3fs finalize_queue=%.3fs finalize_latency=%.3fs",
                                (work.end - work.start) / self.sample_rate,
                                ",".join(f"speaker_{s}" for s in sorted({i.speaker for i in result.intervals}))
                                or "unknown",
                                result.complete,
                                self.queue.qsize(),
                                result.durations_for_log(),
                                processing_seconds,
                                flush_seconds,
                                queue_seconds,
                                finished - work.enqueued_at,
                            )
                        self._processing_seconds = 0.0
                except Exception as exc:
                    work.session.invalid.set()
                    log_exception(logger, "Diarization failed; speaker labels unavailable until the next session", exc)
        finally:
            logger.info("Diarization worker stopped")
            self._session.invalid.set()
            self.diarizer.reset()
