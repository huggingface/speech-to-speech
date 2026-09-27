from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import Literal

TurnLatencyStatus = Literal["completed", "cancelled", "failed", "incomplete"]
TURN_LATENCY_METADATA_KEY = "speech_to_speech.turn_latency"

_active_tracker: ContextVar[TurnLatencyTracker | None] = ContextVar(
    "active_turn_latency_tracker",
    default=None,
)


def active_turn_latency_tracker() -> TurnLatencyTracker | None:
    return _active_tracker.get()


@contextmanager
def bind_active_turn_latency_tracker(tracker: TurnLatencyTracker | None):
    token = _active_tracker.set(tracker)
    try:
        yield tracker
    finally:
        _active_tracker.reset(token)


@dataclass
class TurnLatencyTracker:
    """Server timings, not an additive breakdown or client playback latency.

    ``llm_ttft_s`` ends at the first non-whitespace provider text delta and
    ``llm_s`` covers full generation, including lock waits. ``tts_ttfa_s`` ends
    at the first model audio chunk before trimming and block assembly;
    ``e2e_s`` ends at the first yielded TTS audio. Tool follow-ups use separate
    trackers.
    """

    turn_id: str | None = None
    turn_revision: int | None = None
    stt_s: float | None = None
    llm_ttft_s: float | None = None
    llm_s: float | None = None
    tts_ttfa_s: float | None = None
    e2e_s: float | None = None
    vad_decision_s: float | None = None
    smart_turn_analysis_s: float | None = None
    smart_turn_status: Literal["disabled", "complete", "incomplete", "failed"] | None = None
    smart_turn_grace_s: float | None = None
    smart_turn_processing_delay_s: float | None = None
    smart_turn_wait_s: float | None = None
    _smart_wait_intervals: list[tuple[float, float]] | None = None
    mlx_lock_wait_s: float = 0.0
    status: TurnLatencyStatus = "completed"

    def record_stt(self, seconds: float) -> None:
        self.stt_s = max(0.0, seconds)

    def record_llm(self, seconds: float) -> None:
        self.llm_s = max(0.0, seconds)

    def record_llm_ttft(self, seconds: float) -> None:
        if self.llm_ttft_s is None:
            self.llm_ttft_s = max(0.0, seconds)

    def record_tts_ttfa(self, seconds: float) -> None:
        if self.tts_ttfa_s is None:
            self.tts_ttfa_s = max(0.0, seconds)

    def record_e2e(self, seconds: float) -> None:
        if self.e2e_s is None:
            self.e2e_s = max(0.0, seconds)

    def record_mlx_lock_wait(self, seconds: float) -> None:
        if seconds > 0.0:
            self.mlx_lock_wait_s += max(0.0, seconds)

    def record_smart_wait(self, started_at_s: float, ended_at_s: float) -> None:
        """Count the union of actual gate waits, since workers can wait together."""
        if self.smart_turn_status not in ("complete", "incomplete", "failed") or ended_at_s <= started_at_s:
            return
        intervals = self._smart_wait_intervals or []
        intervals.append((started_at_s, ended_at_s))
        intervals.sort()
        merged: list[tuple[float, float]] = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        self._smart_wait_intervals = merged
        self.smart_turn_wait_s = sum(end - start for start, end in merged)

    def metadata_payload(
        self,
        *,
        response_key: str,
        status: TurnLatencyStatus,
    ) -> dict[str, float | int | str | None]:
        """Return terminal measurements; new fields use nanosecond precision."""

        def compact(seconds: float | None) -> float | None:
            return None if seconds is None else round(seconds, 9)

        return {
            "version": 1,
            "turn_id": self.turn_id,
            "turn_revision": 0 if self.turn_revision is None else self.turn_revision,
            "response_key": response_key,
            "stt_s": self.stt_s,
            "llm_ttft_s": self.llm_ttft_s,
            "llm_s": self.llm_s,
            "tts_ttfa_s": self.tts_ttfa_s,
            "e2e_s": self.e2e_s,
            "vad_decision_s": compact(self.vad_decision_s),
            "smart_analysis_s": compact(self.smart_turn_analysis_s),
            "smart_status": self.smart_turn_status,
            "smart_grace_s": compact(self.smart_turn_grace_s),
            "smart_delay_s": compact(self.smart_turn_processing_delay_s),
            "smart_wait_s": compact(self.smart_turn_wait_s),
            "mlx_lock_wait_s": self.mlx_lock_wait_s,
            "status": status,
        }

    def absorb_pending(self, pending: TurnLatencyTracker) -> None:
        if pending.stt_s is not None:
            self.stt_s = pending.stt_s
        self.mlx_lock_wait_s += pending.mlx_lock_wait_s
        self.vad_decision_s = pending.vad_decision_s
        self.smart_turn_analysis_s = pending.smart_turn_analysis_s
        self.smart_turn_status = pending.smart_turn_status
        self.smart_turn_grace_s = pending.smart_turn_grace_s
        self.smart_turn_processing_delay_s = pending.smart_turn_processing_delay_s
        self.smart_turn_wait_s = pending.smart_turn_wait_s
        self._smart_wait_intervals = pending._smart_wait_intervals

    @staticmethod
    def _fmt(seconds: float | None) -> str:
        if seconds is None:
            return "n/a"
        return f"{seconds:.2f}s"

    def format_log_line(self) -> str | None:
        if self.turn_id is None:
            return None
        revision = 0 if self.turn_revision is None else self.turn_revision
        return (
            f"Turn {self.turn_id} rev={revision} latency: "
            f"stt={self._fmt(self.stt_s)} llm_ttft={self._fmt(self.llm_ttft_s)} "
            f"llm={self._fmt(self.llm_s)} "
            f"tts_ttfa={self._fmt(self.tts_ttfa_s)} e2e={self._fmt(self.e2e_s)} "
            f"vad_decision={self._fmt(self.vad_decision_s)} "
            f"smart_turn_analysis={self._fmt(self.smart_turn_analysis_s)} "
            f"smart_turn_wait={self._fmt(self.smart_turn_wait_s)} "
            f"smart_turn_status={self.smart_turn_status or 'n/a'} "
            f"mlx_lock_wait={self.mlx_lock_wait_s:.2f}s status={self.status}"
        )


class TurnLatencyStore:
    """Thread-safe latency trackers keyed by response_key.

    The realtime service creates response trackers before queueing work.
    Workers only look them up so cancelled work cannot recreate a tracker.

    STT runs before a response_key exists, so interim measurements are held on
    a per-turn pending slot and merged when the response tracker is created.

    Session cleanup: response trackers are indexed by ``session_id`` so
    ``unregister`` can drop only that session's in-flight measurements.
    Empty or stale final transcripts discard their pending turn slots. Remaining
    slots are not session-tagged (STT runs without conn context), so teardown
    clears them once the last tracked session has been removed — which matches
    the one-active-session-per-pipeline unit model used by the realtime server.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._trackers: dict[str, TurnLatencyTracker] = {}
        self._pending_turn: dict[tuple[str, int], TurnLatencyTracker] = {}
        self._turn_responses: dict[tuple[str, int], str] = {}
        self._session_keys: dict[str, set[str]] = defaultdict(set)

    @property
    def active_session_count(self) -> int:
        with self._lock:
            return len(self._session_keys)

    @staticmethod
    def _turn_key(turn_id: str, turn_revision: int | None) -> tuple[str, int]:
        return turn_id, 0 if turn_revision is None else turn_revision

    def _detach_response_from_session(self, session_id: str, response_key: str) -> None:
        keys = self._session_keys.get(session_id)
        if keys is None:
            return
        keys.discard(response_key)
        if not keys:
            self._session_keys.pop(session_id, None)

    def get_or_create_for_turn(
        self,
        turn_id: str | None,
        turn_revision: int | None,
    ) -> TurnLatencyTracker | None:
        if turn_id is None:
            return None
        key = self._turn_key(turn_id, turn_revision)
        with self._lock:
            tracker = self._pending_turn.get(key)
            if tracker is None:
                tracker = TurnLatencyTracker(turn_id=turn_id, turn_revision=key[1])
                self._pending_turn[key] = tracker
            return tracker

    def discard_pending_turn(self, turn_id: str | None, turn_revision: int | None) -> None:
        """Drop final STT measurements that will not be consumed by a response."""
        if turn_id is None:
            return
        with self._lock:
            self._pending_turn.pop(self._turn_key(turn_id, turn_revision), None)

    def get_or_create_response(
        self,
        response_key: str,
        *,
        turn_id: str | None = None,
        turn_revision: int | None = None,
        session_id: str | None = None,
    ) -> TurnLatencyTracker:
        with self._lock:
            tracker = self._trackers.get(response_key)
            if tracker is None:
                revision = None if turn_revision is None else turn_revision
                tracker = TurnLatencyTracker(turn_id=turn_id, turn_revision=revision)
                self._trackers[response_key] = tracker
                if session_id is not None:
                    self._session_keys[session_id].add(response_key)
                if turn_id is not None:
                    self._turn_responses.setdefault(self._turn_key(turn_id, turn_revision), response_key)
                    pending = self._pending_turn.pop(self._turn_key(turn_id, turn_revision), None)
                    if pending is not None:
                        tracker.absorb_pending(pending)
                        if tracker.turn_id is None:
                            tracker.turn_id = pending.turn_id
                            tracker.turn_revision = pending.turn_revision
            elif turn_id is not None and tracker.turn_id is None:
                tracker.turn_id = turn_id
                tracker.turn_revision = turn_revision
            return tracker

    def record_smart_wait(
        self, turn_id: str | None, turn_revision: int | None, started_at_s: float, ended_at_s: float
    ) -> None:
        if turn_id is None:
            return
        with self._lock:
            key = self._turn_key(turn_id, turn_revision)
            response_key = self._turn_responses.get(key)
            tracker = self._trackers.get(response_key) if response_key is not None else self._pending_turn.get(key)
            if tracker is not None:
                tracker.record_smart_wait(started_at_s, ended_at_s)

    def get_response(self, response_key: str | None) -> TurnLatencyTracker | None:
        """Look up an existing response without reviving cancelled measurements."""
        if response_key is None:
            return None
        with self._lock:
            return self._trackers.get(response_key)

    def pop(self, response_key: str | None, *, session_id: str | None = None) -> TurnLatencyTracker | None:
        if response_key is None:
            return None
        with self._lock:
            tracker = self._trackers.pop(response_key, None)
            if tracker is not None and tracker.turn_id is not None:
                key = self._turn_key(tracker.turn_id, tracker.turn_revision)
                if self._turn_responses.get(key) == response_key:
                    self._turn_responses.pop(key, None)
            if session_id is not None:
                self._detach_response_from_session(session_id, response_key)
            return tracker

    def discard_response(self, response_key: str, *, session_id: str | None = None) -> None:
        self.pop(response_key, session_id=session_id)

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            for response_key in self._session_keys.pop(session_id, set()):
                tracker = self._trackers.pop(response_key, None)
                if tracker is not None and tracker.turn_id is not None:
                    key = self._turn_key(tracker.turn_id, tracker.turn_revision)
                    if self._turn_responses.get(key) == response_key:
                        self._turn_responses.pop(key, None)
            if not self._session_keys:
                self._pending_turn.clear()
