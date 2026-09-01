from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import Literal

TurnLatencyStatus = Literal["completed", "cancelled", "failed", "incomplete"]

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
    turn_id: str | None = None
    turn_revision: int | None = None
    stt_s: float | None = None
    llm_s: float | None = None
    tts_ttfa_s: float | None = None
    e2e_s: float | None = None
    mlx_lock_wait_s: float = 0.0
    status: TurnLatencyStatus = "completed"

    def reset(
        self,
        turn_id: str | None = None,
        turn_revision: int | None = None,
    ) -> None:
        self.turn_id = turn_id
        self.turn_revision = turn_revision
        self.stt_s = None
        self.llm_s = None
        self.tts_ttfa_s = None
        self.e2e_s = None
        self.mlx_lock_wait_s = 0.0
        self.status = "completed"

    def record_stt(self, seconds: float) -> None:
        self.stt_s = max(0.0, seconds)

    def record_llm(self, seconds: float) -> None:
        self.llm_s = max(0.0, seconds)

    def record_tts_ttfa(self, seconds: float) -> None:
        if self.tts_ttfa_s is None:
            self.tts_ttfa_s = max(0.0, seconds)

    def record_e2e(self, seconds: float) -> None:
        if self.e2e_s is None:
            self.e2e_s = max(0.0, seconds)

    def record_mlx_lock_wait(self, seconds: float) -> None:
        if seconds > 0.0:
            self.mlx_lock_wait_s += max(0.0, seconds)

    def absorb_pending(self, pending: TurnLatencyTracker) -> None:
        if pending.stt_s is not None:
            self.stt_s = pending.stt_s
        self.mlx_lock_wait_s += pending.mlx_lock_wait_s

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
            f"stt={self._fmt(self.stt_s)} llm={self._fmt(self.llm_s)} "
            f"tts_ttfa={self._fmt(self.tts_ttfa_s)} e2e={self._fmt(self.e2e_s)} "
            f"mlx_lock_wait={self.mlx_lock_wait_s:.2f}s status={self.status}"
        )


class TurnLatencyStore:
    """Thread-safe latency trackers keyed by response_key.

    STT runs before a response_key exists, so interim measurements are held on
    a per-turn pending slot and merged when the response tracker is created.

    Session cleanup: response trackers are indexed by ``session_id`` so
    ``unregister`` can drop only that session's in-flight measurements.
    Pending turn slots are not session-tagged (STT runs on the shared pipeline
    without conn context), so they are cleared only once the last tracked
    session has been removed — which matches the one-active-session-per-pipeline
    unit model used by the realtime server.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._trackers: dict[str, TurnLatencyTracker] = {}
        self._pending_turn: dict[tuple[str, int], TurnLatencyTracker] = {}
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

    def pop(self, response_key: str | None, *, session_id: str | None = None) -> TurnLatencyTracker | None:
        if response_key is None:
            return None
        with self._lock:
            tracker = self._trackers.pop(response_key, None)
            if session_id is not None:
                self._detach_response_from_session(session_id, response_key)
            return tracker

    def discard_response(self, response_key: str, *, session_id: str | None = None) -> None:
        self.pop(response_key, session_id=session_id)

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            for response_key in self._session_keys.pop(session_id, set()):
                self._trackers.pop(response_key, None)
            if not self._session_keys:
                self._pending_turn.clear()
