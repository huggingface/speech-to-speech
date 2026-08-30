from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
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
        self.tts_ttfa_s = max(0.0, seconds)

    def record_e2e(self, seconds: float) -> None:
        self.e2e_s = max(0.0, seconds)

    def record_mlx_lock_wait(self, seconds: float) -> None:
        if seconds > 0.0:
            self.mlx_lock_wait_s += max(0.0, seconds)

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
    """Thread-safe latency trackers keyed by (turn_id, turn_revision)."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._trackers: dict[tuple[str, int], TurnLatencyTracker] = {}

    @staticmethod
    def _key(turn_id: str, turn_revision: int | None) -> tuple[str, int]:
        return turn_id, 0 if turn_revision is None else turn_revision

    def get_or_create(
        self,
        turn_id: str | None,
        turn_revision: int | None,
    ) -> TurnLatencyTracker | None:
        if turn_id is None:
            return None
        key = self._key(turn_id, turn_revision)
        with self._lock:
            tracker = self._trackers.get(key)
            if tracker is None:
                tracker = TurnLatencyTracker(turn_id=turn_id, turn_revision=key[1])
                self._trackers[key] = tracker
            return tracker

    def pop(
        self,
        turn_id: str | None,
        turn_revision: int | None,
    ) -> TurnLatencyTracker | None:
        if turn_id is None:
            return None
        key = self._key(turn_id, turn_revision)
        with self._lock:
            return self._trackers.pop(key, None)
