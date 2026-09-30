from __future__ import annotations

from collections import defaultdict
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
class ComputeLockStat:
    """Per-handler compute-lock total, with the number of acquisitions behind it."""

    total_s: float = 0.0
    count: int = 0

    def add(self, seconds: float) -> None:
        self.total_s += seconds
        self.count += 1

    def merge(self, other: ComputeLockStat) -> None:
        self.total_s += other.total_s
        self.count += other.count


@dataclass
class TurnLatencyTracker:
    """Server timings, not an additive breakdown or client playback latency.

    ``vad_settle_s`` is the final STT input gate. ``llm_ttft_s`` is the first
    text token; ``llm_s`` remains the full generation, including lock waits.
    ``tts_ttfa_s`` ends at the first model audio chunk before trimming and
    block assembly; ``e2e_s`` ends at the first yielded TTS audio. Compute-lock
    waits and holds are summed per handler, so the bracketed breakdown accounts
    for the printed total.

    ``tts=cut`` marks a cancelled turn that had already produced TTS audio. It
    says the cancellation arrived after audio started, not that a sentence was
    truncated: a cancellation landing just after the last chunk is marked too.

    Tool follow-ups use separate trackers.
    """

    turn_id: str | None = None
    turn_revision: int | None = None
    stt_s: float | None = None
    vad_settle_s: float | None = None
    llm_s: float | None = None
    llm_ttft_s: float | None = None
    tts_ttfa_s: float | None = None
    e2e_s: float | None = None
    lock_wait_s: float = 0.0
    lock_hold_s: float = 0.0
    lock_waits: dict[str, ComputeLockStat] = field(default_factory=dict)
    lock_holds: dict[str, ComputeLockStat] = field(default_factory=dict)
    status: TurnLatencyStatus = "completed"

    def record_stt(self, seconds: float) -> None:
        self.stt_s = max(0.0, seconds)

    def record_llm(self, seconds: float) -> None:
        self.llm_s = max(0.0, seconds)

    def record_vad_settle(self, seconds: float) -> None:
        self.vad_settle_s = max(0.0, seconds)

    def record_llm_ttft(self, seconds: float) -> None:
        if self.llm_ttft_s is None:
            self.llm_ttft_s = max(0.0, seconds)

    def record_tts_ttfa(self, seconds: float) -> None:
        if self.tts_ttfa_s is None:
            self.tts_ttfa_s = max(0.0, seconds)

    def record_e2e(self, seconds: float) -> None:
        if self.e2e_s is None:
            self.e2e_s = max(0.0, seconds)

    def record_lock_wait(self, seconds: float, handler_name: str | None = None) -> None:
        if seconds > 0.0:
            self.lock_wait_s += seconds
            if handler_name:
                self.lock_waits.setdefault(handler_name, ComputeLockStat()).add(seconds)

    def record_lock_hold(self, seconds: float, handler_name: str | None = None) -> None:
        if seconds > 0.0:
            self.lock_hold_s += seconds
            if handler_name:
                self.lock_holds.setdefault(handler_name, ComputeLockStat()).add(seconds)

    def absorb_pending(self, pending: TurnLatencyTracker) -> None:
        if pending.stt_s is not None:
            self.stt_s = pending.stt_s
        if pending.vad_settle_s is not None:
            self.vad_settle_s = pending.vad_settle_s
        self.lock_wait_s += pending.lock_wait_s
        self.lock_hold_s += pending.lock_hold_s
        for name, stat in pending.lock_waits.items():
            self.lock_waits.setdefault(name, ComputeLockStat()).merge(stat)
        for name, stat in pending.lock_holds.items():
            self.lock_holds.setdefault(name, ComputeLockStat()).merge(stat)

    @staticmethod
    def _fmt(seconds: float | None) -> str:
        if seconds is None:
            return "n/a"
        return f"{seconds:.2f}s"

    def format_log_line(self) -> str | None:
        if self.turn_id is None:
            return None
        revision = 0 if self.turn_revision is None else self.turn_revision
        fields = [
            f"stt={self._fmt(self.stt_s)}",
            f"llm={self._fmt(self.llm_s)}",
            f"tts_ttfa={self._fmt(self.tts_ttfa_s)}",
            f"e2e={self._fmt(self.e2e_s)}",
        ]
        if self.vad_settle_s is not None:
            fields.insert(1, f"vad_settle={self._fmt(self.vad_settle_s)}")
        if self.llm_ttft_s is not None:
            llm_index = fields.index(f"llm={self._fmt(self.llm_s)}")
            fields.insert(llm_index + 1, f"llm_ttft={self._fmt(self.llm_ttft_s)}")
        fields.append(f"lock_wait={self._fmt_lock(self.lock_wait_s, self.lock_waits)}")
        if self.lock_hold_s > 0.0:
            fields.append(f"lock_hold={self._fmt_lock(self.lock_hold_s, self.lock_holds)}")
        line = f"Turn {self.turn_id} rev={revision} latency: {' '.join(fields)} status={self.status}"
        if self.status == "cancelled" and self.tts_ttfa_s is not None:
            line += " tts=cut"
        return line

    @staticmethod
    def _fmt_lock(total: float, stats: dict[str, ComputeLockStat]) -> str:
        if not stats:
            return f"{total:.2f}s"
        named = ",".join(
            f"{name}:{stat.total_s:.2f}s" + (f"x{stat.count}" if stat.count > 1 else "") for name, stat in stats.items()
        )
        return f"{total:.2f}s[{named}]"


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

    def pending_for_turn(self, turn_id: str | None, turn_revision: int | None) -> TurnLatencyTracker | None:
        """Read the pending slot without creating one."""
        if turn_id is None:
            return None
        with self._lock:
            return self._pending_turn.get(self._turn_key(turn_id, turn_revision))

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
