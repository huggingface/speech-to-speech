from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import Condition

logger = logging.getLogger(__name__)


class TurnPhase(str, Enum):
    LISTENING = "listening"
    SOFT_ENDED = "soft_ended"
    ANSWERING = "answering"
    CLOSED = "closed"


class TurnGateAction(str, Enum):
    ACCEPT = "accept"
    DROP = "drop"
    HOLD = "hold"


@dataclass(frozen=True)
class TurnGate:
    """Decision for turn-tagged work; a HOLD carries its monotonic deadline."""

    action: TurnGateAction
    deadline: float | None = None


class TurnOutputHeld(Exception):
    """Output reached its commit while its turn was held; keep it queued."""


_ACCEPT = TurnGate(TurnGateAction.ACCEPT)
_DROP = TurnGate(TurnGateAction.DROP)


@dataclass(frozen=True)
class _TurnReference:
    sequence: int
    turn_id: str
    revision: int


@dataclass(frozen=True)
class _PendingReopen:
    turn_id: str
    base_revision: int
    candidate_revision: int
    # Output stops waiting at this deadline; the candidate stays until VAD resolves it.
    deadline: float


@dataclass(frozen=True)
class _ReopenGrace:
    turn_id: str
    revision: int
    deadline: float


class SpeculativeTurnTracker:
    """Thread-safe conversation cursor for raw-audio speculative turns."""

    _PENDING_REOPEN_HOLD_S = 2.0

    def __init__(self) -> None:
        self._condition = Condition()
        self._sequence = 0
        self._current: _TurnReference | None = None
        self._committed: set[_TurnReference] = set()
        self._closed_current: _TurnReference | None = None
        self._pending_reopen: _PendingReopen | None = None
        self._reopen_grace: _ReopenGrace | None = None
        self._phase: TurnPhase | None = None
        self._last_final_audio_ms: int | None = None
        self._processing_deadline: float | None = None
        self._unanswered_reopen_ms = 7000
        self.wait_observer: Callable[[str, int, float, float], None] | None = None

    @property
    def phase(self) -> TurnPhase | None:
        with self._condition:
            return self._phase

    def current_turn(self) -> tuple[str | None, int | None]:
        with self._condition:
            if self._current is None:
                return None, None
            return self._current.turn_id, self._current.revision

    def configure_reopen(self, unanswered_reopen_ms: int) -> None:
        with self._condition:
            self._unanswered_reopen_ms = max(0, unanswered_reopen_ms)

    def can_reopen(self, audio_ms: int) -> bool:
        """Use streamed-audio time, not wall time, for the unanswered-turn cap."""
        with self._condition:
            return self._can_reopen_locked(audio_ms)

    def _can_reopen_locked(self, audio_ms: int) -> bool:
        return (
            self._current is not None
            and self._phase == TurnPhase.SOFT_ENDED
            and self._last_final_audio_ms is not None
            and not self._blocks_reopen_locked(self._current.turn_id, self._current.revision)
            and max(0, audio_ms - self._last_final_audio_ms) <= self._unanswered_reopen_ms
        )

    def speech_candidate_started(self, audio_ms: int) -> bool:
        """Hold output while VAD checks whether a resumed fragment is speech."""
        with self._condition:
            if self._pending_reopen is not None:
                return True
            if not self._can_reopen_locked(audio_ms):
                return False
            assert self._current is not None
            return self.begin_reopen_candidate(self._current.turn_id, self._current.revision) is not None

    def speech_candidate_cancelled(self) -> None:
        with self._condition:
            if self._pending_reopen is not None:
                self.cancel_reopen_candidate(self._pending_reopen.turn_id)

    def has_speech_candidate(self) -> bool:
        with self._condition:
            return self._pending_reopen is not None

    def speech_started(self, audio_ms: int) -> tuple[str, int, bool]:
        """Choose a new turn or reopen the unanswered current turn atomically."""
        with self._condition:
            if self._current is not None and self._phase == TurnPhase.LISTENING:
                return self._current.turn_id, self._current.revision, False
            pending = self._pending_reopen
            if pending is None and self._can_reopen_locked(audio_ms):
                assert self._current is not None
                self.begin_reopen_candidate(self._current.turn_id, self._current.revision)
                pending = self._pending_reopen
            if pending is not None and self.confirm_reopen_candidate(
                pending.turn_id, pending.base_revision, pending.candidate_revision
            ):
                return pending.turn_id, pending.candidate_revision, True
            turn_id, revision = self.start_turn()
            return turn_id, revision, False

    def segment_finalized(self, audio_ms: int, output_hold_ms: int = 0, processing_delay_ms: int = 0) -> None:
        """Record audio timing and both Smart Turn deadlines for this revision."""
        with self._condition:
            if self._current is None or self._blocks_reopen_locked(self._current.turn_id, self._current.revision):
                return
            self._phase = TurnPhase.SOFT_ENDED
            self._last_final_audio_ms = audio_ms
            now = time.monotonic()
            self._processing_deadline = now + max(0, processing_delay_ms) / 1000.0
            self.start_reopen_grace(self._current.turn_id, self._current.revision, max(0, output_hold_ms) / 1000.0)
            self._condition.notify_all()

    def segment_discarded(self) -> None:
        """End detected speech that VAD cannot release as a valid segment."""
        with self._condition:
            self.speech_candidate_cancelled()
            if self._phase == TurnPhase.LISTENING:
                self._phase = TurnPhase.SOFT_ENDED
                self._condition.notify_all()

    def processing_deadline(self, turn_id: str | None, revision: int | None) -> float | None:
        """Return the monotonic deadline, or None for legacy unfinalized input."""
        with self._condition:
            if turn_id is None or revision is None or not self._is_current_locked(turn_id, revision):
                return None
            return self._processing_deadline

    def start_turn(self) -> tuple[str, int]:
        """Advance the conversation cursor and return the new turn metadata."""
        with self._condition:
            self._sequence += 1
            self._current = _TurnReference(
                sequence=self._sequence,
                turn_id=f"turn_{self._sequence}",
                revision=0,
            )
            self._closed_current = None
            self._phase = TurnPhase.LISTENING
            self._last_final_audio_ms = None
            self._processing_deadline = None
            self._pending_reopen = None
            self._reopen_grace = None
            logger.debug("Started speculative turn %s", self._current.turn_id)
            self._condition.notify_all()
            return self._current.turn_id, self._current.revision

    def observe(self, turn_id: str | None, revision: int | None) -> None:
        """Compatibility adapter for revisions created outside the tracker.

        New conversation turns must use :meth:`start_turn`. Existing call sites
        may still report a newer revision for the current turn.
        """
        if turn_id is None or revision is None:
            return
        with self._condition:
            if self._current is None:
                self._sequence += 1
                self._current = _TurnReference(self._sequence, turn_id, revision)
                self._phase = TurnPhase.LISTENING
            elif (
                self._current.turn_id == turn_id
                and revision > self._current.revision
                and not self._blocks_reopen_locked(turn_id, self._current.revision)
            ):
                self._current = _TurnReference(self._current.sequence, turn_id, revision)
                self._closed_current = None
                self._phase = TurnPhase.LISTENING
                self._pending_reopen = None
                self._reopen_grace = None
                self._processing_deadline = None
            else:
                return
            logger.debug("Observed speculative turn %s revision %d", turn_id, revision)
            self._condition.notify_all()

    def is_latest(self, turn_id: str | None, revision: int | None) -> bool:
        if turn_id is None or revision is None:
            return True
        with self._condition:
            return self._is_relevant_locked(turn_id, revision)

    def gate(
        self,
        turn_id: str | None,
        revision: int | None,
        *,
        hold_for_grace: bool = True,
        hold_until: float | None = None,
        commit: bool = False,
    ) -> TurnGate:
        """Decide without waiting whether turn-tagged work may proceed.

        HOLD lasts while VAD checks resumed speech, during the reopen grace
        when *hold_for_grace* is set, and until *hold_until*. Committed work
        is never held. With *commit*, ACCEPT commits the turn.
        """
        if turn_id is None or revision is None:
            return _ACCEPT
        with self._condition:
            return self._gate_locked(turn_id, revision, hold_for_grace, hold_until, commit)

    def wait_for_gate(
        self,
        turn_id: str | None,
        revision: int | None,
        *,
        hold_for_grace: bool = True,
        hold_until: float | None = None,
        commit: bool = False,
    ) -> bool:
        """Block a worker thread until :meth:`gate` accepts or drops the work.

        Never call this from the event loop; it waits on tracker state changes
        and falls back to the hold deadline.
        """
        if turn_id is None or revision is None:
            return True
        with self._condition:
            while True:
                decision = self._gate_locked(turn_id, revision, hold_for_grace, hold_until, commit)
                if decision.action is not TurnGateAction.HOLD:
                    return decision.action is TurnGateAction.ACCEPT
                assert decision.deadline is not None
                waiting_for_grace = hold_until is None and not self._pending_holds_locked(turn_id, revision)
                wait_started_at_s = time.perf_counter()
                self._condition.wait(max(0.0, decision.deadline - time.monotonic()))
                if waiting_for_grace and self.wait_observer is not None:
                    self.wait_observer(turn_id, revision, wait_started_at_s, time.perf_counter())

    def start_reopen_grace(self, turn_id: str | None, revision: int | None, grace_s: float) -> None:
        if turn_id is None or revision is None or grace_s <= 0:
            return
        with self._condition:
            if not self._is_current_locked(turn_id, revision) or self._blocks_reopen_locked(turn_id, revision):
                return
            deadline = time.monotonic() + grace_s
            existing = self._reopen_grace
            if (
                existing is None
                or existing.turn_id != turn_id
                or existing.revision != revision
                or deadline > existing.deadline
            ):
                self._reopen_grace = _ReopenGrace(turn_id, revision, deadline)
                logger.debug(
                    "Started speculative reopen grace for turn %s revision %d: %.0fms",
                    turn_id,
                    revision,
                    grace_s * 1000,
                )
                self._condition.notify_all()

    def commit(self, turn_id: str | None, revision: int | None) -> bool:
        """Accept relevant work now; callers consult :meth:`gate` first."""
        if turn_id is None or revision is None:
            return True
        with self._condition:
            return self._commit_locked(turn_id, revision)

    def close(self, turn_id: str | None, revision: int | None) -> None:
        """Release committed state after a response reaches its terminal event.

        A closed turn that is still current stays answered: tool follow-ups
        and client-requested responses for it remain valid, and it cannot
        reopen. The next turn makes it stale.
        """
        if turn_id is None or revision is None:
            return
        with self._condition:
            committed = self._committed_reference_locked(turn_id, revision)
            if committed is None:
                return
            self._committed.remove(committed)
            if self._current == committed:
                self._closed_current = committed
                self._phase = TurnPhase.CLOSED
                self._processing_deadline = None
            self._condition.notify_all()

    def is_current_turn(self, turn_id: str | None) -> bool:
        """Return whether *turn_id* is the conversation's current turn at any revision."""
        with self._condition:
            return self._current is not None and self._current.turn_id == turn_id

    def is_committed(self, turn_id: str | None, revision: int | None = None) -> bool:
        if turn_id is None:
            return False
        with self._condition:
            answered = self._committed | ({self._closed_current} if self._closed_current is not None else set())
            return any(
                reference.turn_id == turn_id and (revision is None or reference.revision == revision)
                for reference in answered
            )

    def begin_reopen_candidate(self, turn_id: str | None, revision: int | None) -> int | None:
        if turn_id is None or revision is None:
            return None
        with self._condition:
            if not self._is_current_locked(turn_id, revision) or self._blocks_reopen_locked(turn_id, revision):
                return None
            pending = self._pending_reopen
            if pending is not None:
                if pending.turn_id == turn_id and pending.base_revision == revision:
                    return pending.candidate_revision
                return None
            candidate_revision = revision + 1
            self._pending_reopen = _PendingReopen(
                turn_id,
                revision,
                candidate_revision,
                deadline=time.monotonic() + self._PENDING_REOPEN_HOLD_S,
            )
            logger.debug(
                "Started speculative reopen candidate for turn %s revision %d -> %d",
                turn_id,
                revision,
                candidate_revision,
            )
            self._condition.notify_all()
            return candidate_revision

    def confirm_reopen_candidate(
        self,
        turn_id: str | None,
        base_revision: int | None,
        candidate_revision: int | None,
    ) -> bool:
        if turn_id is None or base_revision is None or candidate_revision is None:
            return False
        with self._condition:
            pending = self._pending_reopen
            if pending is None or (pending.turn_id, pending.base_revision, pending.candidate_revision) != (
                turn_id,
                base_revision,
                candidate_revision,
            ):
                return False
            if not self._is_current_locked(turn_id, base_revision) or self._blocks_reopen_locked(
                turn_id,
                base_revision,
            ):
                self._pending_reopen = None
                self._condition.notify_all()
                return False
            assert self._current is not None
            self._current = _TurnReference(self._current.sequence, turn_id, candidate_revision)
            self._phase = TurnPhase.LISTENING
            self._processing_deadline = None
            self._pending_reopen = None
            self._reopen_grace = None
            logger.debug(
                "Confirmed speculative reopen candidate for turn %s revision %d",
                turn_id,
                candidate_revision,
            )
            self._condition.notify_all()
            return True

    def cancel_reopen_candidate(self, turn_id: str | None, candidate_revision: int | None = None) -> None:
        if turn_id is None:
            return
        with self._condition:
            pending = self._pending_reopen
            if pending is None or pending.turn_id != turn_id:
                return
            if candidate_revision is not None and pending.candidate_revision != candidate_revision:
                return
            self._pending_reopen = None
            logger.debug("Cancelled speculative reopen candidate for turn %s", turn_id)
            self._condition.notify_all()

    def _commit_locked(self, turn_id: str, revision: int) -> bool:
        committed = self._committed_reference_locked(turn_id, revision)
        if committed is not None:
            return True
        if self._current is None:
            return False
        if not self._is_current_locked(turn_id, revision):
            return False
        self._committed.add(self._current)
        self._phase = TurnPhase.ANSWERING
        self._processing_deadline = None
        if self._grace_matches_locked(turn_id, revision):
            self._reopen_grace = None
        logger.debug("Committed speculative turn %s revision %d", turn_id, revision)
        self._condition.notify_all()
        return True

    def _is_current_locked(self, turn_id: str, revision: int) -> bool:
        return self._current is not None and (self._current.turn_id == turn_id and self._current.revision == revision)

    def _is_relevant_locked(self, turn_id: str, revision: int) -> bool:
        if self._current is None:
            return False
        return (
            self._is_current_locked(turn_id, revision)
            or self._committed_reference_locked(turn_id, revision) is not None
        )

    def _blocks_reopen_locked(self, turn_id: str, revision: int) -> bool:
        """Return whether accepted or finished output makes reopening unsafe."""
        return self._is_closed_current_locked(turn_id, revision) or (
            self._committed_reference_locked(turn_id, revision) is not None
        )

    def _is_closed_current_locked(self, turn_id: str, revision: int) -> bool:
        return self._closed_current is not None and (
            self._closed_current.turn_id == turn_id and self._closed_current.revision == revision
        )

    def _committed_reference_locked(self, turn_id: str, revision: int) -> _TurnReference | None:
        return next(
            (
                reference
                for reference in self._committed
                if reference.turn_id == turn_id and reference.revision == revision
            ),
            None,
        )

    def _gate_locked(
        self,
        turn_id: str,
        revision: int,
        hold_for_grace: bool,
        hold_until: float | None,
        commit: bool,
    ) -> TurnGate:
        if not self._is_relevant_locked(turn_id, revision):
            return _DROP
        if self._committed_reference_locked(turn_id, revision) is None:
            now = time.monotonic()
            deadlines = [deadline for deadline in (hold_until,) if deadline is not None and deadline > now]
            if self._pending_holds_locked(turn_id, revision):
                assert self._pending_reopen is not None
                deadlines.append(self._pending_reopen.deadline)
            grace = self._reopen_grace
            if (
                hold_for_grace
                and grace is not None
                and self._grace_matches_locked(turn_id, revision)
                and self._is_current_locked(turn_id, revision)
                and grace.deadline > now
            ):
                deadlines.append(grace.deadline)
            if deadlines:
                return TurnGate(TurnGateAction.HOLD, min(deadlines))
        if commit:
            self._commit_locked(turn_id, revision)
        return _ACCEPT

    def _pending_holds_locked(self, turn_id: str, revision: int) -> bool:
        pending = self._pending_reopen
        return (
            pending is not None
            and pending.turn_id == turn_id
            and pending.base_revision == revision
            and pending.deadline > time.monotonic()
        )

    def _grace_matches_locked(self, turn_id: str, revision: int) -> bool:
        grace = self._reopen_grace
        return grace is not None and grace.turn_id == turn_id and grace.revision == revision

    def reset(self) -> None:
        with self._condition:
            self._sequence = 0
            self._current = None
            self._phase = None
            self._last_final_audio_ms = None
            self._processing_deadline = None
            self._committed.clear()
            self._closed_current = None
            self._pending_reopen = None
            self._reopen_grace = None
            self._condition.notify_all()
