from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import Condition

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class _ReopenGrace:
    turn_id: str
    revision: int
    deadline: float


class SpeculativeTurnTracker:
    """Thread-safe conversation cursor for raw-audio speculative turns."""

    _PENDING_REOPEN_WAIT_TIMEOUT_S = 2.0

    def __init__(self) -> None:
        self._condition = Condition()
        self._sequence = 0
        self._current: _TurnReference | None = None
        self._committed: set[_TurnReference] = set()
        self._closed_current: _TurnReference | None = None
        self._pending_reopen: _PendingReopen | None = None
        self._reopen_grace: _ReopenGrace | None = None

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
            elif self._current.turn_id == turn_id and revision > self._current.revision:
                self._current = _TurnReference(self._current.sequence, turn_id, revision)
                self._closed_current = None
            else:
                return
            logger.debug("Observed speculative turn %s revision %d", turn_id, revision)
            self._condition.notify_all()

    def is_latest(self, turn_id: str | None, revision: int | None) -> bool:
        if turn_id is None or revision is None:
            return True
        with self._condition:
            return self._is_relevant_locked(turn_id, revision)

    def is_latest_after_pending_reopen(self, turn_id: str | None, revision: int | None) -> bool:
        if turn_id is None or revision is None:
            return True
        with self._condition:
            self._wait_for_pending_reopen_locked(turn_id, revision, self._PENDING_REOPEN_WAIT_TIMEOUT_S)
            return self._is_relevant_locked(turn_id, revision)

    def try_is_latest_after_pending_reopen(self, turn_id: str | None, revision: int | None) -> bool | None:
        """Return ``None`` when a matching reopen candidate is unresolved."""
        if turn_id is None or revision is None:
            return True
        with self._condition:
            if self._has_pending_reopen_locked(turn_id, revision):
                return None
            return self._is_relevant_locked(turn_id, revision)

    def is_latest_after_reopen_grace(self, turn_id: str | None, revision: int | None) -> bool:
        if turn_id is None or revision is None:
            return True
        with self._condition:
            self._wait_for_reopen_gate_locked(turn_id, revision)
            return self._is_relevant_locked(turn_id, revision)

    def try_is_latest_after_reopen_grace(self, turn_id: str | None, revision: int | None) -> bool | None:
        if turn_id is None or revision is None:
            return True
        with self._condition:
            if (
                self._has_pending_reopen_locked(turn_id, revision)
                or self._reopen_grace_remaining_locked(
                    turn_id,
                    revision,
                )
                > 0
            ):
                return None
            return self._is_relevant_locked(turn_id, revision)

    def commit_if_latest_after_pending_reopen(self, turn_id: str | None, revision: int | None) -> bool:
        if turn_id is None or revision is None:
            return True
        with self._condition:
            self._wait_for_pending_reopen_locked(turn_id, revision, self._PENDING_REOPEN_WAIT_TIMEOUT_S)
            return self._commit_locked(turn_id, revision)

    def commit_if_latest_after_reopen_grace(self, turn_id: str | None, revision: int | None) -> bool:
        if turn_id is None or revision is None:
            return True
        with self._condition:
            self._wait_for_reopen_gate_locked(turn_id, revision)
            return self._commit_locked(turn_id, revision)

    def try_commit_if_latest_after_pending_reopen(self, turn_id: str | None, revision: int | None) -> bool | None:
        """Return ``None`` when a matching reopen candidate is unresolved."""
        if turn_id is None or revision is None:
            return True
        with self._condition:
            if self._has_pending_reopen_locked(turn_id, revision):
                return None
            return self._commit_locked(turn_id, revision)

    def try_commit_if_latest_after_reopen_grace(self, turn_id: str | None, revision: int | None) -> bool | None:
        if turn_id is None or revision is None:
            return True
        with self._condition:
            if (
                self._has_pending_reopen_locked(turn_id, revision)
                or self._reopen_grace_remaining_locked(
                    turn_id,
                    revision,
                )
                > 0
            ):
                return None
            return self._commit_locked(turn_id, revision)

    def has_pending_reopen(self, turn_id: str | None, revision: int | None) -> bool:
        if turn_id is None or revision is None:
            return False
        with self._condition:
            return self._has_pending_reopen_locked(turn_id, revision)

    def has_pending_reopen_or_grace(self, turn_id: str | None, revision: int | None) -> bool:
        if turn_id is None or revision is None:
            return False
        with self._condition:
            return (
                self._has_pending_reopen_locked(turn_id, revision)
                or self._reopen_grace_remaining_locked(
                    turn_id,
                    revision,
                )
                > 0
            )

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

    def is_latest_after_stability_window(
        self,
        turn_id: str | None,
        revision: int | None,
        settle_s: float,
    ) -> bool:
        if turn_id is None or revision is None:
            return True
        if settle_s <= 0:
            return self.is_latest_after_pending_reopen(turn_id, revision)
        with self._condition:
            deadline = time.monotonic() + settle_s
            while self._is_relevant_locked(turn_id, revision):
                if self._has_pending_reopen_locked(turn_id, revision):
                    self._wait_for_pending_reopen_locked(
                        turn_id,
                        revision,
                        self._PENDING_REOPEN_WAIT_TIMEOUT_S,
                    )
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return self._is_relevant_locked(turn_id, revision)

    def commit(self, turn_id: str | None, revision: int | None) -> None:
        if turn_id is None or revision is None:
            return
        with self._condition:
            if self._has_pending_reopen_locked(turn_id, revision):
                logger.debug(
                    "Deferring speculative turn %s revision %d commit while reopen is pending",
                    turn_id,
                    revision,
                )
                return
            self._commit_locked(turn_id, revision)

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
            self._pending_reopen = _PendingReopen(turn_id, revision, candidate_revision)
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
            if pending != _PendingReopen(turn_id, base_revision, candidate_revision):
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

    def wait_for_pending_reopen(
        self,
        turn_id: str | None,
        revision: int | None,
        timeout_s: float = _PENDING_REOPEN_WAIT_TIMEOUT_S,
    ) -> None:
        if turn_id is None or revision is None:
            return
        with self._condition:
            self._wait_for_pending_reopen_locked(turn_id, revision, timeout_s)

    def _commit_locked(self, turn_id: str, revision: int) -> bool:
        committed = self._committed_reference_locked(turn_id, revision)
        if committed is not None:
            return True
        if self._current is None:
            return True
        if not self._is_current_locked(turn_id, revision):
            return False
        self._committed.add(self._current)
        if self._grace_matches_locked(turn_id, revision):
            self._reopen_grace = None
        logger.debug("Committed speculative turn %s revision %d", turn_id, revision)
        self._condition.notify_all()
        return True

    def _is_current_locked(self, turn_id: str, revision: int) -> bool:
        return self._current is not None and (self._current.turn_id == turn_id and self._current.revision == revision)

    def _is_relevant_locked(self, turn_id: str, revision: int) -> bool:
        if self._current is None:
            return True
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

    def _has_pending_reopen_locked(self, turn_id: str, revision: int) -> bool:
        pending = self._pending_reopen
        return pending is not None and pending.turn_id == turn_id and pending.base_revision == revision

    def _grace_matches_locked(self, turn_id: str, revision: int) -> bool:
        grace = self._reopen_grace
        return grace is not None and grace.turn_id == turn_id and grace.revision == revision

    def _reopen_grace_remaining_locked(self, turn_id: str, revision: int) -> float:
        if not self._grace_matches_locked(turn_id, revision):
            return 0.0
        if not self._is_current_locked(turn_id, revision):
            self._reopen_grace = None
            return 0.0
        assert self._reopen_grace is not None
        remaining = self._reopen_grace.deadline - time.monotonic()
        if remaining <= 0:
            self._reopen_grace = None
            return 0.0
        return remaining

    def _wait_for_reopen_gate_locked(self, turn_id: str, revision: int) -> None:
        while self._is_relevant_locked(turn_id, revision):
            self._wait_for_pending_reopen_locked(turn_id, revision, self._PENDING_REOPEN_WAIT_TIMEOUT_S)
            if not self._is_relevant_locked(turn_id, revision):
                return
            remaining = self._reopen_grace_remaining_locked(turn_id, revision)
            if remaining <= 0:
                return
            logger.debug("Waiting for speculative reopen grace turn=%s rev=%s", turn_id, revision)
            self._condition.wait(remaining)

    def _wait_for_pending_reopen_locked(self, turn_id: str, revision: int, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        pending = self._pending_reopen
        if not self._has_pending_reopen_locked(turn_id, revision):
            return
        logger.debug("Waiting for pending speculative reopen turn=%s rev=%s", turn_id, revision)
        while self._pending_reopen == pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning("Timed out waiting for pending speculative reopen turn=%s rev=%s", turn_id, revision)
                if self._pending_reopen == pending:
                    self._pending_reopen = None
                    self._condition.notify_all()
                return
            self._condition.wait(remaining)

    def reset(self) -> None:
        with self._condition:
            self._sequence = 0
            self._current = None
            self._committed.clear()
            self._closed_current = None
            self._pending_reopen = None
            self._reopen_grace = None
            self._condition.notify_all()
