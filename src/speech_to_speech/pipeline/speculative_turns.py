from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Condition

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PendingReopen:
    base_revision: int
    candidate_revision: int


@dataclass(frozen=True)
class _ReopenGrace:
    revision: int
    deadline: float


class SpeculativeTurnTracker:
    """Thread-safe revision and conversation-order tracker for speculative turns."""

    _PENDING_REOPEN_WAIT_TIMEOUT_S = 2.0
    _MAX_TRACKED_TURNS = 2048

    def __init__(self, max_tracked_turns: int = _MAX_TRACKED_TURNS) -> None:
        self._condition = Condition()
        self._max_tracked_turns = max_tracked_turns
        self._latest_revision: OrderedDict[str, int] = OrderedDict()
        self._turn_order: dict[str, int] = {}
        self._latest_order: int | None = None
        self._committed_revision: dict[str, int] = {}
        self._committed_ordered_revisions: set[tuple[str, int]] = set()
        self._pending_reopen: dict[str, _PendingReopen] = {}
        self._reopen_grace: dict[str, _ReopenGrace] = {}

    def observe(self, turn_id: str | None, revision: int | None, *, order: int | None = None) -> None:
        if turn_id is None or revision is None:
            return
        with self._condition:
            existing_order = self._turn_order.get(turn_id)
            if order is not None and existing_order is not None and order != existing_order:
                raise ValueError(f"Conversation order for {turn_id!r} changed from {existing_order} to {order}")

            if order is not None:
                self._turn_order[turn_id] = order
                if self._latest_order is None or order > self._latest_order:
                    self._latest_order = order
                    self._drop_superseded_reopen_state_locked()

            current = self._latest_revision.get(turn_id, -1)
            if revision > current:
                self._latest_revision[turn_id] = revision
                self._latest_revision.move_to_end(turn_id)
                self._prune_tracked_turns()
                logger.debug(
                    "Observed speculative turn %s revision %d order %s",
                    turn_id,
                    revision,
                    self._turn_order.get(turn_id),
                )
                self._condition.notify_all()
            elif order is not None:
                self._condition.notify_all()

    def is_latest(self, turn_id: str | None, revision: int | None) -> bool:
        if turn_id is None or revision is None:
            return True
        with self._condition:
            return self._is_latest_locked(turn_id, revision)

    def is_latest_after_pending_reopen(self, turn_id: str | None, revision: int | None) -> bool:
        if turn_id is None or revision is None:
            return True
        with self._condition:
            self._wait_for_pending_reopen_locked(turn_id, revision, self._PENDING_REOPEN_WAIT_TIMEOUT_S)
            return self._is_latest_locked(turn_id, revision)

    def try_is_latest_after_pending_reopen(self, turn_id: str | None, revision: int | None) -> bool | None:
        """Non-blocking variant of ``is_latest_after_pending_reopen``.

        Returns ``None`` when a matching reopen candidate is still pending and
        the caller should retry after it resolves.
        """
        if turn_id is None or revision is None:
            return True
        with self._condition:
            if self._has_pending_reopen_locked(turn_id, revision):
                return None
            return self._is_latest_locked(turn_id, revision)

    def is_latest_after_reopen_grace(self, turn_id: str | None, revision: int | None) -> bool:
        if turn_id is None or revision is None:
            return True
        with self._condition:
            self._wait_for_reopen_gate_locked(turn_id, revision)
            return self._is_latest_locked(turn_id, revision)

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
            return self._is_latest_locked(turn_id, revision)

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
        """Non-blocking variant of ``commit_if_latest_after_pending_reopen``.

        Returns ``None`` when a matching reopen candidate is still pending and
        the caller should retry after it resolves.
        """
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
            if not self._is_latest_locked(turn_id, revision):
                return
            if self._committed_revision.get(turn_id, -1) >= revision:
                return
            deadline = time.monotonic() + grace_s
            existing = self._reopen_grace.get(turn_id)
            if existing is None or existing.revision != revision or deadline > existing.deadline:
                self._reopen_grace[turn_id] = _ReopenGrace(revision=revision, deadline=deadline)
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
            while self._is_latest_locked(turn_id, revision):
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
            return self._is_latest_locked(turn_id, revision)

    def commit(self, turn_id: str | None, revision: int | None) -> None:
        if turn_id is None or revision is None:
            return
        with self._condition:
            pending = self._pending_reopen.get(turn_id)
            if pending is not None and pending.base_revision == revision:
                logger.debug(
                    "Deferring speculative turn %s revision %d commit while reopen is pending", turn_id, revision
                )
                return
            self._commit_locked(turn_id, revision)

    def is_committed(self, turn_id: str | None, revision: int | None = None) -> bool:
        if turn_id is None:
            return False
        with self._condition:
            committed = self._committed_revision.get(turn_id)
            if committed is None:
                if revision is None:
                    return any(
                        committed_turn_id == turn_id for committed_turn_id, _ in self._committed_ordered_revisions
                    )
                return (turn_id, revision) in self._committed_ordered_revisions
            return revision is None or committed >= revision

    def begin_reopen_candidate(self, turn_id: str | None, revision: int | None) -> int | None:
        if turn_id is None or revision is None:
            return None
        with self._condition:
            if self._committed_revision.get(turn_id, -1) >= revision:
                return None
            if not self._is_latest_locked(turn_id, revision):
                return None

            pending = self._pending_reopen.get(turn_id)
            if pending is not None:
                if pending.base_revision == revision:
                    return pending.candidate_revision
                return None

            candidate_revision = revision + 1
            self._pending_reopen[turn_id] = _PendingReopen(
                base_revision=revision,
                candidate_revision=candidate_revision,
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
            pending = self._pending_reopen.get(turn_id)
            if (
                pending is None
                or pending.base_revision != base_revision
                or pending.candidate_revision != candidate_revision
            ):
                return False
            if self._committed_revision.get(turn_id, -1) >= base_revision:
                del self._pending_reopen[turn_id]
                self._prune_tracked_turns()
                self._condition.notify_all()
                return False
            if not self._is_latest_locked(turn_id, base_revision):
                del self._pending_reopen[turn_id]
                self._prune_tracked_turns()
                self._condition.notify_all()
                return False

            self._latest_revision[turn_id] = candidate_revision
            self._latest_revision.move_to_end(turn_id)
            del self._pending_reopen[turn_id]
            self._prune_tracked_turns()
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
            pending = self._pending_reopen.get(turn_id)
            if pending is None:
                return
            if candidate_revision is not None and pending.candidate_revision != candidate_revision:
                return
            del self._pending_reopen[turn_id]
            self._prune_tracked_turns()
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
        """Record *revision* as committed when it is still the tracked latest.

        Returns whether the caller's output for *revision* is still valid.

        A revision-only turn that is no longer tracked still reports success,
        since dropping work the tracker never ordered would be worse than
        emitting it. Once ordered tracking is active, unknown turns are stale:
        this keeps pruned work from becoming valid again.
        """
        if not self._is_latest_locked(turn_id, revision):
            return False
        latest = self._latest_revision.get(turn_id)
        if latest is None:
            return True
        self._committed_revision[turn_id] = revision
        if turn_id in self._turn_order:
            self._committed_ordered_revisions.add((turn_id, revision))
        logger.debug("Committed speculative turn %s revision %d", turn_id, revision)
        self._condition.notify_all()
        return True

    def _has_pending_reopen_locked(self, turn_id: str, revision: int) -> bool:
        pending = self._pending_reopen.get(turn_id)
        return pending is not None and pending.base_revision == revision

    def _is_latest_locked(self, turn_id: str, revision: int) -> bool:
        latest = self._latest_revision.get(turn_id)
        if latest is not None and latest != revision:
            return False
        if (turn_id, revision) in self._committed_ordered_revisions:
            return True
        if latest is None:
            return self._latest_order is None

        order = self._turn_order.get(turn_id)
        if order is None:
            return self._latest_order is None
        return order == self._latest_order

    def _drop_superseded_reopen_state_locked(self) -> None:
        if self._latest_order is None:
            return
        for turn_id in list(self._pending_reopen):
            order = self._turn_order.get(turn_id)
            if order is not None and order < self._latest_order:
                del self._pending_reopen[turn_id]
        for turn_id in list(self._reopen_grace):
            order = self._turn_order.get(turn_id)
            if order is not None and order < self._latest_order:
                del self._reopen_grace[turn_id]

    def _reopen_grace_remaining_locked(self, turn_id: str, revision: int) -> float:
        grace = self._reopen_grace.get(turn_id)
        if grace is None or grace.revision != revision:
            return 0.0
        if not self._is_latest_locked(turn_id, revision):
            del self._reopen_grace[turn_id]
            return 0.0
        remaining = grace.deadline - time.monotonic()
        if remaining <= 0:
            del self._reopen_grace[turn_id]
            self._prune_tracked_turns()
            return 0.0
        return remaining

    def _wait_for_reopen_gate_locked(self, turn_id: str, revision: int) -> None:
        while self._is_latest_locked(turn_id, revision):
            self._wait_for_pending_reopen_locked(turn_id, revision, self._PENDING_REOPEN_WAIT_TIMEOUT_S)
            if not self._is_latest_locked(turn_id, revision):
                return
            remaining = self._reopen_grace_remaining_locked(turn_id, revision)
            if remaining <= 0:
                return
            logger.debug("Waiting for speculative reopen grace turn=%s rev=%s", turn_id, revision)
            self._condition.wait(remaining)

    def _wait_for_pending_reopen_locked(self, turn_id: str, revision: int, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        pending = self._pending_reopen.get(turn_id)
        if pending is None or pending.base_revision != revision:
            return
        logger.debug("Waiting for pending speculative reopen turn=%s rev=%s", turn_id, revision)
        while pending is not None and pending.base_revision == revision:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning("Timed out waiting for pending speculative reopen turn=%s rev=%s", turn_id, revision)
                if self._pending_reopen.get(turn_id) == pending:
                    del self._pending_reopen[turn_id]
                    self._prune_tracked_turns()
                    self._condition.notify_all()
                return
            self._condition.wait(remaining)
            pending = self._pending_reopen.get(turn_id)

    def _prune_tracked_turns(self) -> None:
        if self._max_tracked_turns <= 0:
            return

        self._drop_expired_reopen_graces_locked()
        current_turn_ids = {turn_id for turn_id, order in self._turn_order.items() if order == self._latest_order}
        prunable_turn_ids = [
            turn_id
            for turn_id in self._latest_revision
            if turn_id not in self._pending_reopen
            and turn_id not in self._reopen_grace
            and turn_id not in current_turn_ids
        ]
        prunable_limit = max(0, self._max_tracked_turns - len(current_turn_ids))
        while len(prunable_turn_ids) > prunable_limit:
            turn_id = prunable_turn_ids.pop(0)
            self._latest_revision.pop(turn_id, None)
            self._turn_order.pop(turn_id, None)
            self._committed_revision.pop(turn_id, None)
            self._reopen_grace.pop(turn_id, None)

    def _drop_expired_reopen_graces_locked(self) -> None:
        now = time.monotonic()
        for turn_id, grace in list(self._reopen_grace.items()):
            if not self._is_latest_locked(turn_id, grace.revision) or grace.deadline <= now:
                del self._reopen_grace[turn_id]

    def reset(self) -> None:
        with self._condition:
            self._latest_revision.clear()
            self._turn_order.clear()
            self._latest_order = None
            self._committed_revision.clear()
            self._committed_ordered_revisions.clear()
            self._pending_reopen.clear()
            self._reopen_grace.clear()
            self._condition.notify_all()
