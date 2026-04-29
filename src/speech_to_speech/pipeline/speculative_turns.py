from __future__ import annotations

import logging
from threading import Lock

logger = logging.getLogger(__name__)


class SpeculativeTurnTracker:
    """Thread-safe revision tracker for raw-audio speculative turns."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._latest_revision: dict[str, int] = {}
        self._committed_revision: dict[str, int] = {}

    def observe(self, turn_id: str | None, revision: int | None) -> None:
        if turn_id is None or revision is None:
            return
        with self._lock:
            current = self._latest_revision.get(turn_id, -1)
            if revision > current:
                self._latest_revision[turn_id] = revision
                logger.debug("Observed speculative turn %s revision %d", turn_id, revision)

    def is_latest(self, turn_id: str | None, revision: int | None) -> bool:
        if turn_id is None or revision is None:
            return True
        with self._lock:
            return self._latest_revision.get(turn_id, revision) == revision

    def commit(self, turn_id: str | None, revision: int | None) -> None:
        if turn_id is None or revision is None:
            return
        with self._lock:
            latest = self._latest_revision.get(turn_id, revision)
            if revision == latest:
                self._committed_revision[turn_id] = revision
                logger.debug("Committed speculative turn %s revision %d", turn_id, revision)

    def is_committed(self, turn_id: str | None, revision: int | None = None) -> bool:
        if turn_id is None:
            return False
        with self._lock:
            committed = self._committed_revision.get(turn_id)
            if committed is None:
                return False
            return revision is None or committed >= revision

    def reset(self) -> None:
        with self._lock:
            self._latest_revision.clear()
            self._committed_revision.clear()
