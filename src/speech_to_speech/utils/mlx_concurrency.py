"""Fair process-wide concurrency limit for MLX inference.

MLX 0.32 supports independent computations from multiple threads, so MLX handlers no
longer need to serialize behind a global lock. Sustained three-way STT/LLM/TTS inference
can still overwhelm the Metal driver or expose downstream model races, however. This gate
allows two MLX operations to overlap while queuing additional work in arrival order.
"""

from __future__ import annotations

import logging
import types
from collections import deque
from threading import Condition, get_ident
from time import perf_counter
from typing import Literal

logger = logging.getLogger(__name__)

MAX_CONCURRENT_MLX_OPERATIONS = 2


class _FairConcurrencyGate:
    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("MLX concurrency limit must be at least 1")
        self.limit = limit
        self._available = limit
        self._condition = Condition()
        self._waiters: deque[object] = deque()
        self._depth_by_thread: dict[int, int] = {}

    def acquire(self, timeout: float | None = None) -> bool:
        ident = get_ident()
        with self._condition:
            depth = self._depth_by_thread.get(ident)
            if depth is not None:
                self._depth_by_thread[ident] = depth + 1
                return True

            waiter = object()
            self._waiters.append(waiter)
            deadline = None if timeout is None else perf_counter() + timeout

            try:
                while self._waiters[0] is not waiter or self._available == 0:
                    remaining = None if deadline is None else deadline - perf_counter()
                    if remaining is not None and remaining <= 0:
                        self._waiters.remove(waiter)
                        self._condition.notify_all()
                        return False
                    self._condition.wait(timeout=remaining)
            except BaseException:
                self._waiters.remove(waiter)
                self._condition.notify_all()
                raise

            self._waiters.popleft()
            self._available -= 1
            self._depth_by_thread[ident] = 1
            self._condition.notify_all()
            return True

    def release(self) -> None:
        ident = get_ident()
        with self._condition:
            depth = self._depth_by_thread.get(ident)
            if depth is None:
                raise RuntimeError("MLX concurrency gate released by a non-owner thread")
            if depth > 1:
                self._depth_by_thread[ident] = depth - 1
                return

            del self._depth_by_thread[ident]
            self._available += 1
            self._condition.notify_all()


_mlx_concurrency_gate = _FairConcurrencyGate(MAX_CONCURRENT_MLX_OPERATIONS)


class MLXConcurrencyContext:
    """Reserve one of the process-wide MLX inference slots."""

    def __init__(self, handler_name: str = "Unknown", timeout: float | None = None) -> None:
        self.handler_name = handler_name
        self.timeout = timeout
        self.acquired = False
        self._acquired_at: float | None = None

    def __enter__(self) -> bool:
        start = perf_counter()
        self.acquired = _mlx_concurrency_gate.acquire(timeout=self.timeout)
        wait_s = perf_counter() - start
        if self.acquired:
            self._acquired_at = perf_counter()
            if wait_s >= 0.25:
                logger.info("%s: MLX concurrency slot acquired after %.2fs", self.handler_name, wait_s)
        else:
            log_timeout = logger.debug if self.timeout is not None and self.timeout < 0.25 else logger.warning
            log_timeout(
                "%s: Failed to acquire MLX concurrency slot after %.3fs (timeout=%s)",
                self.handler_name,
                wait_s,
                self.timeout,
            )
        return self.acquired

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> Literal[False]:
        if self.acquired:
            _mlx_concurrency_gate.release()
            if self._acquired_at is not None:
                hold_s = perf_counter() - self._acquired_at
                if hold_s >= 0.25:
                    logger.debug("%s: MLX concurrency slot released after %.2fs", self.handler_name, hold_s)
        return False
