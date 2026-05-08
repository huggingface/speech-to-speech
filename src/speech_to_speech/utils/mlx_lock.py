"""
Global MLX lock to prevent concurrent Metal/MLX model access.

MLX models (STT, LLM, TTS) cannot be used concurrently from multiple threads
on Apple Silicon due to Metal command buffer limitations. This module provides
a global lock that all MLX handlers should acquire before using their models.
"""

import logging
import types
from threading import Lock, RLock, current_thread, get_ident
from time import perf_counter
from typing import Literal

logger = logging.getLogger(__name__)

# Global reentrant lock for MLX model access
# Using RLock (reentrant lock) so the same thread can acquire it multiple times
_mlx_lock = RLock()
_mlx_lock_state = Lock()
_lock_owner_ident: int | None = None
_lock_owner_thread: str | None = None
_lock_owner_handler: str | None = None
_lock_acquired_at: float | None = None
_lock_depth = 0


def _owner_snapshot(now: float | None = None) -> str:
    if _lock_owner_ident is None:
        return "none"
    elapsed = ""
    if _lock_acquired_at is not None:
        elapsed = f" for {(now or perf_counter()) - _lock_acquired_at:.3f}s"
    return f"{_lock_owner_handler} on {_lock_owner_thread}{elapsed}"


def _record_lock_acquired(handler_name: str) -> int:
    global _lock_acquired_at, _lock_depth, _lock_owner_handler, _lock_owner_ident, _lock_owner_thread
    ident = get_ident()
    thread_name = current_thread().name
    now = perf_counter()
    with _mlx_lock_state:
        if _lock_owner_ident == ident:
            _lock_depth += 1
        else:
            _lock_owner_ident = ident
            _lock_owner_thread = thread_name
            _lock_owner_handler = handler_name
            _lock_acquired_at = now
            _lock_depth = 1
        return _lock_depth


def _record_lock_released(handler_name: str) -> tuple[int, float | None]:
    global _lock_acquired_at, _lock_depth, _lock_owner_handler, _lock_owner_ident, _lock_owner_thread
    ident = get_ident()
    now = perf_counter()
    with _mlx_lock_state:
        if _lock_owner_ident != ident:
            logger.warning(
                "%s: MLX lock release requested by non-owner thread; owner=%s",
                handler_name,
                _owner_snapshot(now),
            )
            return _lock_depth, None

        hold_s = None if _lock_acquired_at is None else now - _lock_acquired_at
        _lock_depth -= 1
        depth = _lock_depth
        if _lock_depth <= 0:
            _lock_owner_ident = None
            _lock_owner_thread = None
            _lock_owner_handler = None
            _lock_acquired_at = None
            _lock_depth = 0
        return depth, hold_s


def acquire_mlx_lock(timeout: float | None = None, handler_name: str = "Unknown") -> bool:
    """
    Acquire the global MLX lock.

    Args:
        timeout: Optional timeout in seconds
        handler_name: Name of the handler acquiring the lock (for logging)

    Returns:
        True if lock was acquired, False if timeout occurred
    """
    with _mlx_lock_state:
        owner_before = _owner_snapshot()
    logger.debug("%s: Attempting to acquire MLX lock (owner=%s)", handler_name, owner_before)
    start = perf_counter()
    acquired = _mlx_lock.acquire(timeout=timeout) if timeout else _mlx_lock.acquire(blocking=True)
    wait_s = perf_counter() - start

    if acquired:
        depth = _record_lock_acquired(handler_name)
        if wait_s >= 0.25:
            logger.info(
                "%s: MLX lock acquired after %.2fs (previous_owner=%s, depth=%d)",
                handler_name,
                wait_s,
                owner_before,
                depth,
            )
        else:
            logger.debug(
                "%s: MLX lock acquired after %.3fs (previous_owner=%s, depth=%d)",
                handler_name,
                wait_s,
                owner_before,
                depth,
            )
    else:
        with _mlx_lock_state:
            owner_after = _owner_snapshot()
        logger.warning(
            "%s: Failed to acquire MLX lock after %.3fs (timeout=%s, owner=%s)",
            handler_name,
            wait_s,
            timeout,
            owner_after,
        )

    return acquired


def release_mlx_lock(handler_name: str = "Unknown") -> None:
    """
    Release the global MLX lock.

    Args:
        handler_name: Name of the handler releasing the lock (for logging)
    """
    try:
        depth, hold_s = _record_lock_released(handler_name)
        _mlx_lock.release()
        if hold_s is not None and depth == 0 and hold_s >= 0.25:
            logger.info("%s: MLX lock released after holding %.2fs", handler_name, hold_s)
        else:
            logger.debug(
                "%s: MLX lock released%s",
                handler_name,
                "" if hold_s is None else f" after holding {hold_s:.3f}s (depth={depth})",
            )
    except RuntimeError as e:
        logger.error(f"{handler_name}: Failed to release MLX lock: {e}")


class MLXLockContext:
    """Context manager for MLX lock."""

    def __init__(self, handler_name: str = "Unknown", timeout: float | None = None) -> None:
        self.handler_name = handler_name
        self.timeout = timeout
        self.acquired = False

    def __enter__(self) -> bool:
        self.acquired = acquire_mlx_lock(timeout=self.timeout, handler_name=self.handler_name)
        return self.acquired

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> Literal[False]:
        if self.acquired:
            release_mlx_lock(handler_name=self.handler_name)
        return False  # Do not suppress exceptions
