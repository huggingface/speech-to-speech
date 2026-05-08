"""
Global MLX lock to prevent concurrent Metal/MLX model access.

MLX models (STT, LLM, TTS) cannot be used concurrently from multiple threads
on Apple Silicon due to Metal command buffer limitations. This module provides
a global lock that all MLX handlers should acquire before using their models.
"""

import logging
import types
from threading import RLock
from time import perf_counter
from typing import Literal

logger = logging.getLogger(__name__)

# Global reentrant lock for MLX model access
# Using RLock (reentrant lock) so the same thread can acquire it multiple times
_mlx_lock = RLock()


def acquire_mlx_lock(timeout: float | None = None, handler_name: str = "Unknown") -> bool:
    """
    Acquire the global MLX lock.

    Args:
        timeout: Optional timeout in seconds
        handler_name: Name of the handler acquiring the lock (for logging)

    Returns:
        True if lock was acquired, False if timeout occurred
    """
    logger.debug(f"{handler_name}: Attempting to acquire MLX lock")
    start = perf_counter()
    acquired = _mlx_lock.acquire(timeout=timeout) if timeout else _mlx_lock.acquire(blocking=True)
    wait_s = perf_counter() - start

    if acquired:
        if wait_s >= 0.25:
            logger.info("%s: MLX lock acquired after %.2fs", handler_name, wait_s)
        else:
            logger.debug("%s: MLX lock acquired after %.3fs", handler_name, wait_s)
    else:
        logger.warning(f"{handler_name}: Failed to acquire MLX lock (timeout)")

    return acquired


def release_mlx_lock(handler_name: str = "Unknown") -> None:
    """
    Release the global MLX lock.

    Args:
        handler_name: Name of the handler releasing the lock (for logging)
    """
    try:
        _mlx_lock.release()
        logger.debug(f"{handler_name}: MLX lock released")
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
