"""
Global MLX lock to prevent concurrent Metal/MLX model access.

MLX models (STT, LLM, TTS) cannot be used concurrently from multiple threads
on Apple Silicon due to Metal command buffer limitations. This module provides
a global lock that all MLX handlers should acquire before using their models.
"""

import logging
from threading import RLock

logger = logging.getLogger(__name__)

# Global reentrant lock for MLX model access
# Using RLock (reentrant lock) so the same thread can acquire it multiple times
_mlx_lock = RLock()


def acquire_mlx_lock(timeout=None, handler_name="Unknown"):
    """
    Acquire the global MLX lock.

    Args:
        timeout: Optional timeout in seconds
        handler_name: Name of the handler acquiring the lock (for logging)

    Returns:
        True if lock was acquired, False if timeout occurred
    """
    logger.debug(f"{handler_name}: Attempting to acquire MLX lock")
    acquired = _mlx_lock.acquire(timeout=timeout) if timeout else _mlx_lock.acquire(blocking=True)

    if acquired:
        logger.debug(f"{handler_name}: MLX lock acquired")
    else:
        logger.warning(f"{handler_name}: Failed to acquire MLX lock (timeout)")

    return acquired


def release_mlx_lock(handler_name="Unknown"):
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

    def __init__(self, handler_name="Unknown", timeout=None):
        self.handler_name = handler_name
        self.timeout = timeout
        self.acquired = False

    def __enter__(self):
        self.acquired = acquire_mlx_lock(timeout=self.timeout, handler_name=self.handler_name)
        return self.acquired

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.acquired:
            release_mlx_lock(handler_name=self.handler_name)
        return False  # Don't suppress exceptions
