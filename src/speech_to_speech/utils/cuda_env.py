"""CUDA and cuDNN environment compatibility helpers.

This module provides safe, isolated utilities to prevent host system LD_LIBRARY_PATH
pollution from conflicting with PyTorch's bundled cuDNN wheels on Linux.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

logger = logging.getLogger(__name__)

# Environment variable flag to explicitly disable/enable sanitization
ENV_SANITIZE_CUDA = "S2S_SANITIZE_CUDA_ENV"
ENV_GUARD_FLAG = "_S2S_CLEANED_ENV"

# Path components that indicate host system CUDA/cuDNN installations
CUDA_PATH_PATTERNS = (
    "/usr/local/cuda",
    "/opt/cuda",
    "cuda/lib",
    "cuda/lib64",
    "cudnn/lib",
    "cudnn/lib64",
    "libcudnn",
)


def is_sanitization_enabled() -> bool:
    """Return True if environment sanitization should be evaluated."""
    # Never sanitize on non-Linux platforms (macOS uses Metal, Windows uses PATH)
    if not sys.platform.startswith("linux"):
        return False

    # Never sanitize inside automated test environments
    if "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST"):
        return False

    # Check explicit user configuration (e.g. S2S_SANITIZE_CUDA_ENV=0)
    flag = os.environ.get(ENV_SANITIZE_CUDA, "").strip().lower()
    if flag in {"0", "false", "no", "off", "disable"}:
        return False

    return True


def has_conflicting_cuda_paths(ld_path: str) -> bool:
    """Check if LD_LIBRARY_PATH contains system CUDA/cuDNN directories."""
    if not ld_path:
        return False
    lower_path = ld_path.lower()
    return any(pattern.lower() in lower_path for pattern in CUDA_PATH_PATTERNS)


def clean_ld_library_path(ld_path: str) -> str:
    """Strip system CUDA/cuDNN paths from LD_LIBRARY_PATH, preserving other user paths."""
    if not ld_path:
        return ""
    entries = [entry for entry in ld_path.split(":") if entry.strip()]
    cleaned_entries = [
        entry for entry in entries
        if not any(pattern.lower() in entry.lower() for pattern in CUDA_PATH_PATTERNS)
    ]
    return ":".join(cleaned_entries)


def sanitize_cuda_environment(
    executable: Optional[str] = None,
    argv: Optional[list[str]] = None,
) -> bool:
    """Sanitize LD_LIBRARY_PATH on Linux if conflicting system CUDA/cuDNN is detected.

    Returns True if re-exec occurred (or was simulated), False otherwise.
    """
    if not is_sanitization_enabled():
        return False

    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if not has_conflicting_cuda_paths(ld_path):
        return False

    if os.environ.get(ENV_GUARD_FLAG):
        return False

    new_env = dict(os.environ)
    new_env[ENV_GUARD_FLAG] = "1"
    cleaned = clean_ld_library_path(ld_path)
    if cleaned:
        new_env["LD_LIBRARY_PATH"] = cleaned
    else:
        new_env.pop("LD_LIBRARY_PATH", None)

    exec_path = executable or sys.executable
    exec_args = argv or ([exec_path] + sys.argv)

    try:
        os.execve(exec_path, exec_args, new_env)
        return True
    except Exception as e:
        logger.warning(f"Could not re-exec with sanitized LD_LIBRARY_PATH: {e}")
        return False
