"""Discovery of locally cached language models.

The realtime server picks its LLM at startup, but clients want to offer a
picker over what is actually on disk. Scanning the Hugging Face cache is the
only source of truth for that: a repo is usable iff its weights are already
downloaded, and the same cache also holds TTS/STT/VAD models that must not
show up in an LLM picker.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# A causal-LM checkpoint declares an architecture ending in one of these. This
# is what separates a chat model from the Kokoro / Parakeet / Qwen3-TTS /
# smart-turn checkpoints sitting in the same cache.
_CAUSAL_LM_SUFFIXES = ("ForCausalLM", "ForConditionalGeneration")

# Repos that satisfy the architecture check but are not conversational models.
_EXCLUDED_MODEL_TYPES = frozenset({"qwen3_tts", "kokoro", "parakeet", "whisper"})


def _snapshot_config(repo_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    """Newest snapshot of ``repo_dir`` that has a readable ``config.json``."""
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    for snapshot in sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        config_path = snapshot / "config.json"
        if not config_path.is_file():
            continue
        try:
            with config_path.open() as fh:
                config = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(config, dict):
            return snapshot, config
    return None


def _is_causal_lm(config: dict[str, Any]) -> bool:
    if str(config.get("model_type", "")).lower() in _EXCLUDED_MODEL_TYPES:
        return False
    architectures = config.get("architectures")
    if not isinstance(architectures, list):
        return False
    return any(isinstance(a, str) and a.endswith(_CAUSAL_LM_SUFFIXES) for a in architectures)


def _has_weights(snapshot: Path) -> bool:
    """A cache entry can exist with metadata but no weights (interrupted pull)."""
    return any(snapshot.glob("*.safetensors")) or any(snapshot.glob("*.bin"))


def cache_root() -> Path:
    """The Hugging Face hub cache directory, honouring the usual env overrides."""
    import os

    hf_hub = os.environ.get("HF_HUB_CACHE")
    if hf_hub:
        return Path(hf_hub)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def list_local_language_models() -> list[str]:
    """Repo ids of cached causal-LM checkpoints, e.g. ``["Qwen/Qwen3-0.6B"]``.

    Returns them sorted. Never raises: an unreadable cache yields an empty
    list, which callers treat as "no picker".
    """
    root = cache_root()
    if not root.is_dir():
        return []

    found: list[str] = []
    for repo_dir in root.glob("models--*"):
        if not repo_dir.is_dir():
            continue
        result = _snapshot_config(repo_dir)
        if result is None:
            continue
        snapshot, config = result
        if not _is_causal_lm(config) or not _has_weights(snapshot):
            continue
        # models--Qwen--Qwen3-0.6B -> Qwen/Qwen3-0.6B
        found.append(repo_dir.name[len("models--"):].replace("--", "/"))
    return sorted(found)
