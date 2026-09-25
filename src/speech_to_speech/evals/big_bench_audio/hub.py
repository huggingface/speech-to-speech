"""Publish run reports to a Hugging Face dataset repo.

A job container is ephemeral, so a report that only reaches its filesystem is
gone when the job ends. Pushing to a dataset repo gives runs a durable, browsable
history that ``compare`` can be pointed at later.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_PREFIX = "reports"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(text: str) -> str:
    """Reduce a run label to something safe for a repo path."""
    slug = _UNSAFE.sub("-", text).strip("-")
    return slug[:80] or "run"


def report_filename(report: dict[str, Any], *, prefix: str = DEFAULT_PREFIX) -> str:
    """Path inside the dataset repo for *report*.

    Sorts chronologically by name, and keeps the run label visible so a listing
    is readable without opening the files.
    """
    stamp = report.get("started_at") or datetime.now(timezone.utc).isoformat()
    try:
        moment = datetime.fromisoformat(stamp).astimezone(timezone.utc)
    except ValueError:
        moment = datetime.now(timezone.utc)
    label = slugify(str(report.get("label") or "run"))
    return f"{prefix}/{moment:%Y%m%dT%H%M%S}-{label}.json"


def push_report(
    report: dict[str, Any],
    repo_id: str,
    *,
    path_in_repo: Optional[str] = None,
    token: Optional[str] = None,
    api: Any = None,
) -> str:
    """Upload *report* to a dataset repo, creating it if needed. Returns the file URL."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)

    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=True)
    destination = path_in_repo or report_filename(report)
    api.upload_file(
        path_or_fileobj=json.dumps(report, indent=2).encode(),
        path_in_repo=destination,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"vibe check: {report.get('label') or 'run'}",
    )
    return f"https://huggingface.co/datasets/{repo_id}/blob/main/{destination}"


def load_report(source: str, *, token: Optional[str] = None) -> dict[str, Any]:
    """Read a report from a local path, or from ``<repo_id>:<path>`` on the Hub."""
    local = Path(source)
    if local.is_file():
        return json.loads(local.read_text())

    repo_id, separator, path_in_repo = source.partition(":")
    if not separator or not path_in_repo:
        raise FileNotFoundError(f"No report at {source!r} (expected a path, or '<repo_id>:<path-in-repo>').")

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(repo_id, path_in_repo, repo_type="dataset", token=token)
    return json.loads(Path(downloaded).read_text())
