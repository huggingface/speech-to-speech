"""Leaderboard stored as JSON lines, optionally mirrored to a Hub dataset.

Locally (and on a Space with persistent storage) entries live in
``<DATA_DIR>/leaderboard.jsonl``. When ``LEADERBOARD_REPO`` names a dataset and
``HF_TOKEN`` can write to it, the file is restored from the dataset at startup
and pushed back by a ``CommitScheduler`` every few minutes, so scores survive
Space restarts without paid storage.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

FILENAME = "leaderboard.jsonl"
WEEK_S = 7 * 24 * 3600


class Leaderboard:
    def __init__(self, data_dir: str, repo_id: str = "", token: str = "") -> None:
        self._dir = os.path.join(data_dir, "leaderboard")
        os.makedirs(self._dir, exist_ok=True)
        self._path = os.path.join(self._dir, FILENAME)
        self._lock = threading.Lock()
        self._scheduler = None
        if repo_id and token:
            self._restore(repo_id, token)
            try:
                from huggingface_hub import CommitScheduler

                self._scheduler = CommitScheduler(
                    repo_id=repo_id,
                    repo_type="dataset",
                    folder_path=self._dir,
                    path_in_repo="data",
                    every=5,
                    token=token,
                    private=True,
                )
            except Exception as exc:  # Hub sync is best effort
                logger.warning("leaderboard sync disabled: %s", exc)
        self._entries = self._load()

    def _restore(self, repo_id: str, token: str) -> None:
        if os.path.exists(self._path):
            return
        try:
            from huggingface_hub import hf_hub_download

            src = hf_hub_download(repo_id, f"data/{FILENAME}", repo_type="dataset", token=token)
            with open(src, encoding="utf-8") as fh, open(self._path, "w", encoding="utf-8") as out:
                out.write(fh.read())
            logger.info("restored leaderboard from %s", repo_id)
        except Exception as exc:
            logger.info("no leaderboard to restore from %s (%s)", repo_id, exc)

    def _load(self) -> list[dict[str, Any]]:
        entries = []
        if os.path.exists(self._path):
            with open(self._path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries

    def add(self, entry: dict[str, Any]) -> dict[str, Any]:
        entry = {**entry, "ts": entry.get("ts", time.time())}
        lock = self._scheduler.lock if self._scheduler is not None else self._lock
        with lock:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._entries.append(entry)
        return {
            "global_rank": self._rank(entry, None),
            "topic_rank": self._rank(entry, entry["slug"]),
            "players_on_topic": len({e["player"] for e in self._entries if e["slug"] == entry["slug"]}),
        }

    def _rank(self, entry: dict[str, Any], slug: str | None) -> int:
        best = self._best(slug, None)
        for i, row in enumerate(best, 1):
            if row["player"] == entry["player"]:
                return i
        return len(best) + 1

    def _best(self, slug: str | None, since: float | None) -> list[dict[str, Any]]:
        """Best run per player (per topic when ``slug`` is set), highest first."""
        best: dict[str, dict[str, Any]] = {}
        for e in self._entries:
            if slug and e.get("slug") != slug:
                continue
            if since and e.get("ts", 0) < since:
                continue
            key = e["player"]
            if key not in best or e["score"] > best[key]["score"]:
                best[key] = e
        return sorted(best.values(), key=lambda e: (-e["score"], e.get("ts", 0)))

    def top(self, slug: str | None = None, period: str = "all", limit: int = 10) -> list[dict[str, Any]]:
        since = time.time() - WEEK_S if period == "week" else None
        rows = self._best(slug, since)[:limit]
        keys = (
            "player",
            "verified",
            "score",
            "accuracy",
            "correct",
            "total",
            "best_streak",
            "title",
            "slug",
            "emoji",
            "rank",
            "ts",
        )
        return [{k: r.get(k) for k in keys} for r in rows]

    def topics(self, limit: int = 12) -> list[dict[str, Any]]:
        counts: dict[str, dict[str, Any]] = {}
        for e in self._entries:
            row = counts.setdefault(
                e["slug"], {"slug": e["slug"], "title": e.get("title"), "emoji": e.get("emoji"), "plays": 0}
            )
            row["plays"] += 1
        return sorted(counts.values(), key=lambda r: -r["plays"])[:limit]
