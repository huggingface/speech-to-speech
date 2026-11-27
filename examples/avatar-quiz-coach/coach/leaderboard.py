"""Leaderboard stored as JSON lines in ``<DATA_DIR>/leaderboard/leaderboard.jsonl``.

On the Space, ``DATA_DIR`` is ``/data``: a private Hugging Face storage bucket
mounted read-write into the container, so scores survive restarts and rebuilds.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

FILENAME = "leaderboard.jsonl"
WEEK_S = 7 * 24 * 3600


class Leaderboard:
    def __init__(self, data_dir: str) -> None:
        self._dir = os.path.join(data_dir, "leaderboard")
        os.makedirs(self._dir, exist_ok=True)
        self._path = os.path.join(self._dir, FILENAME)
        self._lock = threading.Lock()
        self._entries = self._load()

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
        with self._lock:
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
