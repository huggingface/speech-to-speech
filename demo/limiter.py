"""
Per-day talk-time budget for the speech-to-speech demo.

Our server isn't in the audio path (the browser dials the compute WebSocket
directly), so it can't cut a live stream. What it *can* do is meter time with a
server-clock, chunked reservation:

  - At grant we reserve the first chunk (CHUNK_SEC) and debit it from the day's
    budget. A parallel grant therefore sees the budget already spent.
  - The client heartbeats; each heartbeat extends the reservation one chunk at a
    time until the daily budget runs out, then we report `expired` so the client
    tears down.
  - On a clean end (sendBeacon) we reconcile to the real elapsed time and refund
    the unused chunk. A crash (no end, no heartbeats) is reaped by a sweep and
    forfeits at most one chunk.

All time is the server's clock. Budgets are per UTC day; a new day is simply a
new row (no explicit reset). Logged-in users are keyed by a hashed HF `sub`;
anonymous users by BOTH a hashed IP and a hashed signed-cookie id, OR-matched
(spent = max of the two) so clearing one identifier doesn't reset the budget.

Storage is SQLite at $USAGE_DB_PATH, else /data (persistent Spaces storage),
else a /tmp fallback. On /tmp the budget is only per-uptime — flagged in logs.
"""

import hashlib
import hmac
import logging
import math
import os
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("s2s.limiter")

# ── Tunables (env-overridable) ───────────────────────────────────────────────
ANON_SEC = int(os.environ.get("LIMIT_ANON_SEC", "300"))   # 5 min/day, not signed in
FREE_SEC = int(os.environ.get("LIMIT_FREE_SEC", "600"))   # 10 min/day, signed in, no PRO
CHUNK_SEC = int(os.environ.get("RESERVE_CHUNK_SEC", "10"))  # reservation granularity
HEARTBEAT_SEC = int(os.environ.get("HEARTBEAT_SEC", "5"))   # advertised client cadence
REAP_AFTER_SEC = int(os.environ.get("SESSION_REAP_SEC", "15"))  # silence before sweep

# Stable across restarts or the hashed keys (and signed cookies) rotate and the
# budget effectively resets. Set it as a Space secret. Falls back to a per-boot
# random value (keys then only hold within one uptime).
_HASH_SECRET = (os.environ.get("USAGE_HASH_SECRET", "").strip() or os.urandom(32).hex()).encode()

_lock = threading.Lock()
_db_path: "Path | None" = None


def budget_for(tier: str) -> "int | None":
    """Daily second-budget for a tier, or None for unlimited.

    Unlimited tiers: 'pro' (paying PRO members) and 'org' (members of an
    allow-listed organisation, see UNLIMITED_ORGS in auth.py)."""
    if tier in ("pro", "org"):
        return None
    if tier == "free":
        return FREE_SEC
    return ANON_SEC


def hash_key(raw: str) -> str:
    """HMAC a raw identifier (sub / ip / cookie id) into an opaque storage key."""
    digest = hmac.new(_HASH_SECRET, raw.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"k_{digest}"


def sign_cookie(value: str) -> str:
    """`<id>.<sig>` so a forged anon-cookie id is rejected on read."""
    sig = hmac.new(_HASH_SECRET, value.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
    return f"{value}.{sig}"


def verify_cookie(signed: str) -> "str | None":
    """Return the id if the signature checks out, else None."""
    if not signed or "." not in signed:
        return None
    value, _, sig = signed.rpartition(".")
    want = hmac.new(_HASH_SECRET, value.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
    return value if hmac.compare_digest(sig, want) else None


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _resolve_db_path() -> Path:
    explicit = os.environ.get("USAGE_DB_PATH", "").strip()
    if explicit:
        return Path(explicit)
    data = Path("/data")
    if data.is_dir() and os.access(data, os.W_OK):
        return data / "s2s-usage.sqlite3"
    logger.warning("No persistent /data — usage budget falls back to /tmp (per-uptime only).")
    return Path(tempfile.gettempdir()) / "s2s-usage.sqlite3"


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(_db_path, timeout=5.0)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=5000")
    return con


def init() -> None:
    """Create the schema. Call once at startup."""
    global _db_path
    _db_path = _resolve_db_path()
    with _lock, _connect() as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS usage_daily (
                   user_key   TEXT NOT NULL,
                   day        TEXT NOT NULL,
                   spent_sec  INTEGER NOT NULL DEFAULT 0,
                   updated_at INTEGER NOT NULL,
                   PRIMARY KEY (user_key, day)
               )"""
        )
        con.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                   session_id   TEXT PRIMARY KEY,
                   keys         TEXT NOT NULL,   -- comma-joined usage_daily keys to debit
                   day          TEXT NOT NULL,
                   tier         TEXT NOT NULL,
                   grant_ts     REAL NOT NULL,
                   last_seen_ts REAL NOT NULL,
                   reserved_sec INTEGER NOT NULL,
                   ended        INTEGER NOT NULL DEFAULT 0
               )"""
        )
    logger.info("Usage limiter ready at %s (anon=%ss free=%ss chunk=%ss)", _db_path, ANON_SEC, FREE_SEC, CHUNK_SEC)


# ── Internal helpers (call under _lock) ───────────────────────────────────────

def _spent(con, key: str, day: str) -> int:
    row = con.execute(
        "SELECT spent_sec FROM usage_daily WHERE user_key=? AND day=?", (key, day)
    ).fetchone()
    return int(row[0]) if row else 0


def _spent_max(con, keys, day: str) -> int:
    """OR-match: the most-spent identifier governs."""
    return max((_spent(con, k, day) for k in keys), default=0)


def _add(con, keys, day: str, delta: int) -> None:
    now = int(time.time())
    for k in keys:
        cur = _spent(con, k, day)
        nxt = max(0, cur + delta)
        con.execute(
            """INSERT INTO usage_daily (user_key, day, spent_sec, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(user_key, day) DO UPDATE SET
                   spent_sec = excluded.spent_sec, updated_at = excluded.updated_at""",
            (k, day, nxt, now),
        )


# ── Public API ────────────────────────────────────────────────────────────────

def remaining(keys, tier: str) -> "int | None":
    """Seconds left today for these keys (None = unlimited). No mutation."""
    budget = budget_for(tier)
    if budget is None:
        return None
    with _lock, _connect() as con:
        return max(0, budget - _spent_max(con, keys, _today()))


def begin(session_id: str, keys, tier: str) -> int:
    """Reserve the first chunk for a new session and record it. Returns the
    chunk reserved (0 if the budget is already exhausted — the first heartbeat
    will then expire it). PRO (unlimited) is never tracked; don't call it here."""
    day = _today()
    budget = budget_for(tier)
    now = time.time()
    with _lock, _connect() as con:
        avail = budget - _spent_max(con, keys, day) if budget is not None else CHUNK_SEC
        chunk = max(0, min(CHUNK_SEC, avail))
        if chunk:
            _add(con, keys, day, chunk)
        con.execute(
            """INSERT OR REPLACE INTO sessions
               (session_id, keys, day, tier, grant_ts, last_seen_ts, reserved_sec, ended)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
            (session_id, ",".join(keys), day, tier, now, now, chunk),
        )
        return chunk


def heartbeat(session_id: str) -> bool:
    """Keep a session alive: extend the reservation toward `elapsed + 1 chunk`,
    debiting the budget chunk by chunk. Returns True while alive, False once the
    budget is spent (caller should tear down) or the session is unknown/ended."""
    now = time.time()
    with _lock, _connect() as con:
        row = con.execute(
            "SELECT keys, day, tier, grant_ts, reserved_sec, ended FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if not row or row[5]:
            return False
        keys = row[0].split(",")
        day, tier, grant_ts, reserved = row[1], row[2], row[3], int(row[4])
        budget = budget_for(tier)
        elapsed = now - grant_ts

        # Grow the reservation one chunk at a time until it covers elapsed + a
        # one-chunk lookahead, or the budget runs dry.
        while reserved < elapsed + CHUNK_SEC:
            if budget is not None and _spent_max(con, keys, day) >= budget:
                break
            _add(con, keys, day, CHUNK_SEC)
            reserved += CHUNK_SEC

        alive = reserved > elapsed  # could we cover the time already elapsed?
        con.execute(
            "UPDATE sessions SET last_seen_ts=?, reserved_sec=?, ended=? WHERE session_id=?",
            (now, reserved, 0 if alive else 1, session_id),
        )
        if not alive:
            _reconcile(con, keys, day, grant_ts, reserved, end_ts=now)
        return alive


def end(session_id: str) -> None:
    """Clean teardown: reconcile to actual elapsed time and refund the unused
    reservation. Idempotent."""
    now = time.time()
    with _lock, _connect() as con:
        row = con.execute(
            "SELECT keys, day, grant_ts, reserved_sec, ended FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if not row or row[4]:
            return
        keys, day, grant_ts, reserved = row[0].split(","), row[1], row[2], int(row[3])
        _reconcile(con, keys, day, grant_ts, reserved, end_ts=now)
        con.execute("UPDATE sessions SET ended=1, last_seen_ts=? WHERE session_id=?", (now, session_id))


def sweep() -> None:
    """Reap sessions that went silent (crash / closed without a beacon): bill
    their elapsed time, refund the rest, mark ended. Forfeits ≤ one chunk."""
    now = time.time()
    cutoff = now - REAP_AFTER_SEC
    with _lock, _connect() as con:
        stale = con.execute(
            "SELECT session_id, keys, day, grant_ts, last_seen_ts, reserved_sec FROM sessions "
            "WHERE ended=0 AND last_seen_ts < ?",
            (cutoff,),
        ).fetchall()
        for session_id, keys_s, day, grant_ts, last_seen, reserved in stale:
            _reconcile(con, keys_s.split(","), day, grant_ts, int(reserved), end_ts=last_seen)
            con.execute("UPDATE sessions SET ended=1 WHERE session_id=?", (session_id,))
        if stale:
            logger.debug("swept %d stale session(s)", len(stale))


def _reconcile(con, keys, day: str, grant_ts: float, reserved: int, end_ts: float) -> None:
    """Refund reserved-but-unused time. Bill elapsed rounded up to a chunk,
    capped at what was reserved."""
    elapsed = max(0.0, end_ts - grant_ts)
    billed = min(reserved, int(math.ceil(elapsed / CHUNK_SEC) * CHUNK_SEC))
    refund = reserved - billed
    if refund > 0:
        _add(con, keys, day, -refund)
