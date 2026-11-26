"""Viva: a voice quiz coach with a lip-synced 3D avatar.

The browser talks to a speech-to-speech backend over the OpenAI Realtime
WebSocket protocol for listening and speaking. This server does everything
else: it researches topics into quiz packs, grades each spoken answer, keeps
score and serves the leaderboard.

Speech backend (first match wins):
  SPEECH_TO_SPEECH_URL  the browser dials this realtime WebSocket URL directly.
  LOAD_BALANCER_URL     a speech-to-speech load balancer; we proxy its /session
                        and /queue handshake so its address stays private.
  SESSION_PROXY_URL     another deployment's session API to piggyback on, e.g.
                        https://smolagents-hf-realtime-voice.hf.space/api
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import httpx
from coach import game as game_mod
from coach import llm, research
from coach.leaderboard import Leaderboard
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("viva")

ROOT = Path(__file__).parent
STATIC = ROOT / "static"
NODE = ROOT / "node_modules"
DATA_DIR = Path(os.environ.get("DATA_DIR") or ("/data" if os.path.isdir("/data") else ROOT / "data"))
PACKS_DIR = ROOT / "packs"

SPEECH_TO_SPEECH_URL = os.environ.get("SPEECH_TO_SPEECH_URL", "").strip()
LOAD_BALANCER_URL = "" if SPEECH_TO_SPEECH_URL else os.environ.get("LOAD_BALANCER_URL", "").strip().rstrip("/")
SESSION_PROXY_URL = (
    "" if SPEECH_TO_SPEECH_URL or LOAD_BALANCER_URL else os.environ.get("SESSION_PROXY_URL", "").strip().rstrip("/")
)
LB_HF_TOKEN = os.environ.get("LB_HF_TOKEN", "").strip()
# Local development against a load balancer that requires a verified user
# token: stands in for the signed-in visitor's OAuth token. Never set it on a
# public deployment, or every visitor is billed to this account.
LB_USER_TOKEN = os.environ.get("LB_USER_TOKEN", "").strip()
RESEARCH_PER_HOUR = int(os.environ.get("RESEARCH_PER_HOUR", "8"))
# Empty: each coach uses its own voice (Max: Aiden, Ada: Serena; see static/js/avatar.js).
VOICE = os.environ.get("COACH_VOICE", "").strip()

app = FastAPI(title="Viva quiz coach")

# ── Optional Hugging Face sign-in (Spaces with `hf_oauth: true`) ─────────────

OAUTH = False
if os.environ.get("OAUTH_CLIENT_ID"):
    try:
        from huggingface_hub import attach_huggingface_oauth, parse_huggingface_oauth

        attach_huggingface_oauth(app)
        OAUTH = True
    except Exception as exc:  # pragma: no cover - depends on the Space runtime
        logger.warning("HF OAuth unavailable: %s", exc)


def _oauth(request: Request) -> Any:
    if not OAUTH:
        return None
    try:
        return parse_huggingface_oauth(request)
    except Exception:
        return None


def _username(request: Request) -> str | None:
    info = _oauth(request)
    user = getattr(info, "user_info", None)
    name = getattr(user, "preferred_username", None) if user else None
    return str(name) if name else None


def _access_token(request: Request) -> str | None:
    token = getattr(_oauth(request), "access_token", None)
    return str(token) if token else None


# ── State ───────────────────────────────────────────────────────────────────

FEATURED = research.load_packs(str(PACKS_DIR))
PACKS: dict[str, dict[str, Any]] = dict(FEATURED)
GENERATED_DIR = DATA_DIR / "packs"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
PACKS.update(research.load_packs(str(GENERATED_DIR)))
GAMES = game_mod.GameStore()
BOARD = Leaderboard(
    str(DATA_DIR), os.environ.get("LEADERBOARD_REPO", "").strip(), os.environ.get("HF_TOKEN", "").strip()
)
_research_log: dict[str, deque[float]] = defaultdict(deque)


def _client_key(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "anon")


def _allow_research(request: Request) -> bool:
    log = _research_log[_username(request) or _client_key(request)]
    now = time.time()
    while log and log[0] < now - 3600:
        log.popleft()
    if len(log) >= RESEARCH_PER_HOUR:
        return False
    log.append(now)
    return True


_NAME_RE = re.compile(r"[^\w .\-]", re.UNICODE)


def _player(request: Request, requested: str) -> tuple[str, bool]:
    username = _username(request)
    if username:
        return username, True
    name = " ".join(_NAME_RE.sub("", requested or "").split())[:24]
    return name or "Anonymous", False


# ── Pages and static assets ─────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")
# Browser libraries straight from node_modules (no bundler): three.js,
# TalkingHead and HeadAudio, resolved through the page's import map.
app.mount("/vendor/three", StaticFiles(directory=NODE / "three"), name="three")
app.mount("/vendor/talkinghead", StaticFiles(directory=NODE / "@met4citizen/talkinghead/modules"), name="talkinghead")
app.mount("/vendor/headaudio", StaticFiles(directory=NODE / "@met4citizen/headaudio/dist"), name="headaudio")


@app.get("/api/config")
def config(request: Request) -> dict[str, Any]:
    mode = "direct" if SPEECH_TO_SPEECH_URL else "lb" if (LOAD_BALANCER_URL or SESSION_PROXY_URL) else "manual"
    return {
        "mode": mode,
        "wsUrl": SPEECH_TO_SPEECH_URL,
        "voice": VOICE,
        "oauth": OAUTH,
        "user": _username(request),
        "llmReady": bool(llm.api_token()),
    }


# ── Speech backend session handshake (load balancer / piggyback) ────────────


def _upstream() -> tuple[str, dict[str, str]]:
    if LOAD_BALANCER_URL:
        headers = {"Content-Type": "application/json"}
        return LOAD_BALANCER_URL, headers
    if SESSION_PROXY_URL:
        return SESSION_PROXY_URL, {"Content-Type": "application/json"}
    raise HTTPException(status_code=404, detail="No speech backend configured.")


def _upstream_headers(request: Request, base: dict[str, str]) -> dict[str, str]:
    headers = dict(base)
    user_token = _access_token(request) or LB_USER_TOKEN or None
    if LOAD_BALANCER_URL:
        gateway = LB_HF_TOKEN or user_token
        if gateway:
            headers["Authorization"] = f"Bearer {gateway}"
        if user_token:
            headers["X-Reachy-Mini-Authorization"] = f"Bearer {user_token}"
    elif request.headers.get("cookie"):
        # Piggyback mode: the upstream Space meters visitors by cookie, so each
        # browser keeps its own identity there.
        headers["Cookie"] = request.headers["cookie"]
    return headers


def _relay(resp: httpx.Response) -> JSONResponse:
    try:
        body = resp.json()
    except ValueError:
        body = {"error": resp.text[:200]}
    out = JSONResponse(body, status_code=resp.status_code)
    if SESSION_PROXY_URL:
        for raw in resp.headers.get_list("set-cookie"):
            parts = [p.strip() for p in raw.split(";") if not p.strip().lower().startswith("domain=")]
            out.headers.append("set-cookie", "; ".join(parts))
    return out


@app.post("/api/session")
async def session(request: Request) -> JSONResponse:
    base, headers = _upstream()
    try:
        async with httpx.AsyncClient(timeout=20.0) as http:
            resp = await http.post(f"{base}/session", headers=_upstream_headers(request, headers), content="{}")
    except httpx.HTTPError as exc:
        logger.warning("speech backend unreachable: %r", exc)
        raise HTTPException(status_code=502, detail="Speech service unreachable.") from exc
    return _relay(resp)


@app.get("/api/queue/{queue_id}")
async def queue_status(queue_id: str, request: Request) -> JSONResponse:
    base, headers = _upstream()
    async with httpx.AsyncClient(timeout=20.0) as http:
        resp = await http.get(f"{base}/queue/{queue_id}", headers=_upstream_headers(request, headers))
    return _relay(resp)


@app.delete("/api/queue/{queue_id}")
async def queue_leave(queue_id: str, request: Request) -> JSONResponse:
    base, headers = _upstream()
    async with httpx.AsyncClient(timeout=20.0) as http:
        resp = await http.delete(f"{base}/queue/{queue_id}", headers=_upstream_headers(request, headers))
    return _relay(resp)


# ── Topics and research ─────────────────────────────────────────────────────


def _pack_card(pack: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": pack["id"],
        "title": pack["title"],
        "emoji": pack.get("emoji", "🎓"),
        "topic": pack["topic"],
        "slug": pack["slug"],
        "difficulty": pack["difficulty"],
        "count": len(pack["questions"]),
        "tagline": pack.get("tagline", ""),
        "featured": pack["id"] in FEATURED,
    }


@app.get("/api/topics")
def topics() -> dict[str, Any]:
    featured = sorted(FEATURED.values(), key=lambda p: p.get("order", 99))
    return {"featured": [_pack_card(p) for p in featured], "popular": BOARD.topics()}


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=160)
    difficulty: str = "pro"
    count: int = Field(default=8, ge=3, le=12)


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/api/research")
async def research_topic(req: ResearchRequest, request: Request) -> StreamingResponse:
    topic = " ".join(req.topic.split())
    difficulty = req.difficulty if req.difficulty in research.DIFFICULTIES else "pro"
    pack_id = research.pack_id_for(topic, difficulty, req.count)

    async def stream():
        cached = PACKS.get(pack_id)
        if cached:
            yield _sse({"type": "status", "stage": "write", "message": "I've studied this one before"})
            yield _sse({"type": "pack", "pack": _client_pack(cached)})
            return
        if not llm.api_token():
            yield _sse(
                {"type": "error", "message": "This deployment has no HF_TOKEN, so it can't research new topics."}
            )
            return
        if not _allow_research(request):
            yield _sse(
                {"type": "error", "message": "You've researched a lot of topics this hour. Try a featured topic!"}
            )
            return
        try:
            async for event in research.build_pack(topic, difficulty, req.count):
                if event["type"] == "pack":
                    pack = event["pack"]
                    PACKS[pack["id"]] = pack
                    (GENERATED_DIR / f"{pack['id']}.json").write_text(json.dumps(pack, ensure_ascii=False, indent=1))
                    event = {"type": "pack", "pack": _client_pack(pack)}
                yield _sse(event)
        except Exception as exc:
            logger.exception("research failed for %r", topic)
            yield _sse({"type": "error", "message": f"Research failed: {exc}"[:240]})

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def _client_pack(pack: dict[str, Any]) -> dict[str, Any]:
    """Pack metadata without answers (answers stay server-side until revealed)."""
    return {
        **_pack_card(pack),
        "brief": pack.get("brief", ""),
        "sources": pack.get("sources", [])[:8],
    }


# ── Games ───────────────────────────────────────────────────────────────────


class StartRequest(BaseModel):
    pack_id: str
    player: str = ""
    coach: str = Field(default="Max", max_length=20)
    question_ids: list[int] | None = None


@app.post("/api/game")
def start_game(req: StartRequest, request: Request) -> dict[str, Any]:
    pack = PACKS.get(req.pack_id)
    if pack is None:
        raise HTTPException(status_code=404, detail="Unknown topic pack.")
    ids = None
    if req.question_ids:
        valid = {q["id"] for q in pack["questions"]}
        ids = [i for i in dict.fromkeys(req.question_ids) if i in valid] or None
    player, verified = _player(request, req.player)
    game = GAMES.create(pack, player, verified, ids)
    game.coach = re.sub(r"[^A-Za-z ]", "", req.coach)[:20] or "Max"
    opening = game.opening()
    game.history.append({"role": "coach", "text": opening})
    return {
        "game_id": game.id,
        "player": player,
        "verified": verified,
        "practice": game.practice,
        "pack": _client_pack(pack),
        "say": opening,
        "question": game.public_question(),
        "state": game.state(),
    }


class TurnRequest(BaseModel):
    transcript: str = Field(default="", max_length=2000)
    intent: str | None = None


def _game(game_id: str) -> game_mod.Game:
    game = GAMES.get(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="This quiz has expired. Start a new one!")
    return game


@app.post("/api/game/{game_id}/turn")
async def turn(game_id: str, req: TurnRequest) -> dict[str, Any]:
    game = _game(game_id)
    if not req.transcript.strip() and req.intent not in game_mod.BUTTON_INTENTS:
        raise HTTPException(status_code=422, detail="Empty answer.")
    async with game.lock:
        return await game_mod.handle_turn(game, req.transcript or f"({req.intent})", req.intent)


@app.post("/api/game/{game_id}/stop")
async def stop(game_id: str) -> dict[str, Any]:
    game = _game(game_id)
    async with game.lock:
        if game.finished:
            return {"finished": True, "summary": game.summary, "state": game.state()}
        return game_mod.finish_now(game, "Okay, let's stop here.")


_placements: dict[str, dict[str, Any]] = {}


@app.post("/api/game/{game_id}/report")
async def report(game_id: str) -> dict[str, Any]:
    game = _game(game_id)
    if not game.finished or game.summary is None:
        raise HTTPException(status_code=409, detail="The quiz isn't finished yet.")
    coaching = await game_mod.coaching_report(game)
    placement = _placements.get(game.id)
    answered = any(r["verdict"] != "pending" for r in game.summary["review"])
    if placement is None and not game.practice and answered:
        s = game.summary
        placement = BOARD.add(
            {
                "player": game.player,
                "verified": game.verified,
                "score": s["score"],
                "accuracy": s["accuracy"],
                "correct": s["correct"],
                "total": s["total"],
                "best_streak": s["best_streak"],
                "rank": s["rank"]["title"],
                "slug": game.pack["slug"],
                "title": game.pack["title"],
                "emoji": game.pack.get("emoji", "🎓"),
                "difficulty": game.pack["difficulty"],
            }
        )
        _placements[game.id] = placement
    return {"summary": game.summary, "coaching": coaching, "placement": placement}


@app.get("/api/leaderboard")
def leaderboard(slug: str | None = None, period: str = "all", limit: int = 10) -> dict[str, Any]:
    limit = max(1, min(50, limit))
    return {"slug": slug, "period": period, "rows": BOARD.top(slug or None, period, limit)}


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict[str, Any]:
    return {"ok": True, "packs": len(PACKS), "games": len(GAMES._games)}


if __name__ == "__main__":
    import uvicorn

    asyncio.run(uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))).serve())
