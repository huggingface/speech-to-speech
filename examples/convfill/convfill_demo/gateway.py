"""Local host for the Hugging Face realtime voice interface.

The speech pipeline is a separate ``speech-to-speech serve`` process. This app
only serves the browser UI, publishes its loopback realtime URL, and keeps all
inference credentials on the server side.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path

import httpx
import websockets
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from convfill_demo.keys import get_endpoint_api_key

HERE = Path(__file__).resolve().parents[3] / "demo"
SPEECH_TO_SPEECH_URL = os.environ.get("SPEECH_TO_SPEECH_URL", "").strip()
PIPELINE_URL = os.environ.get("CONVFILL_PIPELINE_URL", "ws://127.0.0.1:8765/v1/realtime")
_session_active = False
DEFAULT_STARTUP_GREETING = "Greet the user warmly in one short sentence, then invite them to talk."
SERPER_KEY = get_endpoint_api_key("SERPER_API_KEY")
SEARCH_PROXY_URL = os.environ.get("CONVFILL_SEARCH_URL", "").strip()
SEARCH_PROXY_KEY = get_endpoint_api_key("CONVFILL_SEARCH_API_KEY")
STARTUP_GREETING = os.environ.get("STARTUP_GREETING", DEFAULT_STARTUP_GREETING).strip()

logger = logging.getLogger("convfill.speech_demo")


class RedactAccessGrants(logging.Filter):
    """Uvicorn logs WS URLs even when HTTP access logging is disabled."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        cleaned = re.sub(r"([?&](?:__sign|session_token)=)[^&\s\"']+", r"\1<redacted>", message)
        if cleaned != message:
            record.msg, record.args = cleaned, ()
        return True


logging.getLogger("uvicorn.error").addFilter(RedactAccessGrants())
app = FastAPI(title="ConvFill Hugging Face Voice Demo")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def config(request: Request) -> dict[str, object]:
    scheme = "wss" if request.url.scheme == "https" else "ws"
    return {
        "search": bool(SERPER_KEY or SEARCH_PROXY_URL),
        "tools": True,
        "lb": False,
        "allowDirect": True,
        "s2sUrl": SPEECH_TO_SPEECH_URL or f"{scheme}://{request.url.netloc}/v1/realtime",
        "rtc": False,
        "iceServers": [],
        "startupGreeting": STARTUP_GREETING,
        "auth": False,
    }


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    key: str | None = None


@app.post("/api/search")
async def search(req: SearchRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "Empty query.")
    if not SERPER_KEY and SEARCH_PROXY_URL:
        # An existing server may already hold a non-exportable search secret.
        # This destination is deploy-pinned, never supplied by the browser.
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    SEARCH_PROXY_URL,
                    headers={"Authorization": "Bearer " + SEARCH_PROXY_KEY} if SEARCH_PROXY_KEY else {},
                    json={"query": query},
                )
            response.raise_for_status()
            data = response.json()
            return {
                "query": query,
                "answer": data.get("answer"),
                "results": [
                    {key: item.get(key, "") for key in ("title", "snippet", "url")}
                    for item in (data.get("results") or [])[:5]
                ],
            }
        except (httpx.HTTPError, ValueError):
            raise HTTPException(502, "Search provider unavailable.")
    key = SERPER_KEY or (req.key or "").strip()
    if not key:
        raise HTTPException(503, "Search is not configured.")
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            response = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": key},
                json={"q": query, "num": 5},
            )
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError):
        # Never relay provider error bodies: they may contain request data.
        raise HTTPException(502, "Search provider unavailable.")
    box = data.get("answerBox") or {}
    graph = data.get("knowledgeGraph") or {}
    return {
        "query": query,
        "answer": box.get("answer") or box.get("snippet") or graph.get("description"),
        "results": [
            {"title": item.get("title", ""), "snippet": item.get("snippet", ""), "url": item.get("link", "")}
            for item in (data.get("organic") or [])[:5]
        ],
    }


@app.get("/readyz")
async def readyz():
    # Readiness requires the pipeline to accept a session, not just a static UI.
    target = PIPELINE_URL.replace("wss://", "https://", 1).replace("ws://", "http://", 1)
    target = target.rsplit("/v1/realtime", 1)[0] + "/v1/pool"
    try:
        async with httpx.AsyncClient(timeout=3, transport=httpx.AsyncHTTPTransport(retries=2)) as client:
            response = await client.get(target)
            response.raise_for_status()
            for stage in ("REASONER", "TALKER"):
                base_url = os.environ.get(f"CONVFILL_{stage}_URL")
                if base_url:
                    key = get_endpoint_api_key(f"CONVFILL_{stage}_API_KEY")
                    headers = {"Authorization": f"Bearer {key}"} if key else {}
                    response = await client.get(base_url.rstrip("/") + "/models", headers=headers)
                    response.raise_for_status()
                    models = [model["id"] for model in response.json().get("data", [])]
                    expected = (
                        os.environ.get("CONVFILL_REASONER_MODEL", "nvidia/Gemma-4-31B-IT-NVFP4")
                        if stage == "REASONER"
                        else "vysri/Qwen-0.6B-ConvFill"
                    )
                    if expected not in models:
                        raise ValueError("Configured model is not ready")
    except (httpx.HTTPError, ValueError):
        raise HTTPException(503, "Speech pipeline is not ready")
    return {"status": "ready", "session_active": _session_active}


@app.websocket("/v1/realtime")
async def realtime(socket: WebSocket):
    global _session_active
    # Reject cross-origin browser connections and concurrent sessions. The
    # gateway must restrict access to this test demo (e.g. a private HF Space).
    origin = socket.headers.get("origin")
    host = socket.headers.get("host")
    if origin and origin not in (f"https://{host}", f"http://{host}"):
        await socket.close(code=1008)
        return
    if _session_active:
        await socket.close(code=1013)
        return
    _session_active = True
    tasks = []
    try:
        # Browsers require a selected protocol when the SDK offers protocols.
        # Python WS clients allow a missing selection, hiding this mismatch.
        protocols = socket.scope.get("subprotocols", [])
        await socket.accept(subprotocol="realtime" if "realtime" in protocols else None)
        async with websockets.connect(PIPELINE_URL, open_timeout=10, max_size=8 * 1024 * 1024) as backend:

            async def upstream():
                while True:
                    message = await socket.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    await backend.send(message.get("text") or message.get("bytes") or "")

            async def downstream():
                async for message in backend:
                    if isinstance(message, bytes):
                        await socket.send_bytes(message)
                    else:
                        await socket.send_text(message)

            tasks = [asyncio.create_task(upstream()), asyncio.create_task(downstream())]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    except (WebSocketDisconnect, websockets.exceptions.WebSocketException, OSError):
        pass
    finally:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        _session_active = False
        try:
            await socket.close()
        except (RuntimeError, WebSocketDisconnect):
            pass


@app.get("/api/me")
def me() -> dict[str, bool]:
    return {"enabled": False}


@app.get("/vendor/openai-realtime-agents.umd.js", include_in_schema=False)
def agents_sdk_bundle() -> FileResponse:
    path = HERE / "node_modules" / "@openai" / "agents-realtime" / "dist" / "bundle" / "openai-realtime-agents.umd.js"
    if not path.is_file():
        raise HTTPException(
            status_code=503,
            detail="Browser dependencies are missing; run npm ci --prefix demo.",
        )
    return FileResponse(path, media_type="text/javascript")


# Registered last so API and vendor routes win.
app.mount("/", StaticFiles(directory=HERE, html=True), name="static")
