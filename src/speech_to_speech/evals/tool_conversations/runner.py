"""Drive scripted conversations through a running Realtime server.

Text in, tool calls and spoken text out: each user turn is a ``conversation.item.create``
followed by ``response.create``. Tool calls are answered by the client; a call to a
long-running tool is answered ``LATE_RESULT_AFTER_TURNS`` user turns later, followed by its own
``response.create``, which exercises the server's running-tool handling.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from speech_to_speech.evals.tool_conversations.scenarios import (
    INSTRUCTIONS,
    LATE_RESULT_AFTER_TURNS,
    LONG_RUNNING_TOOLS,
    RESULTS,
    TOOLS,
    Conversation,
)

MAX_TOOL_ROUNDS = 3
RESPONSE_TIMEOUT_S = 120.0


@dataclass
class RunnerConfig:
    url: str = "ws://127.0.0.1:8765/v1/realtime"
    headers: dict[str, str] = field(default_factory=dict)
    concurrency: int = 3
    connect_attempts: int = 30


@dataclass
class Response:
    text: str
    calls: list[tuple[str, str, str]]
    status: Optional[str]
    errors: list[str]


async def _response(ws: Any) -> Response:
    await ws.send(json.dumps({"type": "response.create"}))
    errors: list[str] = []
    while True:
        event = json.loads(await asyncio.wait_for(ws.recv(), RESPONSE_TIMEOUT_S))
        kind = event.get("type")
        if kind == "error":
            error = event.get("error") or {}
            errors.append(str(error.get("code") or error.get("message") or "error"))
        elif kind == "response.done":
            body = event.get("response") or {}
            text = ""
            calls = []
            for item in body.get("output") or []:
                if item.get("type") == "message":
                    for part in item.get("content") or []:
                        text += part.get("text") or part.get("transcript") or ""
                elif item.get("type") == "function_call":
                    calls.append((item["name"], item["call_id"], item.get("arguments") or ""))
            return Response(text.strip(), calls, body.get("status"), errors)


async def _send_output(ws: Any, call_id: str, name: str) -> None:
    item = {"type": "function_call_output", "call_id": call_id, "output": json.dumps(RESULTS.get(name, {}))}
    await ws.send(json.dumps({"type": "conversation.item.create", "item": item}))


async def run_conversation(config: RunnerConfig, conversation: Conversation) -> dict[str, Any]:
    """Run one conversation and return its turns as plain dicts."""
    import websockets

    turns: list[dict[str, Any]] = []
    errors: list[str] = []
    pending: list[dict[str, Any]] = []
    async with websockets.connect(config.url, max_size=None, additional_headers=config.headers or None) as ws:
        session = {
            "type": "realtime",
            "instructions": INSTRUCTIONS,
            "tools": TOOLS,
            "audio": {"input": {"turn_detection": None}},
        }
        await ws.send(json.dumps({"type": "session.update", "session": session}))
        for index, turn in enumerate(conversation):
            for call in [c for c in pending if index - c["turn"] >= LATE_RESULT_AFTER_TURNS]:
                pending.remove(call)
                await _send_output(ws, call["call_id"], call["name"])
                reply = await _response(ws)
                errors += reply.errors
                turns.append(
                    {
                        "kind": "late_result",
                        "tool": call["name"],
                        "text": reply.text,
                        "calls": [c[0] for c in reply.calls],
                        "status": reply.status,
                    }
                )
            item = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": turn.ask}]}
            await ws.send(json.dumps({"type": "conversation.item.create", "item": item}))
            rounds = []
            for _ in range(MAX_TOOL_ROUNDS):
                reply = await _response(ws)
                errors += reply.errors
                rounds.append(
                    {
                        "text": reply.text,
                        "calls": [c[0] for c in reply.calls],
                        "arguments": [c[2] for c in reply.calls],
                        "status": reply.status,
                    }
                )
                answered = False
                for name, call_id, _arguments in reply.calls:
                    if name in LONG_RUNNING_TOOLS:
                        pending.append({"call_id": call_id, "name": name, "turn": index})
                    else:
                        await _send_output(ws, call_id, name)
                        answered = True
                if not answered:
                    break
            turns.append(
                {
                    "kind": "user",
                    "index": index,
                    "ask": turn.ask,
                    "expected_tool": turn.expected_tool,
                    "expected_arguments": turn.expected_arguments,
                    "rounds": rounds,
                }
            )
    return {"turns": turns, "errors": errors}


async def run_set(
    config: RunnerConfig, conversations: tuple[Conversation, ...], reps: int, *, seed: int = 0
) -> list[dict[str, Any]]:
    """Run every conversation ``reps`` times, ``config.concurrency`` at a time."""
    jobs = [(index, rep) for index in range(len(conversations)) for rep in range(reps)]
    random.Random(seed).shuffle(jobs)
    semaphore = asyncio.Semaphore(config.concurrency)
    results: list[dict[str, Any]] = []

    async def one(index: int, rep: int) -> None:
        async with semaphore:
            started = time.monotonic()
            row: dict[str, Any] = {"conversation": index, "rep": rep}
            for _attempt in range(config.connect_attempts):
                try:
                    row.update(await run_conversation(config, conversations[index]))
                    row.pop("crash", None)
                    break
                except Exception as exc:  # noqa: BLE001 - recorded in the report
                    row["crash"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    # The server frees a pipeline slot shortly after a client disconnects.
                    if "slots are in use" not in str(exc):
                        break
                    await asyncio.sleep(2)
            row["seconds"] = round(time.monotonic() - started, 1)
            results.append(row)

    await asyncio.gather(*(one(index, rep) for index, rep in jobs))
    return sorted(results, key=lambda r: (r["conversation"], r["rep"]))
