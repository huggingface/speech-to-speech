"""Serper-backed Google search tool for the packaged Realtime audio client."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx

from speech_to_speech.api.openai_realtime.audio_client import ToolResult

logger = logging.getLogger(__name__)

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "").strip()
SERPER_URL = "https://google.serper.dev/search"
MAX_RESULTS = 5

TOOLS = [
    {
        "type": "function",
        "name": "web_search",
        "description": (
            "Search the web for current or factual information you do not already know, such as news, prices, "
            "facts, or documentation. Returns a direct answer when Google provides one, followed by the top "
            "results with titles, snippets, and URLs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The web search query.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }
]


async def execute_tool(name: str, arguments: dict[str, Any]) -> ToolResult:
    """Execute the declared web search tool and request a spoken follow-up."""

    if name != "web_search":
        raise ValueError(f"Unknown tool: {name}")

    query = str(arguments.get("query", "")).strip()
    if not query:
        return ToolResult({"error": "query must be a non-empty string"})
    if not SERPER_API_KEY:
        return ToolResult(
            {
                "query": query,
                "error": "SERPER_API_KEY is not set. Get a free API key at https://serper.dev/.",
            }
        )

    logger.info("web_search query=%s max_results=%d", query, MAX_RESULTS)
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            response = await client.post(
                SERPER_URL,
                headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                json={"q": query, "num": MAX_RESULTS},
            )
    except httpx.RequestError as exc:
        logger.warning("Serper unreachable for %r: %s", query, exc)
        return ToolResult(
            {
                "query": query,
                "error": "Search provider is temporarily unreachable.",
            }
        )

    if response.status_code != 200:
        message = None
        try:
            data = response.json()
            message = data.get("message") if isinstance(data, dict) else None
        except ValueError:
            pass
        detail = f"Search provider error ({response.status_code})"
        if message:
            detail += f": {message}"
        logger.warning("Serper returned HTTP %d for %r", response.status_code, query)
        return ToolResult({"query": query, "error": detail})

    try:
        data = response.json()
    except ValueError:
        logger.warning("Serper returned invalid JSON for %r", query)
        return ToolResult({"query": query, "error": "Search provider returned an invalid response."})

    answer_box = data.get("answerBox") or {}
    answer = answer_box.get("answer") or answer_box.get("snippet") or None
    if not answer:
        knowledge_graph = data.get("knowledgeGraph") or {}
        answer = knowledge_graph.get("description") or None

    return ToolResult(
        {
            "query": query,
            "searched_at": datetime.now(timezone.utc).date().isoformat(),
            "answer": answer,
            "results": [
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link", ""),
                }
                for item in (data.get("organic") or [])[:MAX_RESULTS]
            ],
        }
    )
