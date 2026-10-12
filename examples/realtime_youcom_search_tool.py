"""Keyless You.com search tool for the packaged Realtime audio client."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx

from speech_to_speech.api.openai_realtime.audio_client import ToolResult

logger = logging.getLogger(__name__)

# Optional. Without a key the example uses You.com's free search endpoint (100 searches a day, no key).
YDC_API_KEY = os.environ.get("YDC_API_KEY", "").strip()
YOUCOM_FREE_URL = "https://api.you.com/v1/agents/search"
YOUCOM_SEARCH_URL = "https://ydc-index.io/v1/search"
YOUCOM_USER_AGENT = "youdotcom-integration/huggingface-speech-to-speech"
YOUCOM_PLATFORM_URL = (
    "https://you.com/platform?utm_source=huggingface-speech-to-speech"
    "&utm_medium=oss_integration&utm_campaign=2026-10-oss-integrations&utm_content=example"
)
MAX_RESULTS = 5

TOOLS = [
    {
        "type": "function",
        "name": "web_search",
        "description": (
            "Search the web for current or factual information you do not already know, such as news, prices, "
            "facts, or documentation. Returns the top results with titles, snippets, and URLs."
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


def _snippet(item: dict[str, Any]) -> str:
    """Combine the page description with the first query-matched snippet, skipping duplicates."""

    description = (item.get("description") or "").strip()
    snippet = ((item.get("snippets") or [""])[0] or "").strip()
    if not description or not snippet:
        return description or snippet
    if snippet == description:
        return description
    return f"{description} {snippet}"


async def execute_tool(name: str, arguments: dict[str, Any]) -> ToolResult:
    """Execute the declared web search tool and request a spoken follow-up."""

    if name != "web_search":
        raise ValueError(f"Unknown tool: {name}")

    query = str(arguments.get("query", "")).strip()
    if not query:
        return ToolResult({"error": "query must be a non-empty string"})

    headers = {"User-Agent": YOUCOM_USER_AGENT}
    if YDC_API_KEY:
        url = YOUCOM_SEARCH_URL
        headers["X-API-Key"] = YDC_API_KEY
    else:
        url = YOUCOM_FREE_URL

    logger.info("web_search query=%s max_results=%d", query, MAX_RESULTS)
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            response = await client.get(
                url,
                headers=headers,
                params={"query": query, "count": MAX_RESULTS},
            )
    except httpx.RequestError as exc:
        logger.warning("You.com unreachable for %r: %s", query, exc)
        return ToolResult(
            {
                "query": query,
                "error": "Search provider is temporarily unreachable.",
            }
        )

    if response.status_code == 429 and not YDC_API_KEY:
        logger.warning("You.com free search limit reached for %r", query)
        return ToolResult(
            {
                "query": query,
                "error": (
                    "The free You.com search limit (100 searches a day) is used up. "
                    f"Set YDC_API_KEY for more searches. Get a key at {YOUCOM_PLATFORM_URL}."
                ),
            }
        )

    if response.status_code != 200:
        message = None
        try:
            data = response.json()
            if isinstance(data, dict):
                message = data.get("message") or data.get("detail") or data.get("error")
        except ValueError:
            pass
        detail = f"Search provider error ({response.status_code})"
        if isinstance(message, str) and message:
            detail += f": {message}"
        logger.warning("You.com returned HTTP %d for %r", response.status_code, query)
        return ToolResult({"query": query, "error": detail})

    try:
        data = response.json()
    except ValueError:
        logger.warning("You.com returned invalid JSON for %r", query)
        return ToolResult({"query": query, "error": "Search provider returned an invalid response."})

    results = data.get("results") or {}
    hits = (results.get("web") or []) + (results.get("news") or [])

    return ToolResult(
        {
            "query": query,
            "searched_at": datetime.now(timezone.utc).date().isoformat(),
            # You.com does not return an answer box; the model answers from the results.
            "answer": None,
            "results": [
                {
                    "title": item.get("title", ""),
                    "snippet": _snippet(item),
                    "url": item.get("url", ""),
                }
                for item in hits[:MAX_RESULTS]
            ],
        }
    )
