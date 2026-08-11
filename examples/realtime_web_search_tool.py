"""Keyless DuckDuckGo search tool for the packaged Realtime audio client."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from speech_to_speech.api.openai_realtime.audio_client import ToolResult

logger = logging.getLogger(__name__)

MAX_RESULTS = 5
CURRENT_UTC_DATE = datetime.now(timezone.utc).date().isoformat()

TOOLS = [
    {
        "type": "function",
        "name": "web_search",
        "description": (
            "Search the web for current information and return a short list of results with titles, snippets, "
            f"and URLs. The current UTC date is {CURRENT_UTC_DATE}; resolve relative dates such as today and "
            "tomorrow from it. Call this whenever the user asks to search, check the web, look something up, "
            "find current events, or learn what is happening now. Usually make one focused search; search again "
            "only when the first results are insufficient or conflicting."
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


def _search_web(query: str) -> list[dict[str, Any]]:
    """Run the synchronous DDGS client outside the Realtime event loop."""

    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=MAX_RESULTS))


async def execute_tool(name: str, arguments: dict[str, Any]) -> ToolResult:
    """Execute the declared web search tool and request a spoken follow-up."""

    if name != "web_search":
        raise ValueError(f"Unknown tool: {name}")

    query = str(arguments.get("query", "")).strip()
    if not query:
        return ToolResult({"error": "query must be a non-empty string"})

    logger.info("web_search query=%s max_results=%d", query, MAX_RESULTS)
    try:
        hits = await asyncio.to_thread(_search_web, query)
    except DDGSException as exc:
        logger.warning("web_search failed for %r: %s", query, exc)
        return ToolResult(
            {
                "query": query,
                "error": "Web search is temporarily unavailable.",
            }
        )

    return ToolResult(
        {
            "query": query,
            "searched_at": datetime.now(timezone.utc).date().isoformat(),
            "results": [
                {
                    "title": hit.get("title", ""),
                    "snippet": hit.get("body", ""),
                    "url": hit.get("href", ""),
                }
                for hit in hits
            ],
        }
    )
