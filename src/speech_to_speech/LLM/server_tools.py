"""Server-executed web tools backed by the Keenable API (https://docs.keenable.ai).

These tools are advertised to the language model alongside any client-registered
tools, but they are executed *inside* the pipeline by
:class:`~speech_to_speech.LLM.base_openai_compatible_language_model.BaseOpenAICompatibleHandler`:
the tool call never reaches the realtime client, its output is written to the
conversation as a ``function_call_output``, and the model is re-queried in the
same turn. Any realtime client therefore gets live-web answers with no tool
handling of its own.

Keenable is keyless by default (rate-limited free tier); an API key
(``keen_...``, via ``--keenable_api_key`` or the ``KEENABLE_API_KEY`` env var)
switches to the authenticated endpoints.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import time
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

from speech_to_speech.LLM.tool_call.function_tool import FunctionTool

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.keenable.ai"
_BASE_URL_ENV = "KEENABLE_API_URL"
_API_KEY_ENV = "KEENABLE_API_KEY"

WEB_SEARCH_TOOL = FunctionTool(
    type="function",
    name="web_search",
    description=(
        "Search the live web. Use for anything recent or uncertain: news, current events, "
        "weather, sports, prices, schedules, releases, or facts that may have changed since "
        "your training. Returns ranked results with titles, URLs, descriptions and snippets."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Describe the ideal page in natural language.",
            },
            "site": {
                "type": "string",
                "description": "Optional: restrict results to one domain, e.g. 'reuters.com'.",
            },
            "published_after": {
                "type": "string",
                "description": "Optional: only pages published after this date ('YYYY-MM-DD' or relative like '7d').",
            },
        },
        "required": ["query"],
    },
)

FETCH_PAGE_TOOL = FunctionTool(
    type="function",
    name="fetch_page",
    description=(
        "Fetch a web page's main content as markdown. Use after web_search when a snippet "
        "is not enough and you need to read the page itself."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The http(s) URL of the page to fetch (usually from a web_search result).",
            },
        },
        "required": ["url"],
    },
)


def _reject_private_fetch_target(url: str) -> str | None:
    """Return an error message when *url* points at a private/internal target.

    The Keenable backend enforces this server-side too; the client-side guard
    avoids sending internal hostnames off-box in the first place. Only IP
    literals and well-known internal names are checked — DNS names fall through
    to the backend's guard.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        return f"Refusing to fetch a non-http(s) URL: {url!r}"
    host = (parts.hostname or "").strip().lower().rstrip(".")
    if not host:
        return f"Refusing to fetch a URL with no host: {url!r}"
    if host in ("localhost", "metadata.google.internal"):
        return f"Refusing to fetch a private/internal host: {host!r}"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
        return f"Refusing to fetch a private/internal address: {host!r}"
    return None


class KeenableWebTools:
    """Keenable-backed ``web_search`` / ``fetch_page`` tools with a shared HTTP client.

    ``execute`` never raises: transport and API failures come back as an
    ``{"error": ...}`` JSON string so the model can recover conversationally
    instead of the turn dying.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout_s: float = 10.0,
        max_results: int = 5,
        snippet_max_chars: int = 600,
        fetch_max_chars: int = 6000,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get(_API_KEY_ENV) or None
        self.max_results = max_results
        self.snippet_max_chars = snippet_max_chars
        self.fetch_max_chars = fetch_max_chars
        base_url = (os.environ.get(_BASE_URL_ENV) or _DEFAULT_BASE_URL).rstrip("/")
        headers = {"User-Agent": "speech-to-speech-keenable/1.0"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        # Keyless traffic uses the public endpoints (rate-limited free tier).
        self._path_suffix = "" if self.api_key else "/public"
        self._client = httpx.Client(base_url=base_url, headers=headers, timeout=timeout_s, transport=transport)
        self._handlers = {
            WEB_SEARCH_TOOL.name: self._web_search,
            FETCH_PAGE_TOOL.name: self._fetch_page,
        }

    @property
    def tool_definitions(self) -> list[dict[str, Any]]:
        """Flat Responses-API-style function tool dicts, ready to merge into a request."""
        return [t.model_dump(exclude_none=True) for t in (WEB_SEARCH_TOOL, FETCH_PAGE_TOOL)]

    def voice_guidance(self) -> str:
        """System-prompt suffix teaching the model when/how to use the tools in a voice chat."""
        today = datetime.now().strftime("%A, %B %d, %Y")
        return (
            f"Today's date is {today}. You can access the live web with the web_search and "
            "fetch_page tools; they run on the server and return results to you immediately, "
            "so call them whenever the user asks about news, weather, sports, prices, or any "
            "fact that may have changed recently. Call the tool immediately and silently: no "
            "announcements, no permission-asking, no filler like 'Let me check' — results come "
            "back fast enough to answer right away. When answering from web results, stay "
            "spoken-friendly: summarize in one or two short sentences, name the source when "
            "useful, and never read URLs or long lists aloud."
        )

    def owns(self, name: str) -> bool:
        return name in self._handlers

    def execute(self, name: str, arguments_json: str) -> str:
        """Run tool *name* with the model-provided JSON arguments; always returns JSON."""
        try:
            arguments = json.loads(arguments_json or "{}")
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            return self._error(f"Invalid tool arguments: {exc}")
        handler = self._handlers.get(name)
        if handler is None:
            return self._error(f"Unknown tool: {name}")
        try:
            return handler(arguments)
        except httpx.TimeoutException:
            return self._error("The web request timed out.")
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                detail = exc.response.json().get("error") or ""
            except Exception:
                pass
            return self._error(f"Keenable API error {exc.response.status_code}: {detail}".strip())
        except httpx.HTTPError as exc:
            return self._error(f"Web request failed: {exc}")

    @staticmethod
    def _error(message: str) -> str:
        logger.warning("Keenable tool error: %s", message)
        return json.dumps({"error": message}, ensure_ascii=False)

    def _web_search(self, arguments: dict[str, Any]) -> str:
        query = arguments.get("query")
        if not query or not isinstance(query, str):
            return self._error("web_search requires a 'query' string argument.")
        # "realtime" mode trades a little depth for consistent ~200ms responses —
        # the right point on the curve when a spoken reply is waiting on it.
        payload: dict[str, Any] = {"query": query, "mode": "realtime"}
        for key in ("site", "published_after"):
            value = arguments.get(key)
            if value and isinstance(value, str):
                payload[key] = value
        started = time.perf_counter()
        response = self._client.post(f"/v1/search{self._path_suffix}", json=payload)
        took_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        results = response.json().get("results") or []
        compact = [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "description": r.get("description"),
                "snippet": (r.get("snippet") or "")[: self.snippet_max_chars] or None,
                "published_at": r.get("published_at"),
            }
            for r in results[: self.max_results]
        ]
        logger.info("Keenable web_search(%r) -> %d results in %.0f ms", query, len(compact), took_ms)
        return json.dumps({"results": compact}, ensure_ascii=False)

    def _fetch_page(self, arguments: dict[str, Any]) -> str:
        url = arguments.get("url")
        if not url or not isinstance(url, str):
            return self._error("fetch_page requires a 'url' string argument.")
        rejection = _reject_private_fetch_target(url)
        if rejection is not None:
            return self._error(rejection)
        params: dict[str, Any] = {"url": url, "max_chars": self.fetch_max_chars}
        started = time.perf_counter()
        response = self._client.get(f"/v1/fetch{self._path_suffix}", params=params)
        if response.status_code == 404:
            # Not in the index yet — retry live from the source.
            response = self._client.get(f"/v1/fetch{self._path_suffix}", params={**params, "live": True})
        took_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        data = response.json()
        logger.info("Keenable fetch_page(%r) -> %d chars in %.0f ms", url, len(data.get("content") or ""), took_ms)
        return json.dumps(
            {
                "url": data.get("url"),
                "title": data.get("title"),
                "content": data.get("content"),
            },
            ensure_ascii=False,
        )

    def close(self) -> None:
        self._client.close()
