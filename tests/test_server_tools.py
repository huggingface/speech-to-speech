"""Unit tests for the Keenable-backed server tools (LLM/server_tools.py).

No network: httpx.MockTransport stands in for the Keenable API.

Run with pytest, or standalone:  python tests/test_server_tools.py
"""

from __future__ import annotations

import json

import httpx

from speech_to_speech.LLM.server_tools import (
    FETCH_PAGE_TOOL,
    WEB_SEARCH_TOOL,
    KeenableWebTools,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _search_payload(n: int) -> dict:
    return {
        "query": "q",
        "results": [
            {
                "title": f"Title {i}",
                "url": f"https://example.com/{i}",
                "description": f"Desc {i}",
                "snippet": "s" * 2000,
                "published_at": "2026-01-01T00:00:00Z",
                "acquired_at": "2026-01-02T00:00:00Z",
            }
            for i in range(n)
        ],
    }


def _make_tools(handler_fn, **kwargs) -> tuple[KeenableWebTools, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler_fn(request)

    tools = KeenableWebTools(transport=httpx.MockTransport(record), **kwargs)
    return tools, requests


# ── Definitions ──────────────────────────────────────────────────────────────


def test_tool_definitions_are_flat_function_dicts():
    tools, _ = _make_tools(lambda r: httpx.Response(200, json={}), api_key="keen_x")
    defs = tools.tool_definitions
    assert [d["name"] for d in defs] == ["web_search", "fetch_page"]
    for d in defs:
        assert d["type"] == "function"
        assert "function" not in d  # flat Responses-API shape, not Chat-Completions nested
        assert d["parameters"]["type"] == "object"
    assert WEB_SEARCH_TOOL.parameters["required"] == ["query"]
    assert FETCH_PAGE_TOOL.parameters["required"] == ["url"]


def test_voice_guidance_mentions_tools_and_date():
    tools, _ = _make_tools(lambda r: httpx.Response(200, json={}), api_key="keen_x")
    guidance = tools.voice_guidance()
    assert "web_search" in guidance
    assert "Today's date" in guidance


# ── web_search ───────────────────────────────────────────────────────────────


def test_search_uses_keyed_endpoint_and_header():
    tools, requests = _make_tools(lambda r: httpx.Response(200, json=_search_payload(2)), api_key="keen_secret")
    out = json.loads(tools.execute("web_search", json.dumps({"query": "ai news"})))
    assert len(out["results"]) == 2
    (request,) = requests
    assert request.url.path == "/v1/search"
    assert request.headers["X-API-Key"] == "keen_secret"
    assert json.loads(request.content) == {"query": "ai news", "mode": "realtime"}


def test_search_keyless_uses_public_endpoint(monkeypatch):
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    tools, requests = _make_tools(lambda r: httpx.Response(200, json=_search_payload(1)))
    tools.execute("web_search", json.dumps({"query": "x"}))
    (request,) = requests
    assert request.url.path == "/v1/search/public"
    assert "X-API-Key" not in request.headers


def test_search_truncates_results_and_snippets():
    tools, _ = _make_tools(lambda r: httpx.Response(200, json=_search_payload(9)), api_key="keen_x")
    out = json.loads(tools.execute("web_search", json.dumps({"query": "x"})))
    assert len(out["results"]) == 5  # max_results default
    assert all(len(r["snippet"]) <= 600 for r in out["results"])
    assert out["results"][0]["title"] == "Title 0"
    assert out["results"][0]["url"] == "https://example.com/0"


def test_search_passes_optional_filters_and_drops_unknown():
    tools, requests = _make_tools(lambda r: httpx.Response(200, json=_search_payload(0)), api_key="keen_x")
    tools.execute(
        "web_search",
        json.dumps({"query": "x", "site": "reuters.com", "published_after": "7d", "bogus": "y"}),
    )
    body = json.loads(requests[0].content)
    assert body == {"query": "x", "mode": "realtime", "site": "reuters.com", "published_after": "7d"}


def test_search_requires_query():
    tools, requests = _make_tools(lambda r: httpx.Response(200, json={}), api_key="keen_x")
    out = json.loads(tools.execute("web_search", "{}"))
    assert "error" in out
    assert not requests  # nothing sent


# ── fetch_page ───────────────────────────────────────────────────────────────


def test_fetch_page_get_with_params():
    payload = {"url": "https://example.com/a", "title": "T", "content": "body"}
    tools, requests = _make_tools(lambda r: httpx.Response(200, json=payload), api_key="keen_x")
    out = json.loads(tools.execute("fetch_page", json.dumps({"url": "https://example.com/a"})))
    assert out == {"url": "https://example.com/a", "title": "T", "content": "body"}
    (request,) = requests
    assert request.method == "GET"
    assert request.url.path == "/v1/fetch"
    assert request.url.params["url"] == "https://example.com/a"
    assert request.url.params["max_chars"] == "6000"


def test_fetch_page_retries_live_on_404():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("live") == "true":
            return httpx.Response(200, json={"url": "u", "title": "T", "content": "live!"})
        return httpx.Response(404, json={"error": "Not found"})

    tools, requests = _make_tools(handler, api_key="keen_x")
    out = json.loads(tools.execute("fetch_page", json.dumps({"url": "https://example.com/new"})))
    assert out["content"] == "live!"
    assert len(requests) == 2
    assert requests[1].url.params["live"] == "true"


def test_fetch_page_rejects_private_targets():
    tools, requests = _make_tools(lambda r: httpx.Response(200, json={}), api_key="keen_x")
    for url in (
        "ftp://example.com/x",
        "http://localhost/x",
        "http://127.0.0.1/x",
        "http://10.0.0.8/x",
        "http://[::1]/x",
        "http://metadata.google.internal/x",
    ):
        out = json.loads(tools.execute("fetch_page", json.dumps({"url": url})))
        assert "error" in out, url
    assert not requests  # nothing left the box


# ── execute error handling ───────────────────────────────────────────────────


def test_execute_never_raises():
    tools, _ = _make_tools(lambda r: httpx.Response(429, json={"error": "Rate limit exceeded"}), api_key="keen_x")
    out = json.loads(tools.execute("web_search", json.dumps({"query": "x"})))
    assert "429" in out["error"]
    assert "Rate limit exceeded" in out["error"]

    assert "error" in json.loads(tools.execute("web_search", "not json"))
    assert "error" in json.loads(tools.execute("web_search", "[1, 2]"))
    assert "error" in json.loads(tools.execute("nope", "{}"))


def test_execute_timeout_returns_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("boom")

    tools, _ = _make_tools(handler, api_key="keen_x")
    out = json.loads(tools.execute("web_search", json.dumps({"query": "x"})))
    assert out == {"error": "The web request timed out."}


def test_owns():
    tools, _ = _make_tools(lambda r: httpx.Response(200, json={}), api_key="keen_x")
    assert tools.owns("web_search") and tools.owns("fetch_page")
    assert not tools.owns("camera_snapshot")


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
