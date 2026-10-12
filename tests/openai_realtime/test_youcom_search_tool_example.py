import pytest
from examples import realtime_youcom_search_tool


class FakeResponse:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data


class FakeAsyncClient:
    def __init__(self, response, captured, **_kwargs):
        self.response = response
        self.captured = captured

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, url, **kwargs):
        self.captured.update(url=url, **kwargs)
        return self.response


def _patch_client(monkeypatch, response, captured):
    monkeypatch.setattr(
        realtime_youcom_search_tool.httpx,
        "AsyncClient",
        lambda **kwargs: FakeAsyncClient(response, captured, **kwargs),
    )


@pytest.mark.asyncio
async def test_youcom_search_without_key_uses_free_endpoint(monkeypatch):
    captured = {}
    response = FakeResponse(
        200,
        {
            "results": {
                "web": [
                    {
                        "title": "Forecast",
                        "description": "Bern weather",
                        "snippets": ["Sunny, 21°C"],
                        "url": "https://example.com/weather",
                    },
                    {"title": "No snippets", "description": "Fallback text", "url": "https://example.com/other"},
                ]
            }
        },
    )
    monkeypatch.setattr(realtime_youcom_search_tool, "YDC_API_KEY", "")
    _patch_client(monkeypatch, response, captured)

    result = await realtime_youcom_search_tool.execute_tool("web_search", {"query": "weather in Bern"})

    assert result.output["answer"] is None
    assert result.output["results"] == [
        {"title": "Forecast", "snippet": "Bern weather Sunny, 21°C", "url": "https://example.com/weather"},
        {"title": "No snippets", "snippet": "Fallback text", "url": "https://example.com/other"},
    ]
    assert captured == {
        "url": realtime_youcom_search_tool.YOUCOM_FREE_URL,
        "headers": {"User-Agent": realtime_youcom_search_tool.YOUCOM_USER_AGENT},
        "params": {"query": "weather in Bern", "count": realtime_youcom_search_tool.MAX_RESULTS},
    }


@pytest.mark.asyncio
async def test_youcom_search_with_key_uses_search_api(monkeypatch):
    captured = {}
    monkeypatch.setattr(realtime_youcom_search_tool, "YDC_API_KEY", "test-key")
    _patch_client(monkeypatch, FakeResponse(200, {"results": {"web": []}}), captured)

    result = await realtime_youcom_search_tool.execute_tool("web_search", {"query": "latest news"})

    assert result.output["results"] == []
    assert captured["url"] == realtime_youcom_search_tool.YOUCOM_SEARCH_URL
    assert captured["headers"] == {
        "User-Agent": realtime_youcom_search_tool.YOUCOM_USER_AGENT,
        "X-API-Key": "test-key",
    }


@pytest.mark.asyncio
async def test_youcom_search_reports_free_limit(monkeypatch):
    monkeypatch.setattr(realtime_youcom_search_tool, "YDC_API_KEY", "")
    _patch_client(monkeypatch, FakeResponse(429, {}), {})

    result = await realtime_youcom_search_tool.execute_tool("web_search", {"query": "latest news"})

    assert result.output["query"] == "latest news"
    assert "YDC_API_KEY" in result.output["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("longer_description", [True, False])
async def test_youcom_search_preserves_distinct_text_with_a_shared_prefix(monkeypatch, longer_description):
    prefix = "Bern weather forecast for today and tonight. "
    detail = prefix + "Read our detailed outlook for rainfall, wind and sunshine across the entire Bern region."
    temperature = prefix + "Current temperature: 21°C."
    description, snippet = (detail, temperature) if longer_description else (temperature, detail)
    response = FakeResponse(
        200,
        {
            "results": {
                "web": [
                    {
                        "title": "Forecast",
                        "description": description,
                        "snippets": [snippet],
                        "url": "https://example.com/weather",
                    }
                ]
            }
        },
    )
    monkeypatch.setattr(realtime_youcom_search_tool, "YDC_API_KEY", "")
    _patch_client(monkeypatch, response, {})

    result = await realtime_youcom_search_tool.execute_tool("web_search", {"query": "weather in Bern"})

    text = result.output["results"][0]["snippet"]
    assert description in text
    assert snippet in text


def test_youcom_search_deduplicates_identical_description_and_snippet():
    text = "Current temperature: 21°C."

    assert realtime_youcom_search_tool._snippet({"description": text, "snippets": [text]}) == text
