import pytest
from examples import realtime_web_search_tool


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

    async def post(self, url, **kwargs):
        self.captured.update(url=url, **kwargs)
        return self.response


@pytest.mark.asyncio
async def test_serper_search_returns_answer_and_organic_results(monkeypatch):
    captured = {}
    response = FakeResponse(
        200,
        {
            "answerBox": {"answer": "21°C"},
            "organic": [
                {"title": "Forecast", "snippet": "Sunny", "link": "https://example.com/weather"},
            ],
        },
    )
    monkeypatch.setattr(realtime_web_search_tool, "SERPER_API_KEY", "test-key")
    monkeypatch.setattr(
        realtime_web_search_tool.httpx,
        "AsyncClient",
        lambda **kwargs: FakeAsyncClient(response, captured, **kwargs),
    )

    result = await realtime_web_search_tool.execute_tool("web_search", {"query": "weather in Bern"})

    assert result.output["answer"] == "21°C"
    assert result.output["results"] == [
        {"title": "Forecast", "snippet": "Sunny", "url": "https://example.com/weather"}
    ]
    assert captured == {
        "url": realtime_web_search_tool.SERPER_URL,
        "headers": {"X-API-KEY": "test-key", "Content-Type": "application/json"},
        "json": {"q": "weather in Bern", "num": realtime_web_search_tool.MAX_RESULTS},
    }


@pytest.mark.asyncio
async def test_serper_search_reports_missing_api_key(monkeypatch):
    monkeypatch.setattr(realtime_web_search_tool, "SERPER_API_KEY", "")

    result = await realtime_web_search_tool.execute_tool("web_search", {"query": "latest news"})

    assert result.output == {
        "query": "latest news",
        "error": "SERPER_API_KEY is not set. Get a free API key at https://serper.dev/.",
    }
