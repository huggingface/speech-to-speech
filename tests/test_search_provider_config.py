import runpy
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", ["demo/server.py", "examples/realtime_web_search_tool.py"])
@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "https://google.serper.dev/search"),
        ("", "https://google.serper.dev/search"),
        (" \t ", "https://google.serper.dev/search"),
        ("https://api.litescrape.com/search", "https://api.litescrape.com/search"),
        (" https://api.litescrape.com/search \t", "https://api.litescrape.com/search"),
    ],
)
def test_search_provider_url(monkeypatch, path, value, expected):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "demo"))
    # Keep demo imports in direct mode: no load balancer or OAuth setup.
    monkeypatch.setenv("SPEECH_TO_SPEECH_URL", "ws://localhost:8765/v1/realtime")
    if value is None:
        monkeypatch.delenv("SERPER_URL", raising=False)
    else:
        monkeypatch.setenv("SERPER_URL", value)

    namespace = runpy.run_path(str(REPO_ROOT / path))

    assert namespace["SERPER_URL"] == expected
