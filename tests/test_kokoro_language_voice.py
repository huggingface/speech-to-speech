from __future__ import annotations

import sys
from collections.abc import Iterator
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from speech_to_speech.TTS import kokoro_handler
from speech_to_speech.TTS.kokoro_handler import KokoroTTSHandler


@pytest.fixture
def kokoro_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    calls: dict[str, list] = {"pipelines": [], "voices": []}

    class FakePipeline:
        def __init__(self, *, lang_code: str, device: str) -> None:
            calls["pipelines"].append(lang_code)

        def __call__(self, _text: str, *, voice: str, speed: float) -> Iterator[tuple[str, str, np.ndarray]]:
            calls["voices"].append(voice)
            return iter(())

    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KPipeline=FakePipeline))
    monkeypatch.setattr(kokoro_handler, "platform", "linux")
    monkeypatch.setattr(KokoroTTSHandler, "warmup", lambda self: None)
    monkeypatch.setattr(kokoro_handler.console, "print", lambda *_args, **_kwargs: None)
    return calls


def _handler(lang_code: str, voice: str) -> KokoroTTSHandler:
    handler = KokoroTTSHandler.__new__(KokoroTTSHandler)
    handler.setup(Event(), device="cpu", lang_code=lang_code, voice=voice)
    return handler


def test_english_reply_keeps_configured_american_voice(kokoro_calls: dict[str, list]) -> None:
    handler = _handler("a", "af_heart")

    list(handler._process_kokoro("Hello there.", language_code="en"))

    assert (handler.lang_code, handler.voice) == ("a", "af_heart")
    assert kokoro_calls["pipelines"] == ["a"]
    assert kokoro_calls["voices"] == ["af_heart"]


def test_returning_to_setup_language_restores_configured_voice(kokoro_calls: dict[str, list]) -> None:
    handler = _handler("b", "bf_emma")

    list(handler._process_kokoro("Hola.", language_code="es"))
    list(handler._process_kokoro("Hello again.", language_code="en"))

    assert (handler.lang_code, handler.voice) == ("b", "bf_emma")
    assert kokoro_calls["voices"] == ["ef_dora", "bf_emma"]
