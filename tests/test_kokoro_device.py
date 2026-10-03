from __future__ import annotations

import sys
from collections.abc import Iterator
from threading import Event
from types import SimpleNamespace

import pytest
import torch

from speech_to_speech.TTS import kokoro_handler
from speech_to_speech.TTS.kokoro_handler import KokoroTTSHandler
from speech_to_speech.utils import utils


@pytest.mark.parametrize("model_name", [None, "acme/custom-kokoro"])
def test_auto_npu_reaches_kokoro_initial_and_rebuilt_pipelines(monkeypatch: pytest.MonkeyPatch, model_name) -> None:
    pipeline_devices: list[tuple[str, str, str]] = []

    class FakePipeline:
        def __init__(self, *, lang_code: str, repo_id: str, device: str) -> None:
            pipeline_devices.append((lang_code, repo_id, device))

        def __call__(self, *_args: object, **_kwargs: object) -> Iterator[tuple[object, object, object]]:
            return iter(())

    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KPipeline=FakePipeline))
    monkeypatch.setattr(kokoro_handler, "platform", "linux")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(utils, "is_npu_available", lambda: True)
    monkeypatch.setattr(KokoroTTSHandler, "warmup", lambda self: None)
    monkeypatch.setattr(kokoro_handler.console, "print", lambda *_args, **_kwargs: None)

    handler = KokoroTTSHandler.__new__(KokoroTTSHandler)
    handler.setup(Event(), device="auto", model_name=model_name)
    list(handler._process_kokoro("Bonjour", language_code="fr"))
    handler.on_session_end()

    assert handler.device == "npu"
    expected_model = model_name or "hexgrad/Kokoro-82M"
    assert pipeline_devices == [
        ("b", expected_model, "npu"),
        ("f", expected_model, "npu"),
        ("b", expected_model, "npu"),
    ]
