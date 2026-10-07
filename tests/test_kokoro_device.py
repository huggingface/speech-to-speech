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


def test_auto_npu_reaches_kokoro_initial_and_rebuilt_pipelines(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline_devices: list[tuple[str, str]] = []

    class FakePipeline:
        def __init__(self, *, lang_code: str, device: str) -> None:
            pipeline_devices.append((lang_code, device))

        def __call__(self, *_args: object, **_kwargs: object) -> Iterator[tuple[object, object, object]]:
            return iter(())

    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KPipeline=FakePipeline))
    monkeypatch.setattr(kokoro_handler, "platform", "linux")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(utils, "is_npu_available", lambda: True)
    monkeypatch.setattr(KokoroTTSHandler, "warmup", lambda self: None)
    monkeypatch.setattr(kokoro_handler.console, "print", lambda *_args, **_kwargs: None)

    handler = KokoroTTSHandler.__new__(KokoroTTSHandler)
    handler.setup(Event(), device="auto")
    list(handler._process_kokoro("Bonjour", language_code="fr"))
    handler.on_session_end()

    assert handler.device == "npu"
    assert pipeline_devices == [("b", "npu"), ("f", "npu"), ("b", "npu")]
