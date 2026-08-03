"""Handlers must not configure the root logger or write to stdout via print().

Library modules that call ``logging.basicConfig`` at import time reconfigure the
host application's logging (level, format, handlers). Secondary STT/TTS handlers
historically did this, and used ``print`` for setup/cleanup messages. These tests
pin the hygiene contract used by the rest of the package: module loggers only,
no import-time root configuration.
"""

from __future__ import annotations

import ast
import importlib
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"

_HANDLERS_WITH_FORMER_BASIC_CONFIG = [
    "speech_to_speech/STT/paraformer_handler.py",
    "speech_to_speech/TTS/facebookmms_handler.py",
    "speech_to_speech/TTS/chatTTS_handler.py",
]

_HANDLERS_WITH_FORMER_PRINT = [
    "speech_to_speech/STT/paraformer_handler.py",
    "speech_to_speech/STT/faster_whisper_handler.py",
]


def _module_source(relative_path: str) -> str:
    return (_SRC_ROOT / relative_path).read_text(encoding="utf-8")


def _calls_named(tree: ast.AST, qualname: tuple[str, ...]) -> list[ast.Call]:
    """Return Call nodes whose function is exactly ``a.b.c`` matching *qualname*."""
    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        parts: list[str] = []
        func = node.func
        while isinstance(func, ast.Attribute):
            parts.append(func.attr)
            func = func.value
        if isinstance(func, ast.Name):
            parts.append(func.id)
            if tuple(reversed(parts)) == qualname:
                found.append(node)
    return found


def _top_level_print_calls(tree: ast.AST) -> list[ast.Call]:
    """``print(...)`` calls anywhere in the module (setup/cleanup included)."""
    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            found.append(node)
    return found


@pytest.mark.parametrize("relative_path", _HANDLERS_WITH_FORMER_BASIC_CONFIG)
def test_handler_source_has_no_logging_basic_config(relative_path: str) -> None:
    tree = ast.parse(_module_source(relative_path))
    assert _calls_named(tree, ("logging", "basicConfig")) == [], (
        f"{relative_path} must not call logging.basicConfig; library imports must not "
        f"reconfigure the host application's root logger"
    )


@pytest.mark.parametrize("relative_path", _HANDLERS_WITH_FORMER_PRINT)
def test_handler_source_has_no_print_calls(relative_path: str) -> None:
    tree = ast.parse(_module_source(relative_path))
    assert _top_level_print_calls(tree) == [], (
        f"{relative_path} must use the module logger instead of print() for setup/cleanup"
    )


def test_paraformer_import_does_not_call_basic_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing Paraformer must not touch the root logger configuration."""
    sentinel = MagicMock()
    monkeypatch.setattr(logging, "basicConfig", sentinel)

    module_name = "speech_to_speech.STT.paraformer_handler"
    sys.modules.pop(module_name, None)
    importlib.import_module(module_name)

    sentinel.assert_not_called()


def test_facebookmms_import_does_not_call_basic_config(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = MagicMock()
    monkeypatch.setattr(logging, "basicConfig", sentinel)

    module_name = "speech_to_speech.TTS.facebookmms_handler"
    sys.modules.pop(module_name, None)
    importlib.import_module(module_name)

    sentinel.assert_not_called()


def test_chattts_import_does_not_call_basic_config(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = MagicMock()
    monkeypatch.setattr(logging, "basicConfig", sentinel)
    monkeypatch.setitem(sys.modules, "ChatTTS", MagicMock())

    module_name = "speech_to_speech.TTS.chatTTS_handler"
    sys.modules.pop(module_name, None)
    importlib.import_module(module_name)

    sentinel.assert_not_called()


def test_paraformer_setup_logs_model_name_instead_of_printing(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from speech_to_speech.STT.paraformer_handler import ParaformerSTTHandler

    fake_model = MagicMock()
    fake_model.generate.return_value = [{"text": "warmup"}]
    fake_auto_model = MagicMock(return_value=fake_model)
    fake_funasr = MagicMock()
    fake_funasr.AutoModel = fake_auto_model
    monkeypatch.setitem(sys.modules, "funasr", fake_funasr)

    handler = object.__new__(ParaformerSTTHandler)
    with caplog.at_level(logging.INFO, logger="speech_to_speech.STT.paraformer_handler"):
        handler.setup(model_name="paraformer-zh", device="cpu")

    assert "Loading Paraformer STT model: paraformer-zh" in caplog.text
    fake_auto_model.assert_called_once_with(model="paraformer-zh", device="cpu")


def test_faster_whisper_cleanup_logs_instead_of_printing(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", MagicMock())
    sys.modules.pop("speech_to_speech.STT.faster_whisper_handler", None)
    from speech_to_speech.STT.faster_whisper_handler import FasterWhisperSTTHandler

    handler = object.__new__(FasterWhisperSTTHandler)
    handler.model = object()

    with caplog.at_level(logging.INFO, logger="speech_to_speech.STT.faster_whisper_handler"):
        handler.cleanup()

    assert "Stopping FasterWhisperSTTHandler" in caplog.text
    assert not hasattr(handler, "model")
