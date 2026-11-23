"""Parakeet reports detected languages with -auto and ignores configured preferences."""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

from speech_to_speech.pipeline.messages import VADAudio


class FakeNanoParakeetModel:
    """Stand-in for nano_parakeet's model; records what audio it was called with."""

    def __init__(self, text="Hello there"):
        self.text = text
        self.calls: list = []

    def transcribe(self, audio):
        self.calls.append(audio)
        return self.text


@pytest.fixture
def handler_module(monkeypatch):
    """Import the handler with its optional dependency stubbed out."""
    fake_pkg = types.ModuleType("nano_parakeet")
    fake_pkg.from_pretrained = lambda model_name, device: FakeNanoParakeetModel()
    monkeypatch.setitem(sys.modules, "nano_parakeet", fake_pkg)
    monkeypatch.delitem(sys.modules, "speech_to_speech.STT.parakeet_tdt_handler", raising=False)

    module = importlib.import_module("speech_to_speech.STT.parakeet_tdt_handler")
    monkeypatch.setattr(module, "warm_language_detector", lambda candidates: None)

    yield module

    sys.modules.pop("speech_to_speech.STT.parakeet_tdt_handler", None)


def make_handler(handler_module, *, language="en", model_text="Hello there"):
    handler = object.__new__(handler_module.ParakeetTDTSTTHandler)
    handler.setup(
        model_name="fake-model",
        device="cpu",
        language=language,
        gen_kwargs={},
    )
    handler.model.text = model_text
    return handler


def test_configured_language_is_not_reported_when_detection_fails(handler_module, monkeypatch):
    handler = make_handler(handler_module, language="fr")
    monkeypatch.setattr(handler, "_detect_language_from_text", lambda text: None)

    outputs = list(handler.process(VADAudio(audio=np.zeros(16000, dtype=np.float32), turn_id="t1", turn_revision=0)))

    assert outputs[0].language_code is None


def test_detected_language_is_reported_with_the_auto_suffix(handler_module, monkeypatch):
    handler = make_handler(handler_module, language=None)
    monkeypatch.setattr(handler, "_detect_language_from_text", lambda text: "de")

    outputs = list(handler.process(VADAudio(audio=np.zeros(16000, dtype=np.float32), turn_id="t1", turn_revision=0)))

    assert outputs[0].language_code == "de-auto"


def test_configured_language_does_not_override_detection(handler_module, monkeypatch):
    """Parakeet cannot constrain decoding to the configured language."""
    handler = make_handler(handler_module, language="fr")
    monkeypatch.setattr(handler, "_detect_language_from_text", lambda text: "de")

    outputs = list(handler.process(VADAudio(audio=np.zeros(16000, dtype=np.float32), turn_id="t1", turn_revision=0)))

    assert outputs[0].language_code == "de-auto"


@pytest.mark.parametrize("value", ["auto", "AUTO", " auto ", "", "  ", "none", "NONE", "null", "NULL", None, 123])
def test_auto_sentinels_are_normalized_to_none(handler_module, value):
    handler = make_handler(handler_module, language=value)

    assert handler.start_language is None


def test_detection_failure_does_not_report_a_previous_turn_language(handler_module, monkeypatch):
    handler = make_handler(handler_module, language=None)
    handler.last_language = "es"
    monkeypatch.setattr(handler, "_detect_language_from_text", lambda text: None)

    outputs = list(handler.process(VADAudio(audio=np.zeros(16000, dtype=np.float32), turn_id="t1", turn_revision=0)))

    assert outputs[0].language_code is None
    assert handler.last_language == "es"


def test_last_language_updates_from_tagged_detected_code(handler_module, monkeypatch):
    """A detected 'fr-auto' result must still update last_language to bare 'fr'."""
    handler = make_handler(handler_module, language=None)
    monkeypatch.setattr(handler, "_detect_language_from_text", lambda text: "fr")

    outputs = list(handler.process(VADAudio(audio=np.zeros(16000, dtype=np.float32), turn_id="t1", turn_revision=0)))

    assert outputs[0].language_code == "fr-auto"
    assert handler.last_language == "fr"


def test_configured_language_whitespace_is_stripped(handler_module):
    handler = make_handler(handler_module, language="  fr  ")

    assert handler.start_language == "fr"
