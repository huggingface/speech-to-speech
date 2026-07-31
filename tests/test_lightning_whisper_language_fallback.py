"""Regression tests for the whisper-mlx STT handler's auto-language fallback.

In auto mode the handler used to *discard the user's turn* when Whisper detected a language
outside its 12-entry SUPPORTED_LANGUAGES list and no supported language had been seen yet:

    if language_code not in SUPPORTED_LANGUAGES:
        if self.last_language in SUPPORTED_LANGUAGES:
            transcription_dict = self.model.transcribe(audio, language=self.last_language)
        else:
            transcription_dict = {"text": "", "language": "en"}   # <- transcription dropped

`setup()` also stored `--language auto` verbatim in `last_language`, and "auto" is not in
SUPPORTED_LANGUAGES, so the drop branch was the one taken on the first such utterance.

Whisper transcribes those languages fine; only the pipeline threw the result away. The
handler now reports the language actually transcribed and never drops the text, matching the
behaviour agreed for the transformers-Whisper handler.

`lightning_whisper_mlx` is a macOS-only optional dependency and `torch.mps.empty_cache()`
raises off Apple Silicon, so both are stubbed and these tests run anywhere.
"""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

from speech_to_speech.pipeline.messages import VADAudio


class FakeLightningWhisperMLX:
    """Stand-in for the real model: scripted results, records every transcribe() call."""

    def __init__(self, *args, **kwargs):
        self.calls: list[dict] = []
        # Default warmup result; tests overwrite `results` afterwards.
        self.results = [{"text": "", "language": "en"}]

    def transcribe(self, audio, language=None):
        self.calls.append({"language": language})
        index = min(len(self.calls) - 1, len(self.results) - 1)
        return self.results[index]


@pytest.fixture
def handler_module(monkeypatch):
    """Import the handler with its macOS-only dependency stubbed out."""
    fake_pkg = types.ModuleType("lightning_whisper_mlx")
    fake_pkg.LightningWhisperMLX = FakeLightningWhisperMLX  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lightning_whisper_mlx", fake_pkg)
    monkeypatch.delitem(sys.modules, "speech_to_speech.STT.lightning_whisper_mlx_handler", raising=False)

    module = importlib.import_module("speech_to_speech.STT.lightning_whisper_mlx_handler")
    # torch.mps.empty_cache() raises RuntimeError without an MPS backend.
    monkeypatch.setattr(module.torch.mps, "empty_cache", lambda: None)

    yield module

    # Don't leave a module built against the stub in sys.modules for other tests.
    sys.modules.pop("speech_to_speech.STT.lightning_whisper_mlx_handler", None)


def make_handler(handler_module, *, language, results, last_language=...):
    handler = object.__new__(handler_module.LightningWhisperSTTHandler)
    handler.device = "mps"
    handler.start_language = language
    handler.last_language = (language if language != "auto" else None) if last_language is ... else last_language
    handler.model = FakeLightningWhisperMLX()
    handler.model.results = results
    return handler


def vad_audio():
    return VADAudio(audio=np.zeros(16000, dtype=np.float32), turn_id="turn_1", turn_revision=0)


def run(handler):
    outputs = list(handler.process(vad_audio()))
    assert len(outputs) == 1
    return outputs[0]


# --- the dropped-turn bug -----------------------------------------------------------------


def test_unsupported_detected_language_keeps_the_transcription(handler_module):
    """The core bug: Russian speech came back as empty text and was silently dropped."""
    handler = make_handler(
        handler_module,
        language="auto",
        results=[{"text": "Privet, kak dela?", "language": "ru"}],
    )

    result = run(handler)

    assert result.text == "Privet, kak dela?"
    assert result.language_code == "ru-auto"
    # One transcribe() call: no destructive re-transcription.
    assert len(handler.model.calls) == 1


def test_unsupported_detected_language_is_not_retranscribed_in_last_language(handler_module):
    """With a previous supported language, the audio used to be re-transcribed as that."""
    handler = make_handler(
        handler_module,
        language="auto",
        results=[{"text": "Privet, kak dela?", "language": "ru"}],
        last_language="de",
    )

    result = run(handler)

    assert result.text == "Privet, kak dela?"
    assert result.language_code == "ru-auto"
    assert len(handler.model.calls) == 1
    # An unsupported code must not become the sticky fallback.
    assert handler.last_language == "de"


def test_supported_detected_language_becomes_the_sticky_fallback(handler_module):
    handler = make_handler(
        handler_module,
        language="auto",
        results=[{"text": "Ich heisse Max.", "language": "de"}],
    )

    assert run(handler).language_code == "de-auto"
    assert handler.last_language == "de"


@pytest.mark.parametrize("detected", ["ru", "ar", "sv", "tr"])
def test_no_language_ever_yields_empty_text(handler_module, detected):
    """No detected language may cause the user's turn to be discarded."""
    handler = make_handler(
        handler_module,
        language="auto",
        results=[{"text": "some real transcription", "language": detected}],
    )

    result = run(handler)

    assert result.text == "some real transcription"
    assert result.language_code == f"{detected}-auto"


# --- setup() must not store "auto" as a language code -------------------------------------


@pytest.mark.parametrize(
    ("language", "expected_last_language"),
    [("auto", None), ("de", "de"), (None, None)],
)
def test_setup_does_not_store_auto_as_last_language(handler_module, language, expected_last_language):
    handler = object.__new__(handler_module.LightningWhisperSTTHandler)
    handler.setup(model_name="distil-large-v3", language=language)

    assert handler.start_language == language
    assert handler.last_language == expected_last_language


def test_setup_strips_a_repo_prefix_from_the_model_name(handler_module):
    """Pre-existing behaviour, pinned so the setup change doesn't regress it."""
    handler = object.__new__(handler_module.LightningWhisperSTTHandler)
    handler.setup(model_name="mlx-community/distil-large-v3", language="en")

    assert handler.last_language == "en"


# --- forced language ----------------------------------------------------------------------


def test_forced_language_is_passed_through_and_reported(handler_module):
    handler = make_handler(
        handler_module,
        language="de",
        results=[{"text": "Ich heisse Max.", "language": "it"}],
    )

    result = run(handler)

    assert handler.model.calls == [{"language": "de"}]
    # No "-auto" suffix when the language was pinned.
    assert result.language_code == "de"
    assert result.text == "Ich heisse Max."


def test_language_none_is_treated_as_auto_detect_without_suffix(handler_module):
    """`language=None` means "don't force", but it is not the `auto` display mode."""
    handler = make_handler(
        handler_module,
        language=None,
        results=[{"text": "Hello.", "language": "en"}],
    )

    result = run(handler)

    assert handler.model.calls == [{"language": None}]
    assert result.language_code == "en"


# --- degenerate model output --------------------------------------------------------------


def test_missing_language_key_falls_back_without_crashing(handler_module):
    handler = make_handler(
        handler_module,
        language="auto",
        results=[{"text": "Hello."}],
        last_language="de",
    )

    result = run(handler)

    assert result.text == "Hello."
    assert result.language_code == "de-auto"


def test_missing_language_and_no_fallback_defaults_to_english(handler_module):
    handler = make_handler(
        handler_module,
        language="auto",
        results=[{"text": "Hello."}],
        last_language=None,
    )

    assert run(handler).language_code == "en-auto"


def test_missing_text_key_yields_empty_string_not_a_crash(handler_module):
    handler = make_handler(
        handler_module,
        language="auto",
        results=[{"language": "en"}],
    )

    assert run(handler).text == ""
