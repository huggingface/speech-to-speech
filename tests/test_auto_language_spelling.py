"""``--language AUTO`` must behave like ``--language auto``.

Handlers compare against the literal ``"auto"`` to decide whether the user asked for
detection. Qwen3-ASR and faster-whisper normalize with ``.strip().lower()`` first; the
others did not, so a different spelling was treated as a language *name* and forwarded to
the model. On the transformers Whisper backend that is a hard failure on the first turn:

    ValueError: Unsupported language: auto. Language should be one of: ['english', ...]

It also defeated the session reset added in #556, which restores ``start_language`` unless
it is the sentinel.
"""

from __future__ import annotations

import importlib
import sys
import types
from contextlib import ExitStack
from unittest import mock

import pytest

from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

# Spellings a user can reasonably type for the same intent.
AUTO_SPELLINGS = ["auto", "AUTO", "Auto", " auto ", "\tAuto\n"]

_HANDLERS = [
    ("speech_to_speech.STT.whisper_stt_handler", "WhisperSTTHandler"),
    ("speech_to_speech.STT.mlx_audio_whisper_handler", "MLXAudioWhisperSTTHandler"),
    ("speech_to_speech.STT.lightning_whisper_mlx_handler", "LightningWhisperSTTHandler"),
]

_STUBS = {
    "speech_to_speech.STT.lightning_whisper_mlx_handler": ("lightning_whisper_mlx", "LightningWhisperMLX"),
}


def _load(module_name: str, class_name: str):
    stub = _STUBS.get(module_name)
    with ExitStack() as stack:
        if stub is not None and stub[0] not in sys.modules:
            package, attribute = stub
            module = types.ModuleType(package)
            setattr(module, attribute, object)
            stack.enter_context(mock.patch.dict(sys.modules, {package: module}))
            stack.callback(sys.modules.pop, module_name, None)
        return getattr(importlib.import_module(module_name), class_name)


# --- the canonicalizer --------------------------------------------------------------------


@pytest.mark.parametrize("spelling", AUTO_SPELLINGS)
def test_auto_spellings_canonicalize(spelling):
    assert BaseSTTHandler.canonical_language(spelling) == "auto"


@pytest.mark.parametrize("value", ["de", "en", "yue", "fil"])
def test_real_codes_pass_through_untouched(value):
    """Only the sentinel is normalized; the model validates everything else."""
    assert BaseSTTHandler.canonical_language(value) == value


@pytest.mark.parametrize("value", [None, "", "  "])
def test_empty_values_pass_through(value):
    assert BaseSTTHandler.canonical_language(value) == value


def test_non_strings_pass_through():
    sentinel = object()
    assert BaseSTTHandler.canonical_language(sentinel) is sentinel


def test_a_code_merely_containing_auto_is_not_the_sentinel():
    assert BaseSTTHandler.canonical_language("auto-detect") == "auto-detect"


# --- handlers treat every spelling as detection -------------------------------------------


@pytest.mark.parametrize("spelling", AUTO_SPELLINGS)
@pytest.mark.parametrize(("module_name", "class_name"), _HANDLERS)
def test_setup_records_the_sentinel_canonically(module_name, class_name, spelling, monkeypatch):
    """`start_language` drives every downstream `== "auto"` comparison."""
    cls = _load(module_name, class_name)
    handler = object.__new__(cls)
    handler.start_language = cls.canonical_language(spelling)

    assert handler.start_language == "auto"


@pytest.mark.parametrize("spelling", AUTO_SPELLINGS)
@pytest.mark.parametrize(("module_name", "class_name"), _HANDLERS)
def test_session_reset_clears_every_auto_spelling(module_name, class_name, spelling):
    """#556 restores `start_language` unless it is the sentinel; a stray spelling would be
    restored as if it were a language code."""
    cls = _load(module_name, class_name)
    handler = object.__new__(cls)
    handler.start_language = cls.canonical_language(spelling)
    handler.last_language = "de"

    handler.on_session_end()

    assert handler.last_language is None


@pytest.mark.parametrize("spelling", ["AUTO", " auto "])
def test_whisper_does_not_force_a_bogus_language(spelling):
    """The crash path: a non-canonical sentinel reached `generate(language=...)`."""
    cls = _load("speech_to_speech.STT.whisper_stt_handler", "WhisperSTTHandler")
    handler = object.__new__(cls)
    language = cls.canonical_language(spelling)
    handler.start_language = language
    handler.last_language = language if language != "auto" else None
    handler.gen_kwargs = {}
    if handler.last_language is not None:
        handler.gen_kwargs["language"] = handler.last_language

    assert "language" not in handler.gen_kwargs
    assert handler._forced_language() is None


def test_whisper_still_forces_a_real_language():
    cls = _load("speech_to_speech.STT.whisper_stt_handler", "WhisperSTTHandler")
    handler = object.__new__(cls)
    language = cls.canonical_language("de")
    handler.start_language = language
    handler.last_language = language
    handler.gen_kwargs = {"language": language}

    assert handler._forced_language() == "de"
