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


def _setup_handler(module_name: str, class_name: str, language: str):
    """Run a handler's real setup while replacing model loading and warmup."""
    cls = _load(module_name, class_name)
    handler = object.__new__(cls)

    if class_name == "WhisperSTTHandler":
        model = mock.Mock()
        model.to.return_value = model
        globals_patch = {
            "AutoProcessor": types.SimpleNamespace(from_pretrained=mock.Mock(return_value=object())),
            "AutoModelForSpeechSeq2Seq": types.SimpleNamespace(from_pretrained=mock.Mock(return_value=model)),
        }
        with mock.patch.dict(cls.setup.__globals__, globals_patch), mock.patch.object(cls, "warmup"):
            handler.setup(device="cpu", torch_dtype="float32", language=language, gen_kwargs={})
        return handler

    if class_name == "LightningWhisperSTTHandler":
        globals_patch = {"LightningWhisperMLX": mock.Mock(return_value=object())}
        with mock.patch.dict(cls.setup.__globals__, globals_patch), mock.patch.object(cls, "warmup"):
            handler.setup(language=language)
        return handler

    if class_name == "MLXAudioWhisperSTTHandler":
        package = types.ModuleType("mlx_audio")
        package.__path__ = []
        stt_package = types.ModuleType("mlx_audio.stt")
        stt_package.__path__ = []
        generate = types.ModuleType("mlx_audio.stt.generate")
        model = types.SimpleNamespace(_processor=object())
        generate.load_model = mock.Mock(return_value=model)
        modules = {
            "mlx_audio": package,
            "mlx_audio.stt": stt_package,
            "mlx_audio.stt.generate": generate,
        }
        with mock.patch.dict(sys.modules, modules), mock.patch.object(cls, "warmup"):
            handler.setup(language=language, gen_kwargs={})
        return handler

    raise AssertionError(f"Unhandled test handler: {class_name}")


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


# --- production setup treats every spelling as detection ----------------------------------


@pytest.mark.parametrize("spelling", AUTO_SPELLINGS)
@pytest.mark.parametrize(("module_name", "class_name"), _HANDLERS)
def test_setup_treats_every_auto_spelling_as_detection(module_name, class_name, spelling):
    handler = _setup_handler(module_name, class_name, spelling)

    assert handler.start_language == "auto"
    handler.last_language = "de"

    handler.on_session_end()

    assert handler.last_language is None
    if class_name == "WhisperSTTHandler":
        assert "language" not in handler.gen_kwargs
        assert handler._forced_language() is None
    elif class_name == "MLXAudioWhisperSTTHandler":
        assert handler._forced_language() is None


@pytest.mark.parametrize(("module_name", "class_name"), _HANDLERS)
def test_setup_preserves_a_real_language(module_name, class_name):
    handler = _setup_handler(module_name, class_name, "de")

    assert handler.start_language == "de"
    handler.last_language = "fr"
    handler.on_session_end()
    assert handler.last_language == "de"

    if class_name == "WhisperSTTHandler":
        assert handler.gen_kwargs["language"] == "de"
        assert handler._forced_language() == "de"
    elif class_name == "MLXAudioWhisperSTTHandler":
        assert handler._forced_language() == "de"
