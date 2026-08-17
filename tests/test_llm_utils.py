import importlib
import sys
import types
from contextlib import ExitStack
from unittest import mock

import pytest

from speech_to_speech.LLM.utils import (
    WHISPER_LANGUAGE_TO_LLM_LANGUAGE,
    remove_unspeechable,
    resolve_auto_language,
)


def test_remove_unspeechable_normalizes_smart_apostrophes() -> None:
    assert remove_unspeechable("I’ll reply if here’s the plan.") == "I'll reply if here's the plan."


def test_remove_unspeechable_keeps_text_and_drops_emoji() -> None:
    assert remove_unspeechable("Hello 👋 lobster 🦞") == "Hello  lobster "


def test_remove_unspeechable_keeps_chinese_punctuation() -> None:
    text = "你好，今天怎么样？很好！停顿；说明：一、二。"
    assert remove_unspeechable(text) == text


# --- language name coverage ---------------------------------------------------------------
#
# A language code with no entry in WHISPER_LANGUAGE_TO_LLM_LANGUAGE resolves to a `None`
# language name, and both LLM backends gate the prompt on it:
#
#     if lang_name and self.enable_lang_prompt:
#         active_chat.add_item(make_user_message(f"Please reply to my message in {lang_name}."))
#
# so `--enable_lang_prompt` silently emits nothing for that language. Parakeet TDT is the
# default STT and reports 25 languages, of which only 8 overlapped the original 12-entry map.

# Modules that declare a SUPPORTED_LANGUAGES list of codes they can report.
_STT_HANDLER_MODULES = [
    "speech_to_speech.STT.parakeet_tdt_handler",
    "speech_to_speech.STT.whisper_stt_handler",
    "speech_to_speech.STT.mlx_audio_whisper_handler",
    "speech_to_speech.STT.lightning_whisper_mlx_handler",
    "speech_to_speech.STT.faster_whisper_handler",
]

# These have no optional top-level dependency, so a skip here means something is wrong
# rather than merely uninstalled.
_ALWAYS_IMPORTABLE = {
    "speech_to_speech.STT.parakeet_tdt_handler",
    "speech_to_speech.STT.whisper_stt_handler",
    "speech_to_speech.STT.mlx_audio_whisper_handler",
    # Importable via the stub above, so a skip here would mean the stub stopped working.
    "speech_to_speech.STT.faster_whisper_handler",
}


# Optional third-party modules stubbed purely so a handler's SUPPORTED_LANGUAGES stays
# checkable on CI. The list is a plain literal, so no real dependency is needed to read it,
# and without this the faster-whisper check would silently skip everywhere.
_STUBBABLE_DEPENDENCIES = {
    "speech_to_speech.STT.faster_whisper_handler": ("faster_whisper", "WhisperModel"),
}


def _supported_languages(module_name):
    stub = _STUBBABLE_DEPENDENCIES.get(module_name)
    with ExitStack() as stack:
        if stub is not None and stub[0] not in sys.modules:
            package, attribute = stub
            module = types.ModuleType(package)
            setattr(module, attribute, object)
            stack.enter_context(mock.patch.dict(sys.modules, {package: module}))
            stack.callback(sys.modules.pop, module_name, None)
        try:
            handler_module = importlib.import_module(module_name)
        except ImportError:
            return None
        return list(handler_module.SUPPORTED_LANGUAGES)


@pytest.mark.parametrize("module_name", _STT_HANDLER_MODULES)
def test_every_stt_language_has_an_llm_language_name(module_name):
    """Any language a bundled STT backend can report must be nameable for the prompt."""
    languages = _supported_languages(module_name)
    if languages is None:
        if module_name in _ALWAYS_IMPORTABLE:
            pytest.fail(f"{module_name} should be importable without optional extras")
        pytest.skip(f"{module_name} requires an optional dependency")

    missing = sorted(code for code in languages if code not in WHISPER_LANGUAGE_TO_LLM_LANGUAGE)
    assert missing == [], (
        f"{module_name} can report {missing}, which have no entry in "
        f"WHISPER_LANGUAGE_TO_LLM_LANGUAGE, so --enable_lang_prompt would emit no "
        f"instruction for them"
    )


def test_parakeet_default_stt_is_fully_covered():
    """Explicit guard for the default backend, independent of the parametrized sweep."""
    parakeet = importlib.import_module("speech_to_speech.STT.parakeet_tdt_handler")

    assert len(parakeet.SUPPORTED_LANGUAGES) == 25
    assert set(parakeet.SUPPORTED_LANGUAGES) <= set(WHISPER_LANGUAGE_TO_LLM_LANGUAGE)


def test_language_names_are_lowercase_and_non_empty():
    """The name is interpolated mid-sentence, so it must read as lowercase prose."""
    for code, name in WHISPER_LANGUAGE_TO_LLM_LANGUAGE.items():
        assert name and name == name.lower(), f"{code} -> {name!r}"
        # Multi-word names such as "haitian creole" are fine; punctuation is not.
        assert name.replace(" ", "").isalpha(), f"{code} -> {name!r}"
        assert name == name.strip(), f"{code} -> {name!r}"


def test_whisper_language_coverage_is_complete():
    """Whisper reports 100 languages; a missing name silently drops the prompt."""
    assert len(WHISPER_LANGUAGE_TO_LLM_LANGUAGE) == 100
    # Spot-check languages that only Whisper-family backends can report.
    for code in ("ar", "tr", "he", "th", "yue", "haw"):
        assert code in WHISPER_LANGUAGE_TO_LLM_LANGUAGE


# --- resolve_auto_language ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("sv", ("sv", "swedish")),
        ("sv-auto", ("sv", "swedish")),
        ("ru-auto", ("ru", "russian")),
        ("no-auto", ("no", "norwegian")),
        ("lt", ("lt", "lithuanian")),
        ("en-auto", ("en", "english")),
    ],
)
def test_resolve_auto_language_names_parakeet_languages(code, expected):
    assert resolve_auto_language(code) == expected


@pytest.mark.parametrize("code", [None, ""])
def test_resolve_auto_language_passes_through_empty_codes(code):
    assert resolve_auto_language(code) == (code, None)


def test_resolve_auto_language_returns_no_name_for_unknown_code():
    """Unknown codes still round-trip the code, they just cannot be named."""
    assert resolve_auto_language("xx-auto") == ("xx", None)
