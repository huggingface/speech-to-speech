import importlib
import unicodedata

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


@pytest.mark.parametrize(
    "text",
    [
        "你好。今天天气怎么样？",  # zh — ideographic full stop / fullwidth question mark
        "こんにちは。元気ですか？",  # ja
        "안녕하세요。「반갑습니다」",  # ko — corner brackets
        "नमस्ते। आप कैसे हैं?",  # hi — devanagari danda
        "¿Cómo estás? ¡Bien!",  # es — inverted marks
        "Er sagte: «Guten Tag»",  # de/fr — guillemets
        "مرحبا، كيف حالك؟",  # ar — arabic comma / question mark
    ],
)
def test_remove_unspeechable_keeps_non_ascii_sentence_punctuation(text: str) -> None:
    """Sentence punctuation outside ASCII must survive: it reaches the TTS, the
    client transcript and the chat history, so dropping it flattens prosody and
    corrupts the conversation for every non-English language we support."""
    assert remove_unspeechable(text) == text


def test_remove_unspeechable_keeps_combining_marks() -> None:
    """``\\w`` does not match combining marks, so they were stripped from the
    middle of words: Devanagari matras/virama, Arabic and Hebrew diacritics, and
    NFD-decomposed Latin accents."""
    assert remove_unspeechable("नमस्ते") == "नमस्ते"
    assert remove_unspeechable(unicodedata.normalize("NFD", "café")) == unicodedata.normalize("NFD", "café")


def test_remove_unspeechable_drops_symbols_but_keeps_punctuation() -> None:
    assert remove_unspeechable("100 % ✅ done。") == "100 %  done。"


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
]

# These have no optional top-level dependency, so a skip here means something is wrong
# rather than merely uninstalled.
_ALWAYS_IMPORTABLE = {
    "speech_to_speech.STT.parakeet_tdt_handler",
    "speech_to_speech.STT.whisper_stt_handler",
    "speech_to_speech.STT.mlx_audio_whisper_handler",
}


def _supported_languages(module_name):
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    return list(module.SUPPORTED_LANGUAGES)


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
        assert name.isalpha(), f"{code} -> {name!r}"


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
