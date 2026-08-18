import importlib
import sys
import types
from contextlib import ExitStack
from unittest import mock

import pytest
from nltk import sent_tokenize

from speech_to_speech.LLM.utils import (
    WHISPER_LANGUAGE_TO_LLM_LANGUAGE,
    remove_markdown,
    remove_unspeechable,
    resolve_auto_language,
    sent_tokenize_preserving_markdown_code,
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


def test_remove_markdown_strips_bold_and_italic() -> None:
    assert remove_markdown("**bold** and *italic* text") == "bold and italic text"
    assert remove_markdown("__bold__ and _italic_ text") == "bold and italic text"


def test_remove_markdown_keeps_snake_case_identifiers() -> None:
    assert remove_markdown("function_call_output") == "function_call_output"


def test_remove_markdown_strips_bullets_without_eating_following_lines() -> None:
    assert remove_markdown("* first\n* second\n* third") == "first\nsecond\nthird"
    assert remove_markdown("- one\n- two") == "one\ntwo"


def test_remove_markdown_strips_headings() -> None:
    assert remove_markdown("# Title\nsome text") == "Title\nsome text"
    assert remove_markdown("### Subheading") == "Subheading"
    assert remove_markdown("#include <stdio.h>") == "#include <stdio.h>"


def test_remove_markdown_does_not_eat_multiplication() -> None:
    assert remove_markdown("2 * 3 * 4") == "2 * 3 * 4"
    assert remove_markdown("2*3 = 6") == "2*3 = 6"
    assert remove_markdown("x*y") == "x*y"
    assert remove_markdown("5**2 = 25") == "5**2 = 25"


def test_remove_markdown_does_not_pair_independent_compact_operators() -> None:
    assert remove_markdown("2*3 + 4*5") == "2*3 + 4*5"
    assert remove_markdown("5**2 + 3**4") == "5**2 + 3**4"
    assert remove_markdown("x*y and a*b") == "x*y and a*b"
    assert remove_markdown("x*y*z") == "x*y*z"
    assert remove_markdown("x**y**z") == "x**y**z"
    assert remove_markdown("力*質量*時間") == "力*質量*時間"
    assert remove_markdown("скорость*время*путь") == "скорость*время*путь"
    assert remove_markdown("α*β*γ") == "α*β*γ"
    assert remove_markdown("a*β*c") == "a*β*c"


def test_remove_markdown_preserves_unmatched_delimiters_and_operators() -> None:
    assert remove_markdown("*args") == "*args"
    assert remove_markdown("**kwargs") == "**kwargs"
    assert remove_markdown("_private") == "_private"
    assert remove_markdown("file*.txt") == "file*.txt"
    assert remove_markdown("Price is $5* tax.") == "Price is $5* tax."
    assert remove_markdown("force*mass*time") == "force*mass*time"
    assert remove_markdown("`unclosed") == "`unclosed"
    assert remove_markdown("Do you mean snake case**?") == "Do you mean snake case**?"


def test_remove_markdown_strips_nested_bold_and_code() -> None:
    assert remove_markdown("**bold with `code`**") == "bold with code"


def test_remove_markdown_strips_inline_code() -> None:
    assert remove_markdown("`inline`") == "inline"


def test_remove_markdown_preserves_markdown_like_characters_inside_code() -> None:
    assert remove_markdown("Use `*args` and `**kwargs`.") == "Use *args and **kwargs."
    assert remove_markdown("Match `*.py` files.") == "Match *.py files."
    assert (
        remove_markdown("```python\n*args = values\n# keep_this\n**literal**\n```")
        == "*args = values\n# keep_this\n**literal**\n"
    )


def test_remove_markdown_strips_fenced_code_block_and_language_tag() -> None:
    assert remove_markdown("```python\nname = 'Alice'\n```") == "name = 'Alice'\n"
    assert remove_markdown("```\ncode\n```") == "code\n"
    assert remove_markdown("```code```") == "code"


def test_remove_markdown_is_streaming_safe_across_split_deltas() -> None:
    """remove_markdown must be applied to complete text, not per-delta: a
    delimiter pair split across two deltas (*ita / lic*) has nothing to match
    on its own, so callers accumulate first and strip once, as tested here."""
    deltas = ["*ita", "lic* is a word."]
    accumulated = "".join(deltas)
    assert remove_markdown(accumulated) == "italic is a word."


def test_sentence_tokenization_preserves_complete_emphasis_pairs() -> None:
    assert sent_tokenize_preserving_markdown_code("**Let me check.**", sent_tokenize) == ["**Let me check.**"]
