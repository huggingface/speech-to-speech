"""Per-route language capability, the usable intersection, and honest rejection.

A conversation language is only usable if the STT backend can transcribe it *and* the TTS
backend can speak it. Those sets differ sharply, so a selection one end accepts can still be
silently wrong at the other: the default STT reports 25 languages while Kokoro can voice 8,
leaving 5 usable end to end.

The distinction that matters throughout is *undeclared* versus *unsupported*. A backend that
declares nothing must never be reported as supporting a language, and must never be used to
reject one either.
"""

from __future__ import annotations

import logging
import pathlib

import pytest

from speech_to_speech.backend_registry import STT_BACKENDS, TTS_BACKENDS
from speech_to_speech.pipeline.language_support import (
    backend_languages,
    check_language,
    declared_languages,
    log_usable_languages,
    usable_languages,
)

# --- capability comes from the backend's own constant --------------------------------------


def test_declared_languages_reads_the_backends_own_constant():
    from speech_to_speech.STT.parakeet_tdt_handler import SUPPORTED_LANGUAGES

    assert backend_languages("stt", "parakeet-tdt") == frozenset(SUPPORTED_LANGUAGES)


def test_undeclared_backend_reports_unknown_not_empty():
    """`qwen3` TTS takes a language but publishes no list; empty would mean "supports nothing"."""
    assert backend_languages("tts", "qwen3") is None


def test_unknown_backend_name_is_unknown():
    assert backend_languages("stt", "does-not-exist") is None
    assert declared_languages(None) is None


def test_every_declared_source_points_at_a_real_module():
    """A typo in `languages_source` degrades a backend to "unknown" silently.

    Checked against the source tree rather than by importing: a backend behind an
    uninstalled extra is legitimately unknown at runtime, so import success cannot
    distinguish a missing extra from a wrong path.
    """
    source_root = pathlib.Path(__file__).resolve().parents[1] / "src"
    missing = []
    for registry in (STT_BACKENDS, TTS_BACKENDS):
        for name, spec in registry.items():
            source = spec.capabilities.languages_source
            if source is None:
                continue
            module_name, attribute = source
            module_path = source_root / (module_name.replace(".", "/") + ".py")
            if not module_path.exists():
                missing.append(f"{name}: no module {module_name}")
            elif f"{attribute} " not in module_path.read_text(encoding="utf-8"):
                missing.append(f"{name}: {module_name} has no {attribute}")
    assert missing == []


def test_installed_backends_resolve_to_a_non_empty_set():
    """Whatever does import must yield real codes, not an empty or malformed set."""
    for registry in (STT_BACKENDS, TTS_BACKENDS):
        for name, spec in registry.items():
            if spec.capabilities.languages_source is None:
                continue
            languages = declared_languages(spec)
            if languages is None:
                continue  # optional extra not installed: unknown, which is allowed
            assert languages, f"{name} declares an empty language set"
            assert all(isinstance(code, str) and code for code in languages), name


# --- Kokoro's routing table overstates capability ------------------------------------------


def test_kokoro_declares_voices_not_routes():
    """WHISPER_LANGUAGE_TO_KOKORO_LANG maps de/nl/pl/ru/uk to a British English voice, so the
    map answers "where do I route this?", not "can this be spoken?"."""
    from speech_to_speech.TTS.kokoro_handler import WHISPER_LANGUAGE_TO_KOKORO_LANG

    declared = backend_languages("tts", "kokoro")

    assert "de" in WHISPER_LANGUAGE_TO_KOKORO_LANG
    assert "de" not in declared
    assert {"en", "es", "fr", "it", "pt", "ja", "zh", "hi"} == set(declared)


def test_mms_map_is_the_capability():
    """Unlike Kokoro, every MMS entry names a distinct per-language checkpoint."""
    from speech_to_speech.TTS.facebookmms_handler import WHISPER_LANGUAGE_TO_FACEBOOK_LANGUAGE

    assert backend_languages("tts", "facebookMMS") == frozenset(WHISPER_LANGUAGE_TO_FACEBOOK_LANGUAGE)


# --- the usable intersection ---------------------------------------------------------------


def test_usable_set_is_the_intersection_of_both_routes():
    usable = usable_languages("parakeet-tdt", "kokoro")

    assert usable == frozenset({"en", "es", "fr", "it", "pt"})
    # Strictly smaller than either side: this is the whole point of the check.
    assert usable < backend_languages("stt", "parakeet-tdt")
    assert usable < backend_languages("tts", "kokoro")


def test_usable_set_is_unknown_when_either_route_is_undeclared():
    assert usable_languages("parakeet-tdt", "qwen3") is None
    assert usable_languages("openai", "kokoro") is None


# --- checking a selection ------------------------------------------------------------------


def test_language_supported_by_both_routes_is_accepted():
    result = check_language("es", stt="parakeet-tdt", tts="kokoro")

    assert result.supported is True
    assert result.is_rejected is False


def test_language_missing_from_tts_is_rejected_naming_the_route():
    """German transcribes fine and is then spoken by an English voice -- silently, today."""
    result = check_language("de", stt="parakeet-tdt", tts="kokoro")

    assert result.supported is False
    assert result.is_rejected is True
    assert "TTS backend 'kokoro'" in result.reason


def test_language_missing_from_stt_is_rejected():
    result = check_language("th", stt="parakeet-tdt", tts="facebookMMS")

    assert result.supported is False
    assert "STT backend 'parakeet-tdt'" in result.reason


def test_undeclared_route_yields_unknown_rather_than_either_verdict():
    result = check_language("de", stt="parakeet-tdt", tts="qwen3")

    assert result.supported is None
    assert result.is_rejected is False
    assert "does not declare" in result.reason


@pytest.mark.parametrize("language", [None, "", "auto"])
def test_auto_and_absent_selections_are_always_accepted(language):
    """These ask the pipeline to decide; they do not name a language to support."""
    assert check_language(language, stt="parakeet-tdt", tts="kokoro").supported is True


# --- startup reporting ----------------------------------------------------------------------


def test_startup_reports_the_usable_set(caplog):
    caplog.set_level(logging.INFO)

    log_usable_languages("parakeet-tdt", "kokoro")

    assert "usable end to end" in caplog.text
    assert "es" in caplog.text


def test_startup_stays_quiet_when_capability_is_unknown(caplog):
    caplog.set_level(logging.INFO)

    log_usable_languages("parakeet-tdt", "qwen3")

    assert caplog.records == []


def test_startup_warns_when_the_routes_share_nothing(caplog, monkeypatch):
    import speech_to_speech.pipeline.language_support as module

    monkeypatch.setattr(module, "usable_languages", lambda stt, tts: frozenset())
    caplog.set_level(logging.INFO)

    module.log_usable_languages("stt-x", "tts-y")

    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert "no language in common" in caplog.text
