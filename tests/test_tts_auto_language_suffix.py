"""TTS backends must strip the ``-auto`` suffix before mapping a language.

With ``--language auto`` the STT handlers report ``"de-auto"`` rather than ``"de"``, so
downstream code knows the language may change between turns. The TTS language maps are keyed
on bare Whisper codes, so passing the suffixed value straight into a lookup never matches:

* ``WHISPER_LANGUAGE_TO_KOKORO_LANG.get("de-auto", self.lang_code)`` returns the fallback, so
  Kokoro silently keeps the startup voice and never switches language -- the exact feature
  that code exists to provide.
* ``WHISPER_LANGUAGE_TO_FACEBOOK_LANGUAGE["de-auto"]`` raises ``KeyError``, which MMS catches
  and reports as "Unsupported language" before falling back to English -- for a language it
  fully supports.

Neither failure is loud: Kokoro's "Language change detected" line never fires, and MMS blames
the language.
"""

from __future__ import annotations

import pytest

from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.TTS.facebookmms_handler import (
    WHISPER_LANGUAGE_TO_FACEBOOK_LANGUAGE,
    FacebookMMSTTSHandler,
)
from speech_to_speech.TTS.kokoro_handler import WHISPER_LANGUAGE_TO_KOKORO_LANG, KokoroTTSHandler


def tts_input(language_code: str | None) -> TTSInput:
    return TTSInput(text="Guten Tag.", language_code=language_code, turn_id="t1", turn_revision=0)


# --- the maps only accept bare codes ------------------------------------------------------


@pytest.mark.parametrize("code", ["de", "fr", "en"])
def test_kokoro_map_is_keyed_on_bare_codes(code):
    assert WHISPER_LANGUAGE_TO_KOKORO_LANG.get(code) is not None
    assert WHISPER_LANGUAGE_TO_KOKORO_LANG.get(f"{code}-auto") is None


def test_facebookmms_map_is_keyed_on_bare_codes():
    assert WHISPER_LANGUAGE_TO_FACEBOOK_LANGUAGE["de"] == "deu"
    with pytest.raises(KeyError):
        WHISPER_LANGUAGE_TO_FACEBOOK_LANGUAGE["de-auto"]


# --- Kokoro -------------------------------------------------------------------------------


def _kokoro(monkeypatch, backend="kokoro"):
    """A handler stubbed down to the language decision."""
    handler = object.__new__(KokoroTTSHandler)
    handler.backend = backend
    handler.lang_code = "b"
    handler.voice = "bm_fable"
    handler.speculative_turns = None
    handler.cancel_scope = None
    seen: list[str | None] = []

    def capture(text, language_code=None):
        seen.append(language_code)
        return iter(())

    monkeypatch.setattr(handler, "_process_kokoro", capture, raising=False)
    monkeypatch.setattr(handler, "_process_mlx", capture, raising=False)
    return handler, seen


@pytest.mark.parametrize("backend", ["kokoro", "mlx"])
def test_kokoro_receives_a_bare_language_code(monkeypatch, backend):
    handler, seen = _kokoro(monkeypatch, backend)

    list(handler.process(tts_input("de-auto")))

    assert seen == ["de"]
    assert WHISPER_LANGUAGE_TO_KOKORO_LANG.get(seen[0]) == "b"


def test_kokoro_pinned_language_is_unchanged(monkeypatch):
    handler, seen = _kokoro(monkeypatch)

    list(handler.process(tts_input("fr")))

    assert seen == ["fr"]


def test_kokoro_absent_language_stays_none(monkeypatch):
    """None must survive: it means "no per-turn language", not "unknown language"."""
    handler, seen = _kokoro(monkeypatch)

    list(handler.process(tts_input(None)))

    assert seen == [None]


# --- Facebook MMS -------------------------------------------------------------------------


def _mms(monkeypatch):
    handler = object.__new__(FacebookMMSTTSHandler)
    handler.speculative_turns = None
    handler.cancel_scope = None
    handler.language = "en"
    handler.model_name = None
    handler._initial_language = "en"
    handler._initial_model_name = None
    requested: list[str] = []

    def load_model(language_code, model_name=None):
        # Mirror the real lookup so a suffixed code still fails here.
        WHISPER_LANGUAGE_TO_FACEBOOK_LANGUAGE[language_code]
        requested.append(language_code)
        handler.language = language_code

    monkeypatch.setattr(handler, "load_model", load_model, raising=False)
    monkeypatch.setattr(handler, "generate_audio", lambda text: None, raising=False)
    return handler, requested


def test_facebookmms_loads_the_model_for_an_auto_detected_language(monkeypatch, caplog):
    handler, requested = _mms(monkeypatch)

    list(handler.process(tts_input("de-auto")))

    assert requested == ["de"]
    assert "Unsupported language" not in caplog.text


def test_facebookmms_pinned_language_is_unchanged(monkeypatch):
    handler, requested = _mms(monkeypatch)

    list(handler.process(tts_input("fr")))

    assert requested == ["fr"]


def test_facebookmms_skips_reload_when_language_already_matches(monkeypatch):
    """ "de-auto" must compare equal to an already-loaded "de", not force a reload."""
    handler, requested = _mms(monkeypatch)
    handler.language = "de"

    list(handler.process(tts_input("de-auto")))

    assert requested == []


def test_facebookmms_unsupported_language_still_warns(monkeypatch, caplog):
    """Normalizing must not swallow the genuine unsupported case."""
    handler, requested = _mms(monkeypatch)

    list(handler.process(tts_input("xx-auto")))

    assert requested == []
    assert "Unsupported language" in caplog.text
