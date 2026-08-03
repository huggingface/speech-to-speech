"""Tests that the faster-whisper STT handler reports the spoken language.

`process()` unpacked `segments, info = self.model.transcribe(...)` and then never used
`info`, yielding `Transcription` without a `language_code`. The field defaults to `None`, and
`resolve_auto_language(None)` returns `(None, None)`, so the LLM backends' language prompt

    if lang_name and self.enable_lang_prompt:

could never fire with this backend, and TTS handlers that switch voice per turn never saw a
language. faster-whisper detects the language and exposes it on `info.language` -- the value
was simply dropped.

`--faster_whisper_stt_gen_language auto` also reached `transcribe()` verbatim, which rejects
unrecognised language names; auto-detection needs `language` absent entirely.

`faster_whisper` is an optional extra, so it is stubbed and these tests run anywhere.
"""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

from speech_to_speech.LLM.utils import resolve_auto_language
from speech_to_speech.pipeline.messages import VADAudio


class FakeInfo:
    def __init__(self, language=None, language_probability=None):
        if language is not None:
            self.language = language
        if language_probability is not None:
            self.language_probability = language_probability


class FakeSegment:
    def __init__(self, text, start=0.0, end=1.0):
        self.text = text
        self.start = start
        self.end = end


class FakeWhisperModel:
    """Stand-in for faster_whisper.WhisperModel; records transcribe() kwargs."""

    last_init: dict | None = None

    def __init__(self, model_name, device=None, compute_type=None):
        type(self).last_init = {"model_name": model_name, "device": device, "compute_type": compute_type}
        self.calls: list[dict] = []
        self.segments = [FakeSegment(" Hello there.")]
        self.info = FakeInfo(language="en", language_probability=0.99)

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        return iter(self.segments), self.info


@pytest.fixture
def handler_module(monkeypatch):
    """Import the handler with its optional dependency stubbed out."""
    fake_pkg = types.ModuleType("faster_whisper")
    fake_pkg.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_pkg)
    monkeypatch.delitem(sys.modules, "speech_to_speech.STT.faster_whisper_handler", raising=False)

    module = importlib.import_module("speech_to_speech.STT.faster_whisper_handler")

    yield module

    sys.modules.pop("speech_to_speech.STT.faster_whisper_handler", None)


def make_handler(handler_module, *, language="en", segments=None, info=None):
    handler = object.__new__(handler_module.FasterWhisperSTTHandler)
    gen_kwargs = {"return_timestamps": False}
    if language is not ...:
        gen_kwargs["language"] = language
    handler.setup(model_name="tiny", gen_kwargs=gen_kwargs)
    if segments is not None:
        handler.model.segments = segments
    if info is not None:
        handler.model.info = info
    return handler


def vad_audio():
    return VADAudio(audio=np.zeros(16000, dtype=np.float32), turn_id="turn_1", turn_revision=0)


def run(handler):
    return list(handler.process(vad_audio()))


# --- the dropped language -----------------------------------------------------------------


def test_pinned_language_is_reported_on_the_transcription(handler_module):
    """Previously language_code was None even when the user pinned a language."""
    handler = make_handler(handler_module, language="de")

    outputs = run(handler)

    assert len(outputs) == 1
    assert outputs[0].language_code == "de"
    # A pinned language is passed straight through to transcribe().
    assert handler.model.calls[0]["language"] == "de"


def test_detected_language_is_reported_with_the_auto_suffix(handler_module):
    handler = make_handler(
        handler_module,
        language="auto",
        info=FakeInfo(language="sv", language_probability=0.91),
    )

    outputs = run(handler)

    assert outputs[0].language_code == "sv-auto"


def test_reported_language_resolves_to_an_llm_language_name(handler_module):
    """The whole point: the LLM language prompt can now actually fire."""
    handler = make_handler(handler_module, language="auto", info=FakeInfo(language="de"))

    language_code = run(handler)[0].language_code

    assert resolve_auto_language(language_code) == ("de", "german")


def test_language_code_was_previously_unusable_for_the_prompt(handler_module):
    """Guard the inverse: a missing language resolves to no name, hence no prompt."""
    assert resolve_auto_language(None) == (None, None)


# --- "auto" must not reach transcribe() ---------------------------------------------------


@pytest.mark.parametrize("value", ["auto", "AUTO", " auto ", "", "none", "None", "null"])
def test_auto_sentinels_are_removed_from_gen_kwargs(handler_module, value):
    """transcribe() rejects unrecognised language names, so auto means 'omit it'."""
    handler = make_handler(handler_module, language=value)

    assert "language" not in handler.gen_kwargs
    run(handler)
    assert "language" not in handler.model.calls[0]


def test_absent_language_is_treated_as_auto_detect(handler_module):
    handler = make_handler(handler_module, language=..., info=FakeInfo(language="fr"))

    assert handler.start_language is None
    assert run(handler)[0].language_code == "fr-auto"


def test_non_string_language_is_treated_as_auto_detect(handler_module):
    handler = object.__new__(handler_module.FasterWhisperSTTHandler)
    handler.setup(model_name="tiny", gen_kwargs={"return_timestamps": False, "language": None})

    assert handler.start_language is None
    assert "language" not in handler.gen_kwargs


def test_pinned_language_is_stripped_of_whitespace(handler_module):
    handler = make_handler(handler_module, language="  de  ")

    assert handler.start_language == "de"
    assert handler.gen_kwargs["language"] == "de"
    assert run(handler)[0].language_code == "de"


# --- degenerate detection output ----------------------------------------------------------


def test_missing_language_on_info_reports_none_rather_than_crashing(handler_module):
    handler = make_handler(handler_module, language="auto", info=FakeInfo())

    assert run(handler)[0].language_code is None


def test_empty_detected_language_reports_none(handler_module):
    handler = make_handler(handler_module, language="auto", info=FakeInfo(language=""))

    assert run(handler)[0].language_code is None


def test_empty_transcription_still_yields_nothing(handler_module):
    """Pre-existing behaviour: silence produces no Transcription at all."""
    handler = make_handler(handler_module, language="auto", segments=[FakeSegment("   ")])

    assert run(handler) == []


def test_segment_text_is_joined_and_stripped(handler_module):
    handler = make_handler(
        handler_module,
        language="en",
        segments=[FakeSegment(" Hello"), FakeSegment(" there.")],
    )

    assert run(handler)[0].text == "Hello  there."
