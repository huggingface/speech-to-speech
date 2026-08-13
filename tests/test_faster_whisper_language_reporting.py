"""Tests that the faster-whisper STT handler reports the spoken language.

`process()` unpacked `segments, info = self.model.transcribe(...)` and then never used
`info`, yielding `Transcription` without a `language_code`. The field defaults to `None`, and
`resolve_auto_language(None)` returns `(None, None)`, so the LLM backends' language prompt

    if lang_name and self.enable_lang_prompt:

could never fire with this backend, and TTS handlers that switch voice per turn never saw a
language. faster-whisper detects the language and exposes it on `info.language`.

The reported language is always `info.language`, never the request: faster-whisper does not
always honour a requested language. An English-only checkpoint (`tiny.en`, the default)
overrides any request to `en`, so reporting the request would drive the LLM and TTS in a
language the audio was never transcribed in.

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

    def __init__(self, model_name, device=None, compute_type=None):
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


# --- the reported language must be the one actually used ----------------------------------


def test_reports_the_transcribed_language_not_the_requested_one(handler_module):
    """`tiny.en` overrides a requested non-English language to `en` and records it in info.

    Reporting the request would tell the LLM and TTS to use German for English audio.
    """
    handler = make_handler(handler_module, language="de", info=FakeInfo(language="en"))

    assert run(handler)[0].language_code == "en"


def test_pinned_language_is_reported_when_the_model_honours_it(handler_module):
    handler = make_handler(handler_module, language="de", info=FakeInfo(language="de"))

    outputs = run(handler)

    assert outputs[0].language_code == "de"
    assert handler.model.calls[0]["language"] == "de"


def test_detected_language_is_reported_with_the_auto_suffix(handler_module):
    handler = make_handler(
        handler_module,
        language="auto",
        info=FakeInfo(language="sv", language_probability=0.91),
    )

    assert run(handler)[0].language_code == "sv-auto"


def test_auto_suffix_depends_on_the_request_not_the_detection(handler_module):
    """A pinned request never gets `-auto`, even when the model picks a different language."""
    handler = make_handler(handler_module, language="de", info=FakeInfo(language="fr"))

    code = run(handler)[0].language_code

    assert code == "fr"
    assert not code.endswith("-auto")


def test_reported_language_resolves_to_an_llm_language_name(handler_module):
    """The whole point: the LLM language prompt can now actually fire."""
    handler = make_handler(handler_module, language="auto", info=FakeInfo(language="de"))

    assert resolve_auto_language(run(handler)[0].language_code) == ("de", "german")


@pytest.mark.parametrize("code", ["ar", "tr", "he", "th", "yue"])
def test_languages_outside_the_original_map_now_resolve(handler_module, code):
    """faster-whisper reports ~100 languages; the name map has to cover them."""
    handler = make_handler(handler_module, language="auto", info=FakeInfo(language=code))

    resolved_code, name = resolve_auto_language(run(handler)[0].language_code)

    assert resolved_code == code
    assert name, f"{code} has no LLM language name, so no prompt would be emitted"


# --- setup must not consume the caller's configuration ------------------------------------


def test_setup_does_not_mutate_the_caller_gen_kwargs(handler_module):
    """adapt_gen_kwargs() mutates what it is given, so setup() has to normalize a copy."""
    caller_kwargs = {"return_timestamps": False, "language": "auto"}
    original = dict(caller_kwargs)

    handler = object.__new__(handler_module.FasterWhisperSTTHandler)
    handler.setup(model_name="tiny", gen_kwargs=caller_kwargs)

    assert caller_kwargs == original


def test_setup_does_not_mutate_the_caller_gen_kwargs_when_pinned(handler_module):
    caller_kwargs = {"return_timestamps": True, "language": "de"}
    original = dict(caller_kwargs)

    handler = object.__new__(handler_module.FasterWhisperSTTHandler)
    handler.setup(model_name="tiny", gen_kwargs=caller_kwargs)

    assert caller_kwargs == original


# --- "auto" must not reach transcribe() ---------------------------------------------------


@pytest.mark.parametrize("value", ["auto", "AUTO", " auto "])
def test_auto_is_removed_from_gen_kwargs(handler_module, value):
    """transcribe() rejects unrecognised language names, so auto means 'omit it'."""
    handler = make_handler(handler_module, language=value)

    assert handler.start_language is None
    assert "language" not in handler.gen_kwargs
    run(handler)
    assert "language" not in handler.model.calls[0]


def test_absent_language_is_treated_as_auto_detect(handler_module):
    handler = make_handler(handler_module, language=..., info=FakeInfo(language="fr"))

    assert handler.start_language is None
    assert run(handler)[0].language_code == "fr-auto"


def test_explicit_none_is_treated_as_auto_detect(handler_module):
    handler = make_handler(handler_module, language=None, info=FakeInfo(language="fr"))

    assert handler.start_language is None
    assert run(handler)[0].language_code == "fr-auto"


def test_invalid_language_is_passed_through_rather_than_silently_auto(handler_module):
    """An unrecognised value must reach faster-whisper so it fails loudly."""
    handler = make_handler(handler_module, language="not-a-language", info=FakeInfo(language="en"))

    assert handler.start_language == "not-a-language"
    run(handler)
    assert handler.model.calls[0]["language"] == "not-a-language"


# --- degenerate detection output ----------------------------------------------------------


def test_missing_language_on_info_falls_back_to_the_request(handler_module):
    handler = make_handler(handler_module, language="de", info=FakeInfo())

    assert run(handler)[0].language_code == "de"


def test_missing_language_on_info_in_auto_mode_reports_none(handler_module):
    handler = make_handler(handler_module, language="auto", info=FakeInfo())

    assert run(handler)[0].language_code is None


def test_empty_detected_language_falls_back_to_the_request(handler_module):
    handler = make_handler(handler_module, language="de", info=FakeInfo(language=""))

    assert run(handler)[0].language_code == "de"


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
