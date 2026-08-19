from __future__ import annotations

from collections.abc import Callable
from threading import Event

import numpy as np
import pytest

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.TTS import supertonic_tts_handler as supertonic_module
from speech_to_speech.TTS.supertonic_tts_handler import SupertonicTTSHandler


class FakeTTS:
    sample_rate = 44100

    def __init__(self, synthesize: Callable[..., tuple[np.ndarray, np.ndarray]]) -> None:
        self.synthesize = synthesize


def make_handler(
    synthesize: Callable[..., tuple[np.ndarray, np.ndarray]],
    *,
    blocksize: int = 4,
    cancel_scope: CancelScope | None = None,
) -> SupertonicTTSHandler:
    handler = SupertonicTTSHandler.__new__(SupertonicTTSHandler)
    handler.tts = FakeTTS(synthesize)
    handler.voice_style = object()
    handler.speed = 1.0
    handler.lang = "na"
    handler.blocksize = blocksize
    handler.cancel_scope = cancel_scope
    handler.speculative_turns = None
    return handler


@pytest.mark.parametrize(
    ("setup_kwargs", "message"),
    [
        ({"blocksize": 0}, "blocksize must be positive"),
        ({"lang": "zh"}, "Unsupported Supertonic language code"),
    ],
)
def test_setup_rejects_invalid_configuration_before_loading_models(setup_kwargs, message: str) -> None:
    handler = SupertonicTTSHandler.__new__(SupertonicTTSHandler)

    with pytest.raises(ValueError, match=message):
        handler.setup(Event(), **setup_kwargs)


@pytest.mark.parametrize(
    ("pipeline_code", "expected_code"),
    [
        ("en-auto", "en"),
        ("pt-BR", "pt"),
        ("no", "na"),
        ("sr", "na"),
        ("zh", "na"),
        (None, "na"),
    ],
)
def test_process_normalizes_or_falls_back_from_pipeline_language_codes(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_code: str | None,
    expected_code: str,
) -> None:
    languages = []

    def synthesize(**kwargs):
        languages.append(kwargs["lang"])
        return np.zeros((1, 4), dtype=np.float32), np.array([0.1])

    monkeypatch.setattr(supertonic_module.scipy.signal, "resample_poly", lambda *_args: np.zeros(4))
    handler = make_handler(synthesize)

    list(handler.process(TTSInput(text="Hello", language_code=pipeline_code)))

    assert languages == [expected_code]


def test_process_clips_int16_audio_and_pads_the_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    def synthesize(**_kwargs):
        return np.zeros((1, 3), dtype=np.float32), np.array([0.1])

    monkeypatch.setattr(
        supertonic_module.scipy.signal,
        "resample_poly",
        lambda *_args: np.array([2.0, -2.0, 0.5], dtype=np.float32),
    )
    handler = make_handler(synthesize)

    chunks = list(handler.process(TTSInput(text="Hello", language_code="en")))

    assert len(chunks) == 1
    assert chunks[0].dtype == np.int16
    np.testing.assert_array_equal(chunks[0], np.array([32767, -32768, 16384, 0], dtype=np.int16))


def test_process_drops_audio_when_cancelled_during_synthesis() -> None:
    cancel_scope = CancelScope()

    def synthesize(**_kwargs):
        cancel_scope.cancel()
        return np.zeros((1, 4), dtype=np.float32), np.array([0.1])

    handler = make_handler(synthesize, cancel_scope=cancel_scope)

    assert list(handler.process(TTSInput(text="Hello", language_code="en"))) == []


def test_process_checks_cancellation_before_padded_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    cancel_scope = CancelScope()

    def synthesize(**_kwargs):
        return np.zeros((1, 5), dtype=np.float32), np.array([0.1])

    monkeypatch.setattr(supertonic_module.scipy.signal, "resample_poly", lambda *_args: np.zeros(5))
    handler = make_handler(synthesize, cancel_scope=cancel_scope)
    chunks = handler.process(TTSInput(text="Hello", language_code="en"))

    assert len(next(chunks)) == 4
    cancel_scope.cancel()
    with pytest.raises(StopIteration):
        next(chunks)
