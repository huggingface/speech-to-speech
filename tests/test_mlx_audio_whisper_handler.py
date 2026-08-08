"""Tests for MLXAudioWhisperSTTHandler's MLX lock usage and language resolution.

`utils/mlx_lock.py` documents the global lock as mandatory for every MLX handler: MLX
models share one Metal command queue, so concurrent inference from the STT/LLM/TTS threads
aborts the process with

    -[_MTLCommandBuffer addCompletedHandler:]:1011: failed assertion
    `Completed handler provided after commit call'

This handler was the only MLX inference path that did not acquire it. These tests pin the
lock down (including that it is actually *held* while `generate()` runs, not merely
imported) and cover the language-resolution edge cases around it. No MLX or mlx-audio
install is needed -- the model is faked.
"""

from __future__ import annotations

import sys
import threading
import types

import numpy as np
import pytest

from speech_to_speech.pipeline.messages import VADAudio
from speech_to_speech.STT.mlx_audio_whisper_handler import MLXAudioWhisperSTTHandler
from speech_to_speech.utils import mlx_lock


class FakeResult:
    def __init__(self, text: str, language=None) -> None:
        self.text = text
        if language is not None:
            self.language = language


class FakeModel:
    """Records lock ownership at the moment `generate()` is entered."""

    def __init__(self, result) -> None:
        self.result = result
        self.calls: list[dict] = []
        self.lock_held_during_generate: list[bool] = []

    def generate(self, audio, verbose=False, **gen_kwargs):
        self.calls.append(gen_kwargs)
        # The global MLX lock is an RLock; a non-blocking acquire from another thread
        # fails only if this thread's caller already holds it.
        self.lock_held_during_generate.append(_lock_held_by_other_thread())
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _lock_held_by_other_thread() -> bool:
    """True if the global MLX lock is currently held (probed from a separate thread)."""
    acquired_elsewhere: list[bool] = []

    def probe():
        got = mlx_lock._mlx_lock.acquire(blocking=False)
        acquired_elsewhere.append(got)
        if got:
            mlx_lock._mlx_lock.release()

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    return not acquired_elsewhere[0]


def make_handler(*, start_language, result, last_language=...):
    handler = object.__new__(MLXAudioWhisperSTTHandler)
    handler.model_name = "mlx-community/whisper-large-v3-turbo"
    handler.start_language = start_language
    handler.last_language = (
        (start_language if start_language != "auto" else None) if last_language is ... else last_language
    )
    handler.gen_kwargs = {}
    handler.model = FakeModel(result)
    return handler


def vad_audio():
    return VADAudio(audio=np.zeros(16000, dtype=np.float32), turn_id="turn_1", turn_revision=0)


def run(handler):
    outputs = list(handler.process(vad_audio()))
    assert len(outputs) == 1
    return outputs[0]


# --- the MLX lock ------------------------------------------------------------------------


def test_process_holds_the_global_mlx_lock_during_generate():
    handler = make_handler(start_language="en", result=FakeResult("Hello.", "en"))

    run(handler)

    assert handler.model.lock_held_during_generate == [True]


def test_warmup_holds_the_global_mlx_lock_during_generate():
    handler = make_handler(start_language="en", result=FakeResult("", "en"))

    handler.warmup()

    assert handler.model.lock_held_during_generate == [True]


def test_mlx_lock_is_released_after_process():
    handler = make_handler(start_language="en", result=FakeResult("Hello.", "en"))

    run(handler)

    assert not _lock_held_by_other_thread()


def test_mlx_lock_is_released_when_inference_raises():
    """A failed transcription must not leave the whole pipeline's MLX lock held."""
    handler = make_handler(start_language="en", result=RuntimeError("metal boom"))

    result = run(handler)

    assert result.text == ""
    assert not _lock_held_by_other_thread()


def test_mlx_lock_is_released_when_warmup_raises():
    handler = make_handler(start_language="en", result=RuntimeError("metal boom"))

    handler.warmup()  # warmup swallows failures by design

    assert not _lock_held_by_other_thread()


def test_concurrent_process_calls_are_serialized():
    """Two threads must never be inside generate() at the same time."""
    overlap_detected = []
    inside = []
    inside_lock = threading.Lock()
    barrier_wait_s = 0.05

    class OverlapDetectingModel:
        def generate(self, audio, verbose=False, **gen_kwargs):
            with inside_lock:
                inside.append(1)
                overlap_detected.append(len(inside) > 1)
            # Hold the "GPU" long enough that an unserialized second caller would overlap.
            threading.Event().wait(barrier_wait_s)
            with inside_lock:
                inside.pop()
            return FakeResult("Hello.", "en")

    handlers = [make_handler(start_language="en", result=None) for _ in range(2)]
    for handler in handlers:
        handler.model = OverlapDetectingModel()

    threads = [threading.Thread(target=run, args=(handler,)) for handler in handlers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert overlap_detected == [False, False]


# --- language resolution -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("language", "expected_last_language"),
    [("auto", None), ("de", "de"), (None, None)],
)
def test_setup_does_not_store_auto_as_last_language(monkeypatch, language, expected_last_language):
    """`--language auto` is a request to detect, not a language code. Storing it as
    `last_language` makes it fail every SUPPORTED_LANGUAGES check downstream."""
    fake_model = FakeModel(FakeResult("", "en"))
    fake_model._processor = object()  # skip the WhisperProcessor fallback path

    stt_generate = types.ModuleType("mlx_audio.stt.generate")
    stt_generate.load_model = lambda model_name: fake_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_audio", types.ModuleType("mlx_audio"))
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", types.ModuleType("mlx_audio.stt"))
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.generate", stt_generate)

    handler = object.__new__(MLXAudioWhisperSTTHandler)
    handler.setup(model_name="mlx-community/whisper-large-v3-turbo", language=language)

    assert handler.start_language == language
    assert handler.last_language == expected_last_language
    # setup() warms up, and that warmup must hold the lock too.
    assert fake_model.lock_held_during_generate == [True]


def test_auto_language_reports_the_detected_language():
    handler = make_handler(start_language="auto", result=FakeResult("Hallo.", "de"))

    assert run(handler).language_code == "de-auto"
    assert handler.last_language == "de"


def test_forced_language_is_authoritative_and_passed_to_generate():
    handler = make_handler(start_language="de", result=FakeResult("Hallo.", "it"))

    result = run(handler)

    assert handler.model.calls == [{"language": "de"}]
    assert result.language_code == "de"


def test_missing_language_attribute_falls_back_without_warning_noise():
    handler = make_handler(start_language="auto", result=FakeResult("Hello."), last_language="de")

    assert run(handler).language_code == "de-auto"


def test_none_language_attribute_is_treated_as_absent(caplog):
    """`hasattr(result, "language")` was true even when the value was None, so the handler
    reported `None` as an "unsupported language"."""
    handler = make_handler(start_language="auto", result=FakeResult("Hello.", None), last_language="de")
    # FakeResult only sets .language when non-None, so set it explicitly.
    handler.model.result.language = None

    result = run(handler)

    assert result.language_code == "de-auto"
    assert "unsupported language: None" not in caplog.text.lower()


def test_unsupported_detected_language_falls_back_to_last_language():
    handler = make_handler(start_language="auto", result=FakeResult("Privet.", "ru"), last_language="de")

    assert run(handler).language_code == "de-auto"


def test_unsupported_detected_language_defaults_to_english_without_fallback():
    handler = make_handler(start_language="auto", result=FakeResult("Privet.", "ru"), last_language=None)

    assert run(handler).language_code == "en-auto"


@pytest.mark.parametrize("start_language", ["auto", None, "de"])
def test_language_code_is_always_a_string(start_language):
    handler = make_handler(start_language=start_language, result=FakeResult("Hello."), last_language=None)

    assert isinstance(run(handler).language_code, str)
