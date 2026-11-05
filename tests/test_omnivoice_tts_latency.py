from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.pipeline.turn_latency import TurnLatencyStore
from speech_to_speech.TTS import omnivoice_handler as omnivoice_module
from speech_to_speech.TTS.omnivoice_handler import OmniVoiceTTSHandler


def _waveform(samples: int = 768) -> list[np.ndarray]:
    return [np.linspace(-0.2, 0.2, samples, dtype=np.float32)]


def _handler(*, samples: int = 768, cancel_scope: CancelScope | None = None) -> OmniVoiceTTSHandler:
    handler = OmniVoiceTTSHandler.__new__(OmniVoiceTTSHandler)
    handler.model = SimpleNamespace(
        sampling_rate=16000,
        generate=lambda **_kwargs: _waveform(samples),
    )
    handler.voice_clone_prompt = None
    handler.language_voice_clone_prompts = {}
    handler.instruct = None
    handler.language = None
    handler.num_steps = 32
    handler.speed = 1.0
    handler.blocksize = 512
    handler.cancel_scope = cancel_scope
    handler.speculative_turns = None
    return handler


@pytest.fixture(autouse=True)
def _quiet_console(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(omnivoice_module.console, "print", lambda *_a, **_k: None)


def test_records_ttfa_before_resample_and_e2e_on_first_yield(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [100.0]
    monkeypatch.setattr("speech_to_speech.pipeline.turn_latency.perf_counter", lambda: clock[0])

    def slow_resample(chunk, up, down):
        clock[0] = 100.40
        return np.asarray(chunk, dtype=np.float32)

    monkeypatch.setattr(omnivoice_module.scipy.signal, "resample_poly", slow_resample)

    handler = _handler()
    handler.model.sampling_rate = 24000

    def generate(**_kwargs):
        clock[0] = 100.12
        return _waveform()

    handler.model.generate = generate
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_response("resp-1", turn_id="turn_1", turn_revision=0)

    outputs = list(
        handler.process(
            TTSInput(
                text="hello",
                response_key="resp-1",
                turn_id="turn_1",
                turn_revision=0,
                speech_stopped_at_s=99.0,
            )
        )
    )

    assert outputs
    assert tracker.tts_ttfa_s == pytest.approx(0.12)
    assert tracker.e2e_s == pytest.approx(1.40)


def test_preserves_first_audio_across_later_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [50.0]
    monkeypatch.setattr("speech_to_speech.pipeline.turn_latency.perf_counter", lambda: clock[0])

    def first_generate(**_kwargs):
        clock[0] = 50.08
        return _waveform()

    handler = _handler()
    handler.model.generate = first_generate
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_response("resp-1", turn_id="turn_1", turn_revision=0)

    list(
        handler.process(
            TTSInput(
                text="one",
                response_key="resp-1",
                turn_id="turn_1",
                turn_revision=0,
                speech_stopped_at_s=49.5,
            )
        )
    )
    assert tracker.tts_ttfa_s == pytest.approx(0.08)
    first_e2e = tracker.e2e_s
    assert first_e2e is not None

    clock[0] = 80.0

    def later_generate(**_kwargs):
        clock[0] = 80.50
        return _waveform()

    handler.model.generate = later_generate
    list(
        handler.process(
            TTSInput(
                text="two",
                response_key="resp-1",
                turn_id="turn_1",
                turn_revision=0,
                speech_stopped_at_s=49.5,
            )
        )
    )

    assert tracker.tts_ttfa_s == pytest.approx(0.08)
    assert tracker.e2e_s == pytest.approx(first_e2e)


@pytest.mark.parametrize("audios", [[], [np.array([], dtype=np.float32)]])
def test_empty_audio_leaves_latency_unset(monkeypatch: pytest.MonkeyPatch, audios: list[np.ndarray]) -> None:
    clock = [1.0]
    monkeypatch.setattr("speech_to_speech.pipeline.turn_latency.perf_counter", lambda: clock[0])

    handler = _handler()
    handler.model.generate = lambda **_kwargs: audios
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_response("resp-3", turn_id="turn_3", turn_revision=0)

    assert list(handler.process(TTSInput(text="x", response_key="resp-3", speech_stopped_at_s=0.5))) == []
    assert tracker.tts_ttfa_s is None
    assert tracker.e2e_s is None


@pytest.mark.parametrize("cancel_at", ["generation", "after_first", "discard"])
def test_cancelled_or_discarded_audio_cannot_record_another_response(
    monkeypatch: pytest.MonkeyPatch, cancel_at: str
) -> None:
    clock = [10.0]
    monkeypatch.setattr("speech_to_speech.pipeline.turn_latency.perf_counter", lambda: clock[0])
    handler = _handler(cancel_scope=CancelScope())
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_response("old")
    other = store.get_or_create_response("new")

    def generate(**_kwargs):
        clock[0] = 10.12
        if cancel_at == "generation":
            handler.cancel_scope.cancel()
        elif cancel_at == "discard":
            store.discard_response("old")
        return _waveform()

    handler.model.generate = generate
    output = handler.process(TTSInput(text="hello", response_key="old", speech_stopped_at_s=9.0))
    if cancel_at == "after_first":
        assert next(output).shape == (512,)
        handler.cancel_scope.cancel()
        assert list(output) == []
        assert tracker.tts_ttfa_s == pytest.approx(0.12)
        assert tracker.e2e_s == pytest.approx(1.12)
    elif cancel_at == "discard":
        assert list(output)
        assert store.get_response("old") is None
        assert tracker.tts_ttfa_s is None
        assert tracker.e2e_s is None
    else:
        assert list(output) == []
        assert tracker.tts_ttfa_s is None
        assert tracker.e2e_s is None
    assert other.tts_ttfa_s is None
    assert other.e2e_s is None
