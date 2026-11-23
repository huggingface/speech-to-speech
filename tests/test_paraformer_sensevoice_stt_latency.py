from __future__ import annotations

import numpy as np
import pytest

from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio
from speech_to_speech.pipeline.turn_latency import TurnLatencyStore
from speech_to_speech.STT import paraformer_handler, sense_voice_handler
from tests.test_paraformer_transcription_events import _handler as _paraformer_handler
from tests.test_sense_voice_transcription_events import _handler as _sensevoice_handler


def _vad_audio(mode: str = "final") -> VADAudio:
    return VADAudio(
        audio=np.zeros(16000, dtype=np.float32),
        mode=mode,
        turn_id="turn_1",
        turn_revision=2,
        created_at_s=124.0,
        speech_end_at_s=123.0,
    )


@pytest.fixture(autouse=True)
def _quiet_consoles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paraformer_handler.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(sense_voice_handler.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(paraformer_handler.torch.mps, "empty_cache", lambda: None)


def test_paraformer_final_records_stt_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [100.0]
    monkeypatch.setattr(paraformer_handler, "perf_counter", lambda: clock[0])

    handler = _paraformer_handler()
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_for_turn("turn_1", 2)

    original_generate = handler.model.generate

    def timed_generate(audio):
        clock[0] = 100.15
        return original_generate(audio)

    handler.model.generate = timed_generate  # type: ignore[method-assign]

    events = list(handler.process(_vad_audio("final")))

    assert len(events) == 1
    assert isinstance(events[0], Transcription)
    assert tracker.stt_s == pytest.approx(0.15)


def test_paraformer_progressive_does_not_record_stt(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [50.0]
    monkeypatch.setattr(paraformer_handler, "perf_counter", lambda: clock[0])

    handler = _paraformer_handler()
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_for_turn("turn_1", 2)

    original_generate = handler.model.generate

    def timed_generate(audio):
        clock[0] = 50.22
        return original_generate(audio)

    handler.model.generate = timed_generate  # type: ignore[method-assign]

    events = list(handler.process(_vad_audio("progressive")))

    assert len(events) == 1
    assert isinstance(events[0], PartialTranscription)
    assert tracker.stt_s is None


def test_sensevoice_final_records_stt_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [10.0]
    monkeypatch.setattr(sense_voice_handler, "perf_counter", lambda: clock[0])

    handler = _sensevoice_handler()
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_for_turn("turn_1", 2)

    original_generate = handler._generate

    def timed_generate(audio):
        clock[0] = 10.11
        return original_generate(audio)

    handler._generate = timed_generate  # type: ignore[method-assign]

    events = list(handler.process(_vad_audio("final")))

    assert len(events) == 1
    assert isinstance(events[0], Transcription)
    assert tracker.stt_s == pytest.approx(0.11)


def test_sensevoice_progressive_does_not_record_stt(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [20.0]
    monkeypatch.setattr(sense_voice_handler, "perf_counter", lambda: clock[0])

    handler = _sensevoice_handler()
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_for_turn("turn_1", 2)

    original_generate = handler._generate

    def timed_generate(audio):
        clock[0] = 20.09
        return original_generate(audio)

    handler._generate = timed_generate  # type: ignore[method-assign]

    events = list(handler.process(_vad_audio("progressive")))

    assert len(events) == 1
    assert isinstance(events[0], PartialTranscription)
    assert tracker.stt_s is None
