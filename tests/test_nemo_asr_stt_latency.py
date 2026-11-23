from __future__ import annotations

import pytest

from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.pipeline.turn_latency import TurnLatencyStore
from speech_to_speech.STT import nemo_asr_handler
from tests.test_nemo_asr_handler import _handler, _vad_audio


@pytest.fixture(autouse=True)
def _quiet_console(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nemo_asr_handler.console, "print", lambda *args, **kwargs: None)


def test_final_transcription_records_stt_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [100.0]
    monkeypatch.setattr(nemo_asr_handler, "perf_counter", lambda: clock[0])

    handler = _handler(result="hello")
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_for_turn("turn_1", 2)

    original_transcribe = handler._transcribe

    def timed_transcribe(audio):
        clock[0] = 100.18
        return original_transcribe(audio)

    handler._transcribe = timed_transcribe  # type: ignore[method-assign]

    events = list(handler.process(_vad_audio("final")))

    assert len(events) == 1
    assert isinstance(events[0], Transcription)
    assert tracker.stt_s == pytest.approx(0.18)


def test_progressive_transcription_does_not_record_stt(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [50.0]
    monkeypatch.setattr(nemo_asr_handler, "perf_counter", lambda: clock[0])

    handler = _handler(result="hello")
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_for_turn("turn_1", 2)

    original_transcribe = handler._transcribe

    def timed_transcribe(audio):
        clock[0] = 50.25
        return original_transcribe(audio)

    handler._transcribe = timed_transcribe  # type: ignore[method-assign]

    events = list(handler.process(_vad_audio("progressive")))

    assert len(events) == 1
    assert isinstance(events[0], PartialTranscription)
    assert tracker.stt_s is None


def test_farsi_final_path_records_stt_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [10.0]
    monkeypatch.setattr(nemo_asr_handler, "perf_counter", lambda: clock[0])

    handler = _handler(language="fa", result="unused")
    handler._is_farsi = True
    handler._detects_utterance_language = False
    handler._detect_language_from_text = False
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_for_turn("turn_1", 2)

    def fake_farsi(_audio):
        clock[0] = 10.07
        return "سلام"

    handler._transcribe_farsi = fake_farsi  # type: ignore[method-assign]

    events = list(handler.process(_vad_audio("final")))

    assert len(events) == 1
    assert isinstance(events[0], Transcription)
    assert events[0].text == "سلام"
    assert tracker.stt_s == pytest.approx(0.07)
