"""The delivery ledger queue and the pipeline wiring that installs it."""

from __future__ import annotations

import threading
import time
from queue import Empty, Queue
from types import SimpleNamespace

import numpy as np

from convfill_demo.delivery import DeliveryLedgerQueue, audio_duration_s
from convfill_demo.pipeline import _attach_audio_state
from speech_to_speech.api.openai_realtime.service import PIPELINE_SAMPLE_RATE
from speech_to_speech.pipeline.events import AssistantOutputEvent
from speech_to_speech.pipeline.messages import (
    AUDIO_RESPONSE_DONE,
    PIPELINE_END,
    AssistantTextPart,
    AudioOutput,
)


def _text_event(text: str, response_key: str = "response-one") -> AssistantOutputEvent:
    return AssistantOutputEvent(parts=[AssistantTextPart(text=text)], response_key=response_key)


def _audio(seconds: float, response_key: str = "response-one") -> AudioOutput:
    samples = int(PIPELINE_SAMPLE_RATE * seconds)
    return AudioOutput(audio=b"\x00\x00" * samples, response_key=response_key)


def test_delivered_text_is_recorded_per_response_key():
    ledger = DeliveryLedgerQueue()
    ledger.put(_text_event("First sentence.", "response-one"))
    ledger.put(_text_event("Other response.", "response-two"))
    ledger.put(_text_event("Second sentence.", "response-one"))

    while True:
        try:
            ledger.get_nowait()
        except Empty:
            break

    assert ledger.delivered_text("response-one") == ["First sentence.", "Second sentence."]
    assert ledger.delivered_text("response-two") == ["Other response."]
    assert ledger.delivered_text(None) == []

    ledger.clear_response("response-one")
    assert ledger.delivered_text("response-one") == []
    assert ledger.delivered_text("response-two") == ["Other response."]


def test_durations_use_the_pipeline_sample_rate():
    # The output queue carries pipeline-rate PCM16; the client rate is applied later.
    assert audio_duration_s(_audio(3.0)) == 3.0
    assert audio_duration_s(AudioOutput(audio=np.zeros(PIPELINE_SAMPLE_RATE, dtype=np.int16))) == 1.0


def test_audio_advances_the_playhead_and_expires_after_its_duration():
    ledger = DeliveryLedgerQueue()
    assert ledger.audio_playing() is False

    ledger.put(_audio(0.15))
    assert ledger.pending_audio() is True
    ledger.get_nowait()

    assert ledger.pending_audio() is False
    assert ledger.audio_playing() is True
    time.sleep(0.2)
    assert ledger.audio_playing() is False


def test_reset_playback_drops_scheduled_audio_time():
    ledger = DeliveryLedgerQueue()
    ledger.put(_audio(5.0))
    ledger.get_nowait()
    assert ledger.audio_playing() is True

    ledger.reset_playback()
    assert ledger.audio_playing() is False


def test_sentinels_and_control_items_carry_no_playback_time():
    ledger = DeliveryLedgerQueue()
    ledger.put(AudioOutput(audio=AUDIO_RESPONSE_DONE, response_key="response-one"))
    ledger.put(PIPELINE_END)
    ledger.put(AudioOutput(audio=b"\x00\x00" * 2400, response_key="response-one", cleanup_only=True))
    assert ledger.pending_audio() is False

    while True:
        try:
            ledger.get_nowait()
        except Empty:
            break
    assert ledger.audio_playing() is False


class _FakeAdapter:
    """Stands in for ConvFillRealtimeHandler in the wiring test."""

    def __init__(self):
        self.ledger = None

    def set_delivery_ledger(self, ledger):
        self.ledger = ledger


def test_attach_audio_state_installs_the_ledger_on_the_unit_and_the_tts_handler(monkeypatch):
    import convfill_demo.pipeline as server

    monkeypatch.setattr(server, "ConvFillRealtimeHandler", _FakeAdapter)

    original_queue: Queue = Queue()
    adapter = _FakeAdapter()
    tts_handler = SimpleNamespace(queue_out=original_queue)
    unrelated_handler = SimpleNamespace(queue_out=Queue())
    unit = SimpleNamespace(
        index=0,
        output_queue=original_queue,
        response_playing=threading.Event(),
        handlers=[adapter, unrelated_handler, tts_handler],
    )
    manager = SimpleNamespace(handlers=[SimpleNamespace(pool=[unit])])

    assert _attach_audio_state(manager) is manager

    assert isinstance(unit.output_queue, DeliveryLedgerQueue)
    assert tts_handler.queue_out is unit.output_queue
    assert adapter.ledger is unit.output_queue
    assert unrelated_handler.queue_out is not unit.output_queue


def test_attach_audio_state_moves_queued_items_and_rejects_a_missing_producer(monkeypatch):
    import convfill_demo.pipeline as server

    monkeypatch.setattr(server, "ConvFillRealtimeHandler", _FakeAdapter)

    original_queue: Queue = Queue()
    original_queue.put(PIPELINE_END)
    adapter = _FakeAdapter()
    unit = SimpleNamespace(
        index=0,
        output_queue=original_queue,
        handlers=[adapter, SimpleNamespace(queue_out=original_queue)],
    )
    manager = SimpleNamespace(handlers=[SimpleNamespace(pool=[unit])])
    _attach_audio_state(manager)
    assert unit.output_queue.get_nowait() == PIPELINE_END
    assert original_queue.empty()

    orphan = SimpleNamespace(index=1, output_queue=Queue(), handlers=[_FakeAdapter()])
    orphan_manager = SimpleNamespace(handlers=[SimpleNamespace(pool=[orphan])])
    try:
        _attach_audio_state(orphan_manager)
    except RuntimeError as exc:
        assert "audio output queue" in str(exc)
    else:  # pragma: no cover - the wiring must fail loudly
        raise AssertionError("_attach_audio_state accepted a unit with no audio producer")
