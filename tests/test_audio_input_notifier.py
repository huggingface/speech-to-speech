from queue import Queue
from threading import Thread
from time import sleep

import numpy as np

from speech_to_speech.LLM.audio_input_notifier import AudioInputNotifier
from speech_to_speech.pipeline.events import AudioInputCompletedEvent
from speech_to_speech.pipeline.messages import VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.pipeline.turn_latency import TurnLatencyStore


def _notifier(
    text_output_queue: Queue | None = None,
    speculative_turns: SpeculativeTurnTracker | None = None,
) -> AudioInputNotifier:
    notifier = object.__new__(AudioInputNotifier)
    notifier.setup(
        sample_rate=16000,
        speculative_turns=speculative_turns or SpeculativeTurnTracker(),
        text_output_queue=text_output_queue or Queue(),
    )
    return notifier


def test_audio_input_notifier_uses_per_endpoint_processing_delay():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    notifier = _notifier(speculative_turns=tracker)
    item = VADAudio(
        audio=np.zeros(1600, dtype=np.float32),
        mode="final",
        turn_id="turn_1",
        turn_revision=0,
        processing_delay_s=0.2,
    )
    result: list[bool] = []
    thread = Thread(target=lambda: result.append(notifier.should_process_input(item)))
    thread.start()

    sleep(0.05)
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
    assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert result == [False]


def test_audio_input_notifier_ignores_progressive_audio():
    notifier = _notifier()
    audio = np.zeros(1600, dtype=np.float32)

    assert not notifier.should_process_input(VADAudio(audio=audio, mode="progressive"))


def test_audio_input_notifier_discards_stale_pending_vad_measurement():
    revisions = SpeculativeTurnTracker()
    revisions.observe("turn_1", 1)
    notifier = _notifier(speculative_turns=revisions)
    store = TurnLatencyStore()
    notifier.turn_latency_store = store
    store.get_or_create_for_turn("turn_1", 0).vad_decision_s = 0.3

    assert not notifier.should_process_input(
        VADAudio(audio=np.zeros(1600, dtype=np.float32), mode="final", turn_id="turn_1", turn_revision=0)
    )
    assert store._pending_turn == {}


def test_audio_input_notifier_routes_final_audio_through_realtime_service_queue():
    text_output_queue = Queue()
    notifier = _notifier(text_output_queue=text_output_queue)
    audio = np.zeros(40000, dtype=np.float32)

    outputs = list(
        notifier.process(
            VADAudio(
                audio=audio,
                mode="final",
                turn_id="turn_1",
                turn_revision=2,
            )
        )
    )

    assert outputs == []
    event = text_output_queue.get_nowait()
    assert isinstance(event, AudioInputCompletedEvent)
    assert np.array_equal(event.audio, audio)
    assert event.audio_sample_rate == 16000
    assert event.audio_duration_s == 2.5
    assert event.turn_id == "turn_1"
    assert event.turn_revision == 2


def test_notifier_does_not_restart_expired_tracker_processing_delay(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("speech_to_speech.pipeline.speculative_turns.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("speech_to_speech.LLM.audio_input_notifier.monotonic", lambda: clock[0])
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    tracker.segment_finalized(1000, output_hold_ms=2000, processing_delay_ms=600)
    clock[0] = 10.7
    # Audio processing can finish after the original debounce deadline. Its
    # later message creation must not start a second wait from that point.
    item = VADAudio(
        audio=np.zeros(1600, dtype=np.float32),
        mode="final",
        turn_id="turn_1",
        turn_revision=0,
        processing_delay_s=0.6,
    )
    notifier = _notifier(speculative_turns=tracker)
    result = []
    thread = Thread(target=lambda: result.append(notifier.should_process_input(item)))
    thread.start()
    thread.join(timeout=0.2)
    completed_without_new_wait = not thread.is_alive()
    # Always release a failed baseline waiter before asserting.
    tracker.start_turn()
    thread.join(timeout=1.0)
    assert completed_without_new_wait
    assert result == [True]
