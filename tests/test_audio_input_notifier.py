from queue import Queue
from threading import Thread
from time import sleep

import numpy as np

from speech_to_speech.LLM.audio_input_notifier import AudioInputNotifier
from speech_to_speech.pipeline.events import AudioInputCompletedEvent
from speech_to_speech.pipeline.messages import VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.pipeline.turn_latency import TurnLatencyStore
from tests.turns import reopen


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


def test_audio_input_notifier_ignores_progressive_audio():
    notifier = _notifier()
    audio = np.zeros(1600, dtype=np.float32)

    assert not notifier.should_process_input(VADAudio(audio=audio, mode="progressive"))


def test_audio_input_notifier_discards_stale_pending_vad_measurement():
    revisions = SpeculativeTurnTracker()
    revisions.start_turn()
    reopen(revisions)
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


def _run_notifier_smart_turn(tracker: SpeculativeTurnTracker) -> tuple[object, Thread, list[bool]]:
    tracker.start_turn()
    token = tracker.segment_finalized(1000, pending_smart_turn_s=2.0)
    assert token is not None
    notifier = _notifier(speculative_turns=tracker)
    item = VADAudio(audio=np.zeros(1600, dtype=np.float32), mode="final", turn_id="turn_1", turn_revision=0)
    result: list[bool] = []
    thread = Thread(target=lambda: result.append(notifier.should_process_input(item)))
    thread.start()
    sleep(0.03)
    assert thread.is_alive()
    return token, thread, result


def test_audio_input_notifier_waits_for_pending_smart_turn_and_incomplete_delay():
    tracker = SpeculativeTurnTracker()
    token, thread, result = _run_notifier_smart_turn(tracker)
    assert tracker.resolve_smart_turn("turn_1", 0, output_hold_ms=2000, processing_delay_ms=250, token=token)
    sleep(0.04)
    assert thread.is_alive()
    reopen(tracker, "turn_1", 0)
    thread.join(timeout=1.0)
    assert not thread.is_alive() and result == [False]


def test_audio_input_notifier_proceeds_immediately_when_smart_turn_resolves_complete():
    tracker = SpeculativeTurnTracker()
    token, thread, result = _run_notifier_smart_turn(tracker)
    assert tracker.resolve_smart_turn("turn_1", 0, output_hold_ms=800, processing_delay_ms=0, token=token)
    thread.join(timeout=1.0)
    assert not thread.is_alive() and result == [True]
