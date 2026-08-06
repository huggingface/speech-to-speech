from queue import Queue
from threading import Thread
from time import sleep

import numpy as np

from speech_to_speech.LLM.audio_input_notifier import AudioInputNotifier
from speech_to_speech.pipeline.events import AudioInputCompletedEvent
from speech_to_speech.pipeline.messages import VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker


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
