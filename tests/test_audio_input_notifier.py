from queue import Queue
from threading import Event, Thread
from time import sleep

import numpy as np

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.audio_input_notifier import AudioInputNotifier
from speech_to_speech.pipeline.events import AudioInputCompletedEvent
from speech_to_speech.pipeline.messages import GenerateResponseRequest, VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker


def _notifier(
    runtime_config: RuntimeConfig | None = None,
    text_output_queue: Queue | None = None,
    speculative_turns: SpeculativeTurnTracker | None = None,
) -> AudioInputNotifier:
    notifier = object.__new__(AudioInputNotifier)
    notifier.setup(
        runtime_config=runtime_config,
        should_listen=Event(),
        sample_rate=16000,
        speculative_turns=speculative_turns,
        text_output_queue=text_output_queue,
    )
    return notifier


def test_audio_input_notifier_uses_per_endpoint_processing_delay():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    notifier = _notifier(RuntimeConfig(), speculative_turns=tracker)
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
    notifier = _notifier(RuntimeConfig())
    audio = np.zeros(1600, dtype=np.float32)

    assert not notifier.should_process_input(VADAudio(audio=audio, mode="progressive"))


def test_audio_input_notifier_forwards_final_audio_to_llm_request():
    runtime_config = RuntimeConfig()
    notifier = _notifier(runtime_config)
    audio = np.zeros(1600, dtype=np.float32)

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

    assert len(outputs) == 1
    request = outputs[0]
    assert isinstance(request, GenerateResponseRequest)
    assert request.runtime_config is runtime_config
    assert np.array_equal(request.audio, audio)
    assert request.audio_sample_rate == 16000
    assert request.turn_id == "turn_1"
    assert request.turn_revision == 2


def test_audio_input_notifier_prefers_runtime_config_from_vad_audio():
    setup_config = RuntimeConfig()
    item_config = RuntimeConfig()
    notifier = _notifier(setup_config)
    audio = np.zeros(1600, dtype=np.float32)

    request = next(notifier.process(VADAudio(audio=audio, runtime_config=item_config, mode="final")))

    assert request.runtime_config is item_config


def test_audio_input_notifier_routes_realtime_audio_through_service_queue():
    text_output_queue = Queue()
    notifier = _notifier(text_output_queue=text_output_queue)
    audio = np.zeros(40000, dtype=np.float32)

    outputs = list(
        notifier.process(
            VADAudio(
                audio=audio,
                runtime_config=RuntimeConfig(),
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
