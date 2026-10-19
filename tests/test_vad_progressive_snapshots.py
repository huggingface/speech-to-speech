from queue import Queue
from threading import Event

import numpy as np
import pytest
import torch

from speech_to_speech.pipeline.events import SpeechStartedEvent, SpeechStoppedEvent
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.VAD.vad_handler import VADHandler
from tests.test_vad_iterator import _FakeVADModel

SR = 16000
CHUNK = 512
INTERVAL = 16 * CHUNK


@pytest.fixture
def vad(monkeypatch):
    def create(probabilities, **settings):
        monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: (_FakeVADModel(probabilities), None))
        listen = Event()
        listen.set()
        return VADHandler(
            Event(),
            Queue(),
            Queue(),
            setup_kwargs={
                "should_listen": listen,
                "speculative_turns": SpeculativeTurnTracker(),
                "text_output_queue": Queue(),
                "speech_pad_ms": 0,
                "smart_turn": False,
                **settings,
            },
        )

    return create


def feed(handler, count, *, value=8000):
    chunk = np.full(handler.sample_rate * 32 // 1000, value, dtype=np.int16).tobytes()
    return [output for _ in range(count) for output in handler.process(chunk)]


@pytest.mark.parametrize("sample_rate", [8000, SR])
def test_periodic_snapshots_keep_capturing_until_the_normal_speech_end(vad, sample_rate):
    handler = vad([0.9] * 48 + [0.1] * 3, sample_rate=sample_rate)
    chunk_samples = sample_rate * 32 // 1000
    handler.transcription_window_samples = 16 * chunk_samples

    snapshots = feed(handler, 48)

    assert [output.mode for output in snapshots] == ["progressive"] * 3
    assert [len(output.audio) for output in snapshots] == [16 * chunk_samples, 32 * chunk_samples, 48 * chunk_samples]
    assert not handler.enable_realtime_transcription
    assert handler.iterator.triggered
    assert sum(len(chunk) for chunk in handler.iterator.speech_buffer()) == 48 * chunk_samples
    assert handler.speculative_turns.current_turn() == (snapshots[0].turn_id, snapshots[0].turn_revision)
    started = handler.text_output_queue.get_nowait()
    assert isinstance(started, SpeechStartedEvent)
    assert handler.text_output_queue.empty()  # No speech-stop or caption events at a snapshot.

    (final,) = feed(handler, 3, value=0)

    assert final.mode == "final"
    assert len(final.audio) == 51 * chunk_samples
    assert not handler.iterator.triggered
    for snapshot in snapshots:
        assert (snapshot.turn_id, snapshot.turn_revision) == (final.turn_id, final.turn_revision)
        np.testing.assert_array_equal(snapshot.audio, final.audio[: len(snapshot.audio)])
    assert isinstance(handler.text_output_queue.get_nowait(), SpeechStoppedEvent)
    assert handler.text_output_queue.empty()


def test_no_periodic_snapshots_are_emitted_by_default(vad):
    handler = vad([0.9] * 48 + [0.1] * 3)

    assert handler.transcription_window_samples is None
    assert feed(handler, 48) == []
    (final,) = feed(handler, 3, value=0)
    assert final.mode == "final"
    assert len(final.audio) == 51 * CHUNK


def test_held_short_fragment_can_be_added_to_the_prefix_only_at_finalization(vad):
    handler = vad([0.9] * 6 + [0.1] * 3 + [0.9] * 16 + [0.1] * 3, short_segment_merge_ms=384)
    handler.transcription_window_samples = INTERVAL

    assert feed(handler, 6) == []
    assert feed(handler, 3, value=0) == []
    (snapshot,) = feed(handler, 16, value=12000)
    (final,) = feed(handler, 3, value=0)

    assert snapshot.mode == "progressive" and final.mode == "final"
    assert (snapshot.turn_id, snapshot.turn_revision) == (final.turn_id, final.turn_revision)
    assert len(snapshot.audio) == INTERVAL
    assert len(final.audio) == 28 * CHUNK
    np.testing.assert_array_equal(snapshot.audio, final.audio[9 * CHUNK : 25 * CHUNK])
    assert not np.array_equal(snapshot.audio, final.audio[: len(snapshot.audio)])


def test_snapshot_cadence_preserves_minimum_valid_speech(vad):
    handler = vad([0.9] * 12)
    handler.transcription_window_samples = 4 * CHUNK  # Less than the 384 ms speech threshold.

    assert feed(handler, 11) == []
    (snapshot,) = feed(handler, 1)
    assert snapshot.mode == "progressive"
    assert len(snapshot.audio) == 12 * CHUNK
    assert handler.iterator.triggered


def test_periodic_snapshots_coexist_with_live_transcription_updates(vad, monkeypatch):
    now = [0.0]
    monkeypatch.setattr("speech_to_speech.VAD.vad_handler.time.time", lambda: now[0])
    handler = vad([0.9] * 33, enable_realtime_transcription=True)
    handler.transcription_window_samples = INTERVAL

    (first,) = feed(handler, 16)
    assert len(first.audio) == INTERVAL
    now[0] = 0.5
    (live,) = feed(handler, 1)
    assert len(live.audio) == 17 * CHUNK
    assert feed(handler, 15) == []
    (periodic,) = feed(handler, 1)  # Another interval, with no wall-clock time advanced.
    assert len(periodic.audio) == 33 * CHUNK
    assert all(output.mode == "progressive" for output in (first, live, periodic))
    assert len({(output.turn_id, output.turn_revision) for output in (first, live, periodic)}) == 1


def test_reopened_turn_preserves_snapshot_cadence_and_audio_prefix(vad):
    handler = vad([0.9] * 16 + [0.1] * 3 + [0.9] * 13)
    handler.transcription_window_samples = INTERVAL
    (first,) = feed(handler, 16)
    (final,) = feed(handler, 3, value=0)

    assert feed(handler, 12, value=12000) == []
    (resumed,) = feed(handler, 1, value=12000)

    assert resumed.mode == "progressive"
    assert resumed.turn_id == first.turn_id == final.turn_id
    assert resumed.turn_revision == final.turn_revision + 1
    assert len(resumed.audio) == 2 * INTERVAL
    np.testing.assert_array_equal(resumed.audio[: len(final.audio)], final.audio)
    assert handler.iterator.triggered


def test_new_turn_resets_snapshot_cadence_and_audio_prefix(vad):
    handler = vad([0.9] * 16 + [0.1] * 3 + [0.9] * 16)
    handler.transcription_window_samples = INTERVAL
    feed(handler, 16)
    (final,) = feed(handler, 3, value=0)
    assert handler.speculative_turns.wait_for_gate(
        final.turn_id, final.turn_revision, hold_for_grace=False, commit=True
    )

    assert feed(handler, 12, value=12000) == []
    assert handler._last_window_snapshot_samples == 0
    assert feed(handler, 3, value=12000) == []
    (snapshot,) = feed(handler, 1, value=12000)

    assert snapshot.turn_id != final.turn_id
    assert len(snapshot.audio) == INTERVAL
    np.testing.assert_array_equal(snapshot.audio, np.full(INTERVAL, 12000 / 32768, dtype=np.float32))


def test_session_reset_clears_snapshot_progress_but_keeps_configured_cadence(vad):
    handler = vad([0.9] * 32)
    handler.transcription_window_samples = INTERVAL
    feed(handler, 16)

    handler.on_session_end()

    assert handler._last_window_snapshot_samples == handler._window_check_after_samples == 0
    assert handler.transcription_window_samples == INTERVAL
    assert handler.iterator.speech_buffer() == []
    assert feed(handler, 15, value=12000) == []
    (snapshot,) = feed(handler, 1, value=12000)
    assert len(snapshot.audio) == INTERVAL
    np.testing.assert_array_equal(snapshot.audio, np.full(INTERVAL, 12000 / 32768, dtype=np.float32))


def test_snapshot_checks_do_not_recount_the_buffer_for_every_audio_chunk(vad, monkeypatch):
    handler = vad([0.9] * 32)
    handler.transcription_window_samples = INTERVAL
    feed(handler, 16)
    recounts = []
    duration = handler._speech_buffer_duration_ms

    def counted_duration():
        recounts.append(None)
        return duration()

    monkeypatch.setattr(handler, "_speech_buffer_duration_ms", counted_duration)

    assert feed(handler, 15) == []
    assert len(recounts) == 1
    (snapshot,) = feed(handler, 1)
    assert len(recounts) == 2
    assert len(snapshot.audio) == 2 * INTERVAL
