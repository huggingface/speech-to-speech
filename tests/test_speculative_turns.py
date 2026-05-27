import time
from queue import Queue
from threading import Event, Thread
from typing import Literal

import numpy as np
import torch

from speech_to_speech.pipeline.events import SpeechStartedEvent, SpeechStoppedEvent
from speech_to_speech.pipeline.messages import VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.VAD.vad_handler import VADHandler


def test_pending_reopen_defers_commit_until_cancelled():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)

    tracker.commit("turn_1", 0)

    assert candidate_revision == 1
    assert not tracker.is_committed("turn_1", 0)

    tracker.cancel_reopen_candidate("turn_1", candidate_revision)
    tracker.commit("turn_1", 0)

    assert tracker.is_committed("turn_1", 0)


def test_confirmed_reopen_makes_previous_revision_stale():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)

    assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)

    assert not tracker.is_latest("turn_1", 0)
    assert tracker.is_latest("turn_1", 1)


def test_tracker_prunes_old_turn_revisions():
    tracker = SpeculativeTurnTracker(max_tracked_turns=2)
    tracker.observe("turn_1", 0)
    tracker.commit("turn_1", 0)
    tracker.observe("turn_2", 0)
    tracker.observe("turn_3", 0)

    assert list(tracker._latest_revision) == ["turn_2", "turn_3"]
    assert "turn_1" not in tracker._committed_revision


def test_tracker_keeps_pending_reopen_while_pruning():
    tracker = SpeculativeTurnTracker(max_tracked_turns=1)
    tracker.observe("turn_1", 0)
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)

    tracker.observe("turn_2", 0)

    assert candidate_revision == 1
    assert "turn_1" in tracker._latest_revision
    assert "turn_1" in tracker._pending_reopen
    assert "turn_2" in tracker._latest_revision


def test_pending_reopen_wait_timeout_clears_candidate():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)

    tracker.wait_for_pending_reopen("turn_1", 0, timeout_s=0)

    assert candidate_revision == 1
    assert tracker._pending_reopen == {}


def test_commit_if_latest_waits_for_pending_reopen_and_drops_confirmed_reopen():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)

    assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)
    assert not tracker.commit_if_latest_after_pending_reopen("turn_1", 0)
    assert not tracker.is_committed("turn_1", 0)


def test_commit_if_latest_commits_after_pending_reopen_is_cancelled():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)

    tracker.cancel_reopen_candidate("turn_1", candidate_revision)

    assert tracker.commit_if_latest_after_pending_reopen("turn_1", 0)
    assert tracker.is_committed("turn_1", 0)


def test_try_is_latest_after_pending_reopen_reports_pending_without_blocking():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)

    assert tracker.has_pending_reopen("turn_1", 0)
    assert tracker.try_is_latest_after_pending_reopen("turn_1", 0) is None
    assert tracker.try_commit_if_latest_after_pending_reopen("turn_1", 0) is None

    tracker.cancel_reopen_candidate("turn_1", candidate_revision)

    assert tracker.try_is_latest_after_pending_reopen("turn_1", 0) is True
    assert tracker.try_commit_if_latest_after_pending_reopen("turn_1", 0) is True
    assert tracker.is_committed("turn_1", 0)


def test_try_is_latest_after_reopen_grace_reports_pending_without_blocking():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    tracker.start_reopen_grace("turn_1", 0, grace_s=0.05)

    assert tracker.try_is_latest_after_reopen_grace("turn_1", 0) is None
    assert tracker.try_commit_if_latest_after_reopen_grace("turn_1", 0) is None

    time.sleep(0.06)

    assert tracker.try_is_latest_after_reopen_grace("turn_1", 0) is True
    assert tracker.try_commit_if_latest_after_reopen_grace("turn_1", 0) is True
    assert tracker.is_committed("turn_1", 0)


def test_reopen_grace_wait_drops_confirmed_reopen():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    tracker.start_reopen_grace("turn_1", 0, grace_s=0.2)
    result: dict[str, bool] = {}

    def wait_for_grace():
        result["is_latest"] = tracker.is_latest_after_reopen_grace("turn_1", 0)

    thread = Thread(target=wait_for_grace)
    thread.start()

    time.sleep(0.02)
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
    assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert result == {"is_latest": False}


def test_is_latest_after_stability_window_catches_reopen_started_during_wait():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)

    def reopen_turn():
        time.sleep(0.02)
        candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
        assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)

    thread = Thread(target=reopen_turn)
    thread.start()

    assert not tracker.is_latest_after_stability_window("turn_1", 0, settle_s=0.2)
    thread.join(timeout=1.0)


def test_vad_direct_reopen_path_uses_tracker_candidate_protocol():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    handler = object.__new__(VADHandler)
    handler.enable_realtime_transcription = True
    handler._speech_started_emitted = False
    handler._current_turn_id = "turn_1"
    handler._current_turn_revision = 0
    handler._last_final_audio_ms = 1000
    handler.speculative_reopen_ms = 1200
    handler.speculative_turns = tracker
    handler._pending_reopen_candidate = None

    turn_id, revision, reopened = handler._ensure_turn_for_speech_start(1100)

    assert (turn_id, revision, reopened) == ("turn_1", 1, True)
    assert not tracker.is_latest("turn_1", 0)
    assert tracker.is_latest("turn_1", 1)
    assert tracker._pending_reopen == {}


def test_vad_reopens_speculative_turn_when_live_transcription_disabled():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    handler = object.__new__(VADHandler)
    handler.enable_realtime_transcription = False
    handler._speech_started_emitted = False
    handler._current_turn_id = "turn_1"
    handler._current_turn_revision = 0
    handler._last_final_audio_ms = 1000
    handler.speculative_reopen_ms = 1200
    handler.speculative_turns = tracker
    handler._pending_reopen_candidate = None

    turn_id, revision, reopened = handler._ensure_turn_for_speech_start(1100)

    assert (turn_id, revision, reopened) == ("turn_1", 1, True)
    assert not tracker.is_latest("turn_1", 0)
    assert tracker.is_latest("turn_1", 1)


def test_vad_realtime_path_does_not_emit_progressive_when_live_transcription_disabled():
    class FakeIterator:
        buffer = [object()]

    handler = object.__new__(VADHandler)
    handler.enable_realtime_transcription = False
    handler.iterator = FakeIterator()

    assert list(handler._process_realtime(None)) == []


class _StaticVADIterator:
    def __init__(
        self,
        *,
        triggered: bool,
        vad_output: list[torch.Tensor] | None,
        buffer_chunks: list[torch.Tensor] | None = None,
        speech_chunks: list[torch.Tensor] | None = None,
        voiced_samples: int = 0,
        last_utterance_voiced_samples: int = 0,
    ) -> None:
        self.triggered = triggered
        self._vad_output = vad_output
        self.buffer = buffer_chunks or []
        self._speech_chunks = speech_chunks or self.buffer
        self.voiced_samples = voiced_samples
        self.last_utterance_voiced_samples = last_utterance_voiced_samples

    def __call__(self, _chunk: torch.Tensor) -> list[torch.Tensor] | None:
        return self._vad_output

    def speech_buffer(self) -> list[torch.Tensor]:
        return self._speech_chunks


def _vad_handler_for_iterator(iterator: _StaticVADIterator) -> VADHandler:
    handler = object.__new__(VADHandler)
    handler.should_listen = Event()
    handler.should_listen.set()
    handler.sample_rate = 16000
    handler.min_silence_ms = 300
    handler.min_speech_ms = 500
    handler.max_speech_ms = float("inf")
    handler.enable_realtime_transcription = False
    handler.realtime_processing_pause = 0.5
    handler.text_output_queue = Queue()
    handler.speculative_turns = SpeculativeTurnTracker()
    handler.speculative_reopen_ms = 1000
    handler._last_turn_detection = None
    handler.iterator = iterator
    handler.audio_enhancement = False
    handler.last_process_time = 0.0
    handler._total_samples = 0
    handler._last_log_time = time.time()
    handler._log_chunks = 0
    handler._log_speech_starts = 0
    handler._log_speech_ends = 0
    handler._log_progressive_yields = 0
    handler._speech_started_emitted = False
    handler._turn_counter = 0
    handler._current_turn_id = None
    handler._current_turn_revision = None
    handler._speculative_audio_prefix = None
    handler._last_final_wall_time = None
    handler._last_final_audio_ms = None
    handler._pending_reopen_candidate = None
    return handler


def _audio_bytes(samples: int = 512) -> bytes:
    return np.zeros(samples, dtype=np.int16).tobytes()


def test_vad_interruption_uses_voiced_duration_not_padded_segment():
    chunks = [torch.zeros(512) for _ in range(20)]
    iterator = _StaticVADIterator(
        triggered=True,
        vad_output=None,
        buffer_chunks=chunks,
        speech_chunks=chunks,
        voiced_samples=10 * 512,
    )
    handler = _vad_handler_for_iterator(iterator)

    assert list(handler.process(_audio_bytes())) == []

    assert handler.text_output_queue.empty()
    assert handler._speech_started_emitted is False


def test_vad_interruption_emits_after_voiced_threshold():
    chunks = [torch.zeros(512) for _ in range(20)]
    iterator = _StaticVADIterator(
        triggered=True,
        vad_output=None,
        buffer_chunks=chunks,
        speech_chunks=chunks,
        voiced_samples=16 * 512,
    )
    handler = _vad_handler_for_iterator(iterator)

    assert list(handler.process(_audio_bytes())) == []

    event = handler.text_output_queue.get_nowait()
    assert isinstance(event, SpeechStartedEvent)
    assert event.interrupt_response is True
    assert handler._speech_started_emitted is True


def test_vad_final_synthetic_start_does_not_interrupt_response():
    final_chunks = [torch.zeros(512) for _ in range(31)]
    iterator = _StaticVADIterator(
        triggered=False,
        vad_output=final_chunks,
        last_utterance_voiced_samples=2 * 512,
    )
    handler = _vad_handler_for_iterator(iterator)

    outputs = list(handler.process(_audio_bytes()))

    assert len(outputs) == 1
    started = handler.text_output_queue.get_nowait()
    stopped = handler.text_output_queue.get_nowait()
    assert isinstance(started, SpeechStartedEvent)
    assert started.interrupt_response is False
    assert isinstance(stopped, SpeechStoppedEvent)


def test_vad_keeps_single_speculative_audio_prefix():
    handler = object.__new__(VADHandler)
    handler._speculative_audio_prefix = None
    first_segment = np.array([1.0, 2.0], dtype=np.float32)
    second_segment = np.array([3.0], dtype=np.float32)
    third_segment = np.array([4.0], dtype=np.float32)

    first_output = handler._combined_turn_audio(first_segment)
    handler._speculative_audio_prefix = first_output
    second_output = handler._combined_turn_audio(second_segment)
    handler._speculative_audio_prefix = second_output
    third_output = handler._combined_turn_audio(third_segment)

    assert first_output is first_segment
    np.testing.assert_array_equal(second_output, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(third_output, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))


def _vad_audio(
    turn_id: str = "turn_1",
    revision: int = 0,
    mode: Literal["progressive", "final"] | None = "progressive",
) -> VADAudio:
    return VADAudio(audio=np.zeros(512, dtype=np.float32), mode=mode, turn_id=turn_id, turn_revision=revision)


def test_vad_drops_superseded_progressive_audio_from_output_queue():
    handler = object.__new__(VADHandler)
    handler.queue_out = Queue()
    handler.speculative_turns = None
    first_progressive = _vad_audio()
    final_audio = _vad_audio(mode="final")
    second_progressive = _vad_audio()
    other_turn_progressive = _vad_audio(turn_id="turn_2")

    handler.queue_out.put(first_progressive)
    handler.queue_out.put(final_audio)
    handler.queue_out.put(second_progressive)
    handler.queue_out.put(other_turn_progressive)

    dropped = handler._drop_superseded_vad_audio(_vad_audio())

    assert dropped == 2
    queued_items = list(handler.queue_out.queue)
    assert queued_items == [final_audio, other_turn_progressive]


def test_vad_drops_stale_progressive_revisions_from_output_queue():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 1)
    handler = object.__new__(VADHandler)
    handler.queue_out = Queue()
    handler.speculative_turns = tracker
    stale_progressive = _vad_audio(revision=0)
    current_progressive = _vad_audio(revision=1)

    handler.queue_out.put(stale_progressive)
    handler.queue_out.put(current_progressive)

    handler.before_emit_output(_vad_audio(revision=1))

    assert handler.queue_out.empty()


def test_vad_final_audio_replaces_queued_progressive_audio_for_same_revision():
    handler = object.__new__(VADHandler)
    handler.queue_out = Queue()
    handler.speculative_turns = None
    progressive_audio = _vad_audio()
    final_audio = _vad_audio(mode="final")
    other_turn_progressive = _vad_audio(turn_id="turn_2")

    handler.queue_out.put(progressive_audio)
    handler.queue_out.put(other_turn_progressive)

    handler.before_emit_output(final_audio)

    queued_items = list(handler.queue_out.queue)
    assert queued_items == [other_turn_progressive]


def test_vad_progressive_processing_pause_increases_with_speech_duration():
    handler = object.__new__(VADHandler)
    handler.realtime_processing_pause = 0.25

    assert handler._progressive_processing_pause(7_999) == 0.25
    assert handler._progressive_processing_pause(8_000) == 0.5
    assert handler._progressive_processing_pause(15_000) == 1.0
    assert handler._progressive_processing_pause(30_000) == 1.5


def test_vad_progressive_processing_pause_is_capped():
    handler = object.__new__(VADHandler)
    handler.realtime_processing_pause = 0.5

    assert handler._progressive_processing_pause(30_000) == 2.0
