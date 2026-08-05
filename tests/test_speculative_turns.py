import time
from queue import Queue
from threading import Event, Thread
from typing import Literal

import numpy as np
import pytest
import torch

from speech_to_speech.pipeline.events import SpeechStartedEvent, SpeechStoppedEvent
from speech_to_speech.pipeline.messages import VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.VAD.smart_turn import SmartTurnResult
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


def test_is_latest_after_stability_window_survives_cancelled_reopen_candidate():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    started = Event()
    result: list[bool] = []

    def wait_for_stability():
        started.set()
        result.append(tracker.is_latest_after_stability_window("turn_1", 0, settle_s=0.2))

    thread = Thread(target=wait_for_stability)
    thread.start()
    assert started.wait(timeout=1.0)

    time.sleep(0.02)
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
    time.sleep(0.02)
    tracker.cancel_reopen_candidate("turn_1", candidate_revision)

    time.sleep(0.03)
    assert thread.is_alive()
    assert result == []

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert result == [True]


def test_commit_after_reset_does_not_resurrect_untracked_turn():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    tracker.reset()

    tracker.commit("turn_1", 0)

    assert tracker._committed_revision == {}
    assert not tracker.is_committed("turn_1", 0)


def test_commit_after_prune_does_not_resurrect_untracked_turn():
    tracker = SpeculativeTurnTracker(max_tracked_turns=1)
    tracker.observe("turn_1", 0)
    tracker.observe("turn_2", 0)

    tracker.commit("turn_1", 0)

    assert list(tracker._latest_revision) == ["turn_2"]
    assert tracker._committed_revision == {}


@pytest.mark.parametrize(
    "commit_method",
    [
        "commit_if_latest_after_pending_reopen",
        "commit_if_latest_after_reopen_grace",
        "try_commit_if_latest_after_pending_reopen",
        "try_commit_if_latest_after_reopen_grace",
    ],
)
def test_commit_if_latest_variants_keep_untracked_turn_out_of_committed_state(commit_method):
    """An untracked turn still reports success -- callers treat `False` as "drop this
    output" -- but it must not be written back into `_committed_revision`."""
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    tracker.reset()

    assert getattr(tracker, commit_method)("turn_1", 0) is True
    assert tracker._committed_revision == {}


def test_reused_turn_id_after_reset_is_not_reported_as_committed():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    tracker.reset()
    tracker.commit("turn_1", 0)

    tracker.observe("turn_1", 0)

    assert not tracker.is_committed("turn_1", 0)
    assert tracker.begin_reopen_candidate("turn_1", 0) == 1


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
    handler.unanswered_reopen_ms = 1200
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
    handler.unanswered_reopen_ms = 1200
    handler.speculative_turns = tracker
    handler._pending_reopen_candidate = None

    turn_id, revision, reopened = handler._ensure_turn_for_speech_start(1100)

    assert (turn_id, revision, reopened) == ("turn_1", 1, True)
    assert not tracker.is_latest("turn_1", 0)
    assert tracker.is_latest("turn_1", 1)


def test_vad_starts_new_turn_after_committed_turn_would_have_reopened():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    tracker.commit("turn_1", 0)
    handler = object.__new__(VADHandler)
    handler.enable_realtime_transcription = False
    handler._speech_started_emitted = False
    handler._current_turn_id = "turn_1"
    handler._current_turn_revision = 0
    handler._turn_counter = 1
    handler._last_final_audio_ms = 1000
    handler.speculative_reopen_ms = 1200
    handler.speculative_turns = tracker
    handler._pending_reopen_candidate = None

    turn_id, revision, reopened = handler._ensure_turn_for_speech_start(1100)

    assert (turn_id, revision, reopened) == ("turn_2", 0, False)
    assert tracker.is_committed("turn_1", 0)
    assert tracker.is_latest("turn_2", 0)


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
        active_speech_samples: int = 0,
        last_utterance_active_speech_samples: int = 0,
    ) -> None:
        self.triggered = triggered
        self._vad_output = vad_output
        self.buffer = buffer_chunks or []
        self._speech_chunks = speech_chunks or self.buffer
        self.active_speech_samples = active_speech_samples
        self.last_utterance_active_speech_samples = last_utterance_active_speech_samples

    def __call__(self, _chunk: torch.Tensor) -> list[torch.Tensor] | None:
        return self._vad_output

    def speech_buffer(self) -> list[torch.Tensor]:
        return self._speech_chunks


class _StaticSmartTurnAnalyzer:
    def __init__(self, *results: SmartTurnResult) -> None:
        self._results = iter(results)
        self.calls: list[np.ndarray] = []

    def predict(self, audio: np.ndarray, *, sample_rate: int) -> SmartTurnResult:
        assert sample_rate == 16000
        self.calls.append(audio.copy())
        return next(self._results)


def _vad_handler_for_iterator(iterator: _StaticVADIterator) -> VADHandler:
    handler = object.__new__(VADHandler)
    handler.should_listen = Event()
    handler.should_listen.set()
    handler.sample_rate = 16000
    handler.min_silence_ms = 300
    handler.min_speech_ms = 384
    handler.min_speech_continuation_ms = handler.min_speech_ms
    handler.max_speech_ms = float("inf")
    handler.enable_realtime_transcription = False
    handler.realtime_processing_pause = 0.5
    handler.text_output_queue = Queue()
    handler.speculative_turns = SpeculativeTurnTracker()
    handler.speculative_reopen_ms = 800
    handler.unanswered_reopen_ms = 7000
    handler._last_turn_detection = None
    handler.smart_turn_analyzer = None
    handler.smart_turn_max_wait_ms = 2000
    handler.smart_turn_incomplete_delay_ms = 600
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
    handler._speculative_raw_audio_prefix = None
    handler._last_final_wall_time = None
    handler._last_final_audio_ms = None
    handler._pending_reopen_candidate = None
    handler.short_segment_merge_ms = 0
    handler._pending_short_segment = None
    return handler


def _audio_bytes(samples: int = 512) -> bytes:
    return np.zeros(samples, dtype=np.int16).tobytes()


def _drain_text_events(handler: VADHandler) -> None:
    while not handler.text_output_queue.empty():
        handler.text_output_queue.get_nowait()


def test_vad_interruption_uses_active_speech_duration_not_padded_segment():
    chunks = [torch.zeros(512) for _ in range(20)]
    iterator = _StaticVADIterator(
        triggered=True,
        vad_output=None,
        buffer_chunks=chunks,
        speech_chunks=chunks,
        active_speech_samples=10 * 512,
    )
    handler = _vad_handler_for_iterator(iterator)
    handler.enable_realtime_transcription = True

    assert list(handler.process(_audio_bytes())) == []

    assert handler.text_output_queue.empty()
    assert handler._speech_started_emitted is False


def test_vad_pending_reopen_starts_before_active_speech_threshold():
    chunks = [torch.zeros(512) for _ in range(12)]
    iterator = _StaticVADIterator(
        triggered=True,
        vad_output=None,
        buffer_chunks=chunks,
        speech_chunks=chunks,
        active_speech_samples=8 * 512,
    )
    handler = _vad_handler_for_iterator(iterator)
    tracker = handler.speculative_turns
    tracker.observe("turn_1", 0)
    handler._current_turn_id = "turn_1"
    handler._current_turn_revision = 0
    handler._last_final_audio_ms = 0

    assert list(handler.process(_audio_bytes())) == []

    assert tracker.has_pending_reopen("turn_1", 0)
    tracker.commit("turn_1", 0)
    assert not tracker.is_committed("turn_1", 0)
    assert handler.text_output_queue.empty()
    assert handler._speech_started_emitted is False


def test_vad_interruption_emits_after_active_speech_threshold():
    chunks = [torch.zeros(512) for _ in range(20)]
    iterator = _StaticVADIterator(
        triggered=True,
        vad_output=None,
        buffer_chunks=chunks,
        speech_chunks=chunks,
        active_speech_samples=12 * 512,
    )
    handler = _vad_handler_for_iterator(iterator)

    assert list(handler.process(_audio_bytes())) == []

    event = handler.text_output_queue.get_nowait()
    assert isinstance(event, SpeechStartedEvent)
    assert event.interrupt_response is True
    assert handler._speech_started_emitted is True


def test_vad_live_transcription_without_speculative_turns_stops_listening_on_final():
    final_chunks = [torch.zeros(512) for _ in range(31)]
    iterator = _StaticVADIterator(
        triggered=False,
        vad_output=final_chunks,
        last_utterance_active_speech_samples=12 * 512,
    )
    handler = _vad_handler_for_iterator(iterator)
    handler.enable_realtime_transcription = True
    handler.speculative_turns = None

    outputs = list(handler.process(_audio_bytes()))

    assert len(outputs) == 1
    assert not handler.should_listen.is_set()


def test_vad_discards_final_segment_when_active_speech_is_short():
    final_chunks = [torch.zeros(512) for _ in range(31)]
    iterator = _StaticVADIterator(
        triggered=False,
        vad_output=final_chunks,
        last_utterance_active_speech_samples=11 * 512,
    )
    handler = _vad_handler_for_iterator(iterator)

    outputs = list(handler.process(_audio_bytes()))

    assert outputs == []
    assert handler.text_output_queue.empty()


def _drive_final_segment(handler: VADHandler, active_chunks: int = 12, segment_chunks: int = 31) -> list:
    handler.iterator = _StaticVADIterator(
        triggered=False,
        vad_output=[torch.zeros(512) for _ in range(segment_chunks)],
        last_utterance_active_speech_samples=active_chunks * 512,
    )
    return list(handler.process(_audio_bytes()))


def test_vad_complete_smart_turn_selects_shorter_speculative_grace():
    handler = _vad_handler_for_iterator(_StaticVADIterator(triggered=False, vad_output=None))
    handler.smart_turn_analyzer = _StaticSmartTurnAnalyzer(
        SmartTurnResult(complete=True, probability=0.98, inference_ms=12.5)
    )

    outputs = _drive_final_segment(handler)

    assert len(outputs) == 1
    assert outputs[0].processing_delay_s == 0.0
    grace = handler.speculative_turns._reopen_grace["turn_1"]
    assert grace.revision == 0
    assert 0.6 < grace.deadline - time.monotonic() <= 0.8


def test_vad_incomplete_smart_turn_selects_longer_speculative_grace():
    handler = _vad_handler_for_iterator(_StaticVADIterator(triggered=False, vad_output=None))
    analyzer = _StaticSmartTurnAnalyzer(SmartTurnResult(complete=False, probability=0.2, inference_ms=12.5))
    handler.smart_turn_analyzer = analyzer

    outputs = _drive_final_segment(handler)

    assert len(outputs) == 1
    assert outputs[0].processing_delay_s == 0.6
    grace = handler.speculative_turns._reopen_grace["turn_1"]
    assert grace.revision == 0
    assert 1.8 < grace.deadline - time.monotonic() <= 2.0
    np.testing.assert_array_equal(analyzer.calls[0], outputs[0].audio)


def test_vad_incomplete_smart_turn_commits_after_longer_grace_without_resumed_speech():
    handler = _vad_handler_for_iterator(_StaticVADIterator(triggered=False, vad_output=None))
    handler.smart_turn_max_wait_ms = 50
    handler.smart_turn_analyzer = _StaticSmartTurnAnalyzer(
        SmartTurnResult(complete=False, probability=0.2, inference_ms=12.5)
    )

    outputs = _drive_final_segment(handler)

    assert len(outputs) == 1
    assert handler.speculative_turns.try_commit_if_latest_after_reopen_grace("turn_1", 0) is None
    time.sleep(0.06)
    assert handler.speculative_turns.try_commit_if_latest_after_reopen_grace("turn_1", 0) is True
    assert handler.speculative_turns.is_committed("turn_1", 0)


def test_vad_resumed_speech_during_smart_turn_grace_creates_new_revision():
    handler = _vad_handler_for_iterator(_StaticVADIterator(triggered=False, vad_output=None))
    analyzer = _StaticSmartTurnAnalyzer(
        SmartTurnResult(complete=False, probability=0.2, inference_ms=12.5),
        SmartTurnResult(complete=True, probability=0.9, inference_ms=12.5),
    )
    handler.smart_turn_analyzer = analyzer

    first_outputs = _drive_final_segment(handler)
    assert len(first_outputs) == 1
    assert (first_outputs[0].turn_id, first_outputs[0].turn_revision) == ("turn_1", 0)
    _drain_text_events(handler)

    # Resume after the normal 800ms grace but before Smart Turn's 2000ms
    # incomplete-turn grace has elapsed.
    handler._total_samples = int(1.5 * handler.sample_rate)
    resumed_outputs = _drive_final_segment(handler, active_chunks=12, segment_chunks=12)

    assert len(resumed_outputs) == 1
    assert (resumed_outputs[0].turn_id, resumed_outputs[0].turn_revision) == ("turn_1", 1)
    assert not handler.speculative_turns.is_latest("turn_1", 0)
    assert handler.speculative_turns.is_latest("turn_1", 1)
    assert len(analyzer.calls) == 2
    assert len(analyzer.calls[1]) == len(first_outputs[0].audio) + 12 * 512


def test_vad_reanalyzes_resumed_turn_with_raw_audio_after_enhancement():
    handler = _vad_handler_for_iterator(_StaticVADIterator(triggered=False, vad_output=None))
    analyzer = _StaticSmartTurnAnalyzer(
        SmartTurnResult(complete=False, probability=0.2, inference_ms=12.5),
        SmartTurnResult(complete=True, probability=0.9, inference_ms=12.5),
    )
    handler.smart_turn_analyzer = analyzer
    handler.audio_enhancement = True

    def enhance_audio(audio: np.ndarray) -> np.ndarray:
        audio += 1.0
        return audio

    handler._apply_audio_enhancement = enhance_audio

    first_outputs = _drive_final_segment(handler)
    assert len(first_outputs) == 1
    np.testing.assert_array_equal(first_outputs[0].audio, np.ones(31 * 512, dtype=np.float32))

    handler._total_samples = int(1.5 * handler.sample_rate)
    resumed_outputs = _drive_final_segment(handler, active_chunks=12, segment_chunks=12)

    assert len(resumed_outputs) == 1
    assert len(analyzer.calls) == 2
    np.testing.assert_array_equal(analyzer.calls[0], np.zeros(31 * 512, dtype=np.float32))
    np.testing.assert_array_equal(analyzer.calls[1], np.zeros(43 * 512, dtype=np.float32))
    np.testing.assert_array_equal(resumed_outputs[0].audio, np.ones(43 * 512, dtype=np.float32))


def test_vad_max_speech_is_enforced_before_smart_turn():
    handler = _vad_handler_for_iterator(_StaticVADIterator(triggered=False, vad_output=None))
    analyzer = _StaticSmartTurnAnalyzer(SmartTurnResult(complete=False, probability=0.2, inference_ms=12.5))
    handler.smart_turn_analyzer = analyzer
    handler.max_speech_ms = 1500

    assert _drive_final_segment(handler, segment_chunks=63) == []
    assert analyzer.calls == []


def _handler_after_soft_ended_turn() -> VADHandler:
    handler = _vad_handler_for_iterator(_StaticVADIterator(triggered=False, vad_output=None))
    outputs = _drive_final_segment(handler, active_chunks=12, segment_chunks=12)
    assert len(outputs) == 1
    assert (outputs[0].turn_id, outputs[0].turn_revision) == ("turn_1", 0)
    _drain_text_events(handler)
    return handler


def test_continuation_start_confirms_at_lower_bar():
    handler = _handler_after_soft_ended_turn()
    handler.min_speech_continuation_ms = 192
    chunks = [torch.zeros(512) for _ in range(8)]
    handler.iterator = _StaticVADIterator(
        triggered=True,
        vad_output=None,
        buffer_chunks=chunks,
        speech_chunks=chunks,
        active_speech_samples=8 * 512,
    )

    assert list(handler.process(_audio_bytes())) == []

    started = handler.text_output_queue.get_nowait()
    assert isinstance(started, SpeechStartedEvent)
    assert (started.turn_id, started.turn_revision, started.reopened) == ("turn_1", 1, True)
    assert handler._speech_started_emitted is True


def test_trailing_continuation_fragment_accepted_at_finalization():
    handler = _handler_after_soft_ended_turn()
    handler.min_speech_continuation_ms = 192

    outputs = _drive_final_segment(handler, active_chunks=8, segment_chunks=8)

    assert len(outputs) == 1
    assert (outputs[0].turn_id, outputs[0].turn_revision) == ("turn_1", 1)
    started = handler.text_output_queue.get_nowait()
    assert isinstance(started, SpeechStartedEvent)
    assert (started.turn_id, started.turn_revision, started.reopened) == ("turn_1", 1, True)


def test_continuation_bar_inactive_when_turn_committed():
    handler = _handler_after_soft_ended_turn()
    handler.min_speech_continuation_ms = 192
    tracker = handler.speculative_turns
    tracker.commit("turn_1", 0)

    outputs = _drive_final_segment(handler, active_chunks=8, segment_chunks=8)

    assert outputs == []
    assert handler.text_output_queue.empty()
    assert handler._current_turn_id == "turn_1"
    assert handler._current_turn_revision == 0
    assert tracker.is_committed("turn_1", 0)


def test_entry_bar_unchanged_for_new_speech():
    handler = _vad_handler_for_iterator(_StaticVADIterator(triggered=False, vad_output=None))
    handler.min_speech_continuation_ms = 192

    outputs = _drive_final_segment(handler, active_chunks=8, segment_chunks=8)

    assert outputs == []
    assert handler.text_output_queue.empty()
    assert handler._current_turn_id is None
    assert handler._turn_counter == 0


def test_confirmed_segment_not_discarded_at_finalization():
    handler = _handler_after_soft_ended_turn()
    handler.min_speech_continuation_ms = 192
    chunks = [torch.zeros(512) for _ in range(8)]
    handler.iterator = _StaticVADIterator(
        triggered=True,
        vad_output=None,
        buffer_chunks=chunks,
        speech_chunks=chunks,
        active_speech_samples=8 * 512,
    )

    assert list(handler.process(_audio_bytes())) == []
    started = handler.text_output_queue.get_nowait()
    assert isinstance(started, SpeechStartedEvent)
    assert (started.turn_id, started.turn_revision, started.reopened) == ("turn_1", 1, True)

    handler.iterator = _StaticVADIterator(
        triggered=False,
        vad_output=chunks,
        last_utterance_active_speech_samples=8 * 512,
    )
    outputs = list(handler.process(_audio_bytes()))

    assert len(outputs) == 1
    assert (outputs[0].turn_id, outputs[0].turn_revision) == ("turn_1", 1)


def test_continuation_threshold_clamping():
    assert VADHandler._resolve_min_speech_continuation_ms(384, 0) == 384
    assert VADHandler._resolve_min_speech_continuation_ms(384, 50) == 100
    assert VADHandler._resolve_min_speech_continuation_ms(384, 500) == 384
    assert VADHandler._resolve_min_speech_continuation_ms(384, 192) == 192


def test_vad_reopens_unanswered_turn_after_grace_window():
    handler = _vad_handler_for_iterator(_StaticVADIterator(triggered=False, vad_output=None))
    handler.unanswered_reopen_ms = 8000
    tracker = handler.speculative_turns

    outputs = _drive_final_segment(handler)
    assert len(outputs) == 1
    assert (outputs[0].turn_id, outputs[0].turn_revision) == ("turn_1", 0)
    assert handler._last_final_audio_ms is not None
    while not handler.text_output_queue.empty():
        handler.text_output_queue.get_nowait()

    # Advance the audio clock so the resumed speech starts well past
    # speculative_reopen_ms (800) but within unanswered_reopen_ms (8000).
    handler._total_samples = 16000 * 3

    outputs = _drive_final_segment(handler)

    assert len(outputs) == 1
    assert (outputs[0].turn_id, outputs[0].turn_revision) == ("turn_1", 1)
    started = handler.text_output_queue.get_nowait()
    assert isinstance(started, SpeechStartedEvent)
    assert (started.turn_id, started.turn_revision, started.reopened) == ("turn_1", 1, True)
    assert not tracker.is_latest("turn_1", 0)


def test_vad_does_not_reopen_committed_turn():
    handler = _vad_handler_for_iterator(_StaticVADIterator(triggered=False, vad_output=None))
    handler.unanswered_reopen_ms = 8000
    tracker = handler.speculative_turns

    outputs = _drive_final_segment(handler)
    assert len(outputs) == 1
    assert (outputs[0].turn_id, outputs[0].turn_revision) == ("turn_1", 0)
    while not handler.text_output_queue.empty():
        handler.text_output_queue.get_nowait()

    tracker.commit("turn_1", 0)
    handler._total_samples = 16000 * 3

    outputs = _drive_final_segment(handler)

    assert len(outputs) == 1
    assert (outputs[0].turn_id, outputs[0].turn_revision) == ("turn_2", 0)
    started = handler.text_output_queue.get_nowait()
    assert isinstance(started, SpeechStartedEvent)
    assert started.reopened is False


def test_vad_new_turn_after_unanswered_cap():
    handler = _vad_handler_for_iterator(_StaticVADIterator(triggered=False, vad_output=None))
    handler.unanswered_reopen_ms = 8000

    outputs = _drive_final_segment(handler)
    assert len(outputs) == 1
    assert (outputs[0].turn_id, outputs[0].turn_revision) == ("turn_1", 0)
    while not handler.text_output_queue.empty():
        handler.text_output_queue.get_nowait()

    # Advance the audio clock so the resumed speech starts past the cap.
    handler._total_samples = 16000 * 12

    outputs = _drive_final_segment(handler)

    assert len(outputs) == 1
    assert (outputs[0].turn_id, outputs[0].turn_revision) == ("turn_2", 0)
    started = handler.text_output_queue.get_nowait()
    assert isinstance(started, SpeechStartedEvent)
    assert started.reopened is False


def test_vad_does_not_hold_sub_floor_fragments():
    handler = _vad_handler_for_iterator(
        _StaticVADIterator(
            triggered=False,
            vad_output=[torch.zeros(512)],
            last_utterance_active_speech_samples=512,
        )
    )
    handler.short_segment_merge_ms = 384

    outputs = list(handler.process(_audio_bytes()))

    assert outputs == []
    assert handler._pending_short_segment is None
    assert handler.text_output_queue.empty()


def test_vad_stitches_adjacent_short_segments_before_discarding():
    first_chunks = [torch.zeros(512) for _ in range(7)]
    second_chunks = [torch.zeros(512) for _ in range(8)]
    handler = _vad_handler_for_iterator(
        _StaticVADIterator(
            triggered=False,
            vad_output=first_chunks,
            last_utterance_active_speech_samples=4 * 512,
        )
    )
    handler.short_segment_merge_ms = 384

    assert list(handler.process(_audio_bytes())) == []
    assert handler.text_output_queue.empty()
    assert handler._pending_short_segment is not None

    handler.iterator = _StaticVADIterator(
        triggered=False,
        vad_output=second_chunks,
        last_utterance_active_speech_samples=8 * 512,
    )
    outputs = list(handler.process(_audio_bytes()))

    assert len(outputs) == 1
    assert len(outputs[0].audio) == 15 * 512
    started = handler.text_output_queue.get_nowait()
    stopped = handler.text_output_queue.get_nowait()
    assert isinstance(started, SpeechStartedEvent)
    assert started.interrupt_response is False
    assert isinstance(stopped, SpeechStoppedEvent)
    assert handler._pending_short_segment is None


def test_vad_pending_short_segment_contributes_to_early_speech_start():
    first_chunks = [torch.zeros(512) for _ in range(7)]
    current_chunks = [torch.zeros(512) for _ in range(8)]
    handler = _vad_handler_for_iterator(
        _StaticVADIterator(
            triggered=False,
            vad_output=first_chunks,
            last_utterance_active_speech_samples=4 * 512,
        )
    )
    handler.short_segment_merge_ms = 384

    assert list(handler.process(_audio_bytes())) == []

    handler.iterator = _StaticVADIterator(
        triggered=True,
        vad_output=None,
        buffer_chunks=current_chunks,
        speech_chunks=current_chunks,
        active_speech_samples=8 * 512,
    )

    assert list(handler.process(_audio_bytes())) == []
    event = handler.text_output_queue.get_nowait()
    assert isinstance(event, SpeechStartedEvent)
    assert event.interrupt_response is True
    assert handler._speech_started_emitted is True


def test_vad_pending_short_segment_does_not_start_on_sub_floor_current_fragment():
    first_chunks = [torch.zeros(512) for _ in range(10)]
    current_chunks = [torch.zeros(512) for _ in range(3)]
    handler = _vad_handler_for_iterator(
        _StaticVADIterator(
            triggered=False,
            vad_output=first_chunks,
            last_utterance_active_speech_samples=9 * 512,
        )
    )
    handler.short_segment_merge_ms = 384

    assert list(handler.process(_audio_bytes())) == []
    assert handler._pending_short_segment is not None

    # Pending holds 288ms active speech; the live fragment has only 96ms,
    # below the noise floor, so the combined 384ms must not start speech.
    handler.iterator = _StaticVADIterator(
        triggered=True,
        vad_output=None,
        buffer_chunks=current_chunks,
        speech_chunks=current_chunks,
        active_speech_samples=3 * 512,
    )

    assert list(handler.process(_audio_bytes())) == []
    assert handler.text_output_queue.empty()
    assert handler._speech_started_emitted is False


def test_vad_stitching_preserves_silence_gap_between_segments():
    first_chunks = [torch.zeros(512) for _ in range(7)]
    second_chunks = [torch.zeros(512) for _ in range(8)]
    handler = _vad_handler_for_iterator(
        _StaticVADIterator(
            triggered=False,
            vad_output=first_chunks,
            last_utterance_active_speech_samples=4 * 512,
        )
    )
    handler.short_segment_merge_ms = 384

    assert list(handler.process(_audio_bytes())) == []
    assert handler._pending_short_segment is not None
    assert handler._pending_short_segment.end_ms == 32

    # Advance the audio clock so the second segment starts 32ms after the
    # pending one ends; the stitched audio must include that silent gap.
    handler._total_samples = 9 * 512
    handler.iterator = _StaticVADIterator(
        triggered=False,
        vad_output=second_chunks,
        last_utterance_active_speech_samples=8 * 512,
    )
    outputs = list(handler.process(_audio_bytes()))

    assert len(outputs) == 1
    assert len(outputs[0].audio) == 16 * 512


def test_vad_final_synthetic_start_does_not_interrupt_response():
    final_chunks = [torch.zeros(512) for _ in range(31)]
    iterator = _StaticVADIterator(
        triggered=False,
        vad_output=final_chunks,
        last_utterance_active_speech_samples=12 * 512,
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
