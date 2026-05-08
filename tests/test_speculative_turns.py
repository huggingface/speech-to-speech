import time
from threading import Thread

import numpy as np

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
