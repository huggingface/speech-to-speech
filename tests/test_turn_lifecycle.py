from __future__ import annotations

import pytest

from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker, TurnPhase


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("speech_to_speech.pipeline.speculative_turns.time.monotonic", lambda: now[0])
    return now


def test_tracker_owns_full_speech_and_response_lifecycle():
    tracker = SpeculativeTurnTracker()
    assert tracker.phase is None
    assert tracker.current_turn() == (None, None)

    turn_id, revision, reopened = tracker.speech_started(0)
    assert not reopened
    assert tracker.phase == TurnPhase.LISTENING
    # More confirmed audio in the same segment does not mint another turn.
    assert tracker.speech_started(400) == (turn_id, revision, False)
    tracker.segment_finalized(1000)
    assert tracker.phase == TurnPhase.SOFT_ENDED
    assert tracker.commit_current(turn_id, revision)
    assert tracker.phase == TurnPhase.ANSWERING
    tracker.close(turn_id, revision)
    assert tracker.phase == TurnPhase.CLOSED
    assert not tracker.can_reopen(1100)

    # Explicit tool/client follow-ups still belong to the answered turn.
    assert tracker.commit_current(turn_id, revision)
    assert tracker.phase == TurnPhase.ANSWERING
    tracker.close(turn_id, revision)
    assert tracker.speech_started(1100) == ("turn_2", 0, False)


@pytest.mark.parametrize("gap_ms,reopens", [(0, True), (800, True), (6999, True), (7000, True), (7001, False)])
def test_unanswered_reopen_cap_uses_audio_endpoint(gap_ms, reopens):
    tracker = SpeculativeTurnTracker()
    tracker.configure_reopen(7000)
    tracker.speech_started(0)
    tracker.segment_finalized(1000)

    result = tracker.speech_started(1000 + gap_ms)

    assert result == (("turn_1", 1, True) if reopens else ("turn_2", 0, False))
    assert not tracker.is_latest("turn_1", 0)
    assert tracker.phase == TurnPhase.LISTENING


@pytest.mark.parametrize("hold_ms,delay_ms", [(800, 0), (2000, 600)])
def test_output_hold_and_processing_delay_have_separate_deadlines(clock, hold_ms, delay_ms):
    tracker = SpeculativeTurnTracker()
    tracker.speech_started(0)
    tracker.segment_finalized(1000, output_hold_ms=hold_ms, processing_delay_ms=delay_ms)
    assert tracker.processing_delay_remaining("turn_1", 0) == pytest.approx(delay_ms / 1000)
    assert not tracker.commit_current("turn_1", 0)

    clock[0] = 100.0 + hold_ms / 1000 - 0.001
    assert tracker.processing_delay_remaining("turn_1", 0) == 0
    assert tracker.has_pending_reopen_or_grace("turn_1", 0)
    assert not tracker.commit_current("turn_1", 0)

    clock[0] += 0.002
    assert tracker.commit_current("turn_1", 0)
    assert tracker.phase == TurnPhase.ANSWERING


def test_push_to_talk_pause_expires_output_hold_without_advancing_audio_cap(clock):
    tracker = SpeculativeTurnTracker()
    tracker.speech_started(0)
    tracker.segment_finalized(1000, output_hold_ms=2000, processing_delay_ms=600)

    clock[0] += 100.0

    assert not tracker.has_pending_reopen_or_grace("turn_1", 0)
    assert tracker.processing_delay_remaining("turn_1", 0) == 0
    assert tracker.can_reopen(1100)
    assert tracker.speech_started(1100) == ("turn_1", 1, True)


@pytest.mark.parametrize("resume_ms", [1100, 1900])
def test_resumed_speech_reopens_during_processing_or_output_hold(clock, resume_ms):
    tracker = SpeculativeTurnTracker()
    tracker.speech_started(0)
    tracker.segment_finalized(1000, output_hold_ms=2000, processing_delay_ms=600)
    clock[0] += (resume_ms - 1000) / 1000

    assert tracker.speech_candidate_started(resume_ms)
    assert not tracker.commit_current("turn_1", 0)
    assert tracker.speech_started(resume_ms + 192) == ("turn_1", 1, True)
    assert not tracker.is_latest("turn_1", 0)
    assert tracker.processing_delay_remaining("turn_1", 1) == 0


def test_noise_candidate_cancellation_releases_output_without_new_revision(clock):
    tracker = SpeculativeTurnTracker()
    tracker.speech_started(0)
    tracker.segment_finalized(1000, output_hold_ms=800)
    assert tracker.speech_candidate_started(1100)
    clock[0] += 1.0
    assert not tracker.commit_current("turn_1", 0)

    tracker.speech_candidate_cancelled()

    assert tracker.current_turn() == ("turn_1", 0)
    assert tracker.phase == TurnPhase.SOFT_ENDED
    assert tracker.commit_current("turn_1", 0)


def test_candidate_admitted_before_cap_can_confirm_after_speech_threshold():
    tracker = SpeculativeTurnTracker()
    tracker.speech_started(0)
    tracker.segment_finalized(1000)
    assert tracker.speech_candidate_started(7999)

    assert tracker.speech_started(8191) == ("turn_1", 1, True)


def test_committed_older_completion_does_not_close_current_turn():
    tracker = SpeculativeTurnTracker()
    tracker.speech_started(0)
    tracker.segment_finalized(1000)
    assert tracker.commit_current("turn_1", 0)
    assert tracker.speech_started(1100) == ("turn_2", 0, False)
    tracker.segment_finalized(2000)
    assert tracker.is_latest("turn_1", 0)

    tracker.close("turn_1", 0)

    assert not tracker.is_latest("turn_1", 0)
    assert tracker.current_turn() == ("turn_2", 0)
    assert tracker.phase == TurnPhase.SOFT_ENDED
    assert tracker.can_reopen(2100)


def test_discarded_speech_does_not_leave_tracker_listening():
    tracker = SpeculativeTurnTracker()
    tracker.speech_started(0)
    tracker.segment_discarded()
    assert tracker.phase == TurnPhase.SOFT_ENDED
    assert not tracker.can_reopen(1000)

    assert tracker.speech_started(1000) == ("turn_2", 0, False)


def test_reset_clears_lifecycle_and_both_deadlines(clock):
    tracker = SpeculativeTurnTracker()
    tracker.speech_started(0)
    tracker.segment_finalized(1000, output_hold_ms=2000, processing_delay_ms=600)
    tracker.speech_candidate_started(1100)

    tracker.reset()

    assert tracker.current_turn() == (None, None)
    assert tracker.phase is None
    assert not tracker.has_speech_candidate()
    assert not tracker.can_reopen(1100)
    assert tracker.processing_deadline("turn_1", 0) is None
    assert not tracker.has_pending_reopen_or_grace("turn_1", 0)
    assert not tracker.commit_current("turn_1", 0)
