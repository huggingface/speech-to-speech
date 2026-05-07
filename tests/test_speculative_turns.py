from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker


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
