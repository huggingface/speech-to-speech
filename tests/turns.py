"""Set up speculative turn revisions for downstream tests."""

from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker


def reopen(tracker: SpeculativeTurnTracker, turn_id: str = "turn_1", revision: int = 0) -> int:
    """Advance an uncommitted turn through the candidate protocol; skip VAD admission checks."""
    candidate = tracker.begin_reopen_candidate(turn_id, revision)
    assert candidate == revision + 1
    assert tracker.confirm_reopen_candidate(turn_id, revision, candidate)
    return candidate
