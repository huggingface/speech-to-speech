"""Drive the speculative turn tracker through its public reopen path."""

from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker


def reopen(tracker: SpeculativeTurnTracker, turn_id: str = "turn_1", revision: int = 0) -> int:
    """Reopen the current turn the way VAD does and return the new revision."""
    candidate = tracker.begin_reopen_candidate(turn_id, revision)
    assert candidate == revision + 1
    assert tracker.confirm_reopen_candidate(turn_id, revision, candidate)
    return candidate
