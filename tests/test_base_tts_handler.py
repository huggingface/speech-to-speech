from __future__ import annotations

import queue
import threading

import pytest

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.pipeline.turn_latency import TurnLatencyStore
from speech_to_speech.TTS.base_tts_handler import BaseTTSHandler
from tests.turns import reopen


class _Tracker:
    def __init__(self, accept: bool) -> None:
        self.accept = accept
        self.calls: list[tuple] = []

    def wait_for_gate(self, turn_id, revision, *, commit=False) -> bool:
        self.calls.append((turn_id, revision, commit))
        return self.accept


class _Handler(BaseTTSHandler):
    def process(self, tts_input):  # pragma: no cover - not used
        yield from ()


def _handler(tracker: _Tracker | None) -> _Handler:
    handler = _Handler(threading.Event(), queue.Queue(), queue.Queue())
    handler.speculative_turns = tracker  # type: ignore[assignment]
    return handler


def _text(**kwargs) -> TTSInput:
    return TTSInput(text="hi", turn_id="t", turn_revision=1, **kwargs)


def test_end_of_response_passes_without_tracker() -> None:
    end = EndOfResponse(turn_id="t", turn_revision=1)
    assert _handler(None)._gate_end_of_response(end) is True
    assert end.cleanup_only is False


def test_end_of_response_accepted_does_not_mark_cleanup_only() -> None:
    tracker = _Tracker(accept=True)
    end = EndOfResponse(turn_id="t", turn_revision=1, response_key="r")
    assert _handler(tracker)._gate_end_of_response(end) is True
    assert end.cleanup_only is False
    assert tracker.calls == [("t", 1, False)]


def test_stale_end_of_response_with_key_is_forwarded_as_cleanup_only() -> None:
    end = EndOfResponse(turn_id="t", turn_revision=1, response_key="r")
    assert _handler(_Tracker(accept=False))._gate_end_of_response(end) is True
    assert end.cleanup_only is True


def test_stale_end_of_response_without_key_is_dropped() -> None:
    end = EndOfResponse(turn_id="t", turn_revision=1)
    assert _handler(_Tracker(accept=False))._gate_end_of_response(end) is False
    assert end.cleanup_only is False


@pytest.mark.parametrize("commit", [True, False])
def test_admit_input_forwards_commit_flag(commit: bool) -> None:
    tracker = _Tracker(accept=True)
    assert _handler(tracker)._admit_input(_text(), commit=commit) is True
    assert tracker.calls == [("t", 1, commit)]


def test_admit_input_commits_by_default_and_drops_stale() -> None:
    tracker = _Tracker(accept=False)
    assert _handler(tracker)._admit_input(_text()) is False
    assert tracker.calls == [("t", 1, True)]


def test_admit_input_passes_without_tracker() -> None:
    assert _handler(None)._admit_input(_text()) is True


def test_speculative_admit_and_commit_first_audio_during_reopen_grace() -> None:
    tracker = SpeculativeTurnTracker()
    turn_id, rev = tracker.start_turn()
    tracker.start_reopen_grace(turn_id, rev, grace_s=0.08)
    handler = _Handler(threading.Event(), queue.Queue(), queue.Queue())
    handler.speculative_turns = tracker
    tts_input = TTSInput(text="hi", turn_id=turn_id, turn_revision=rev)

    # Speculative admission succeeds immediately without committing the turn.
    assert handler._admit_input(tts_input, commit=False, hold_for_grace=False) is True
    assert not tracker.is_committed(turn_id, rev)

    # _commit_first_audio blocks until reopen grace elapses, then commits.
    done = threading.Event()
    committed: list[bool] = []

    def run_commit() -> None:
        committed.append(handler._commit_first_audio(tts_input))
        done.set()

    thread = threading.Thread(target=run_commit)
    thread.start()
    assert not done.wait(0.02)
    assert done.wait(1.0)
    thread.join(timeout=1.0)

    assert committed == [True]
    assert tracker.is_committed(turn_id, rev)


def test_is_synthesis_cancelled_and_latency_timer_honor_speculative_reopen() -> None:
    tracker = SpeculativeTurnTracker()
    turn_id, rev = tracker.start_turn()
    handler = _Handler(threading.Event(), queue.Queue(), queue.Queue())
    handler.speculative_turns = tracker
    handler.cancel_scope = CancelScope()
    store = TurnLatencyStore()
    handler.turn_latency_store = store
    store.get_or_create_response("resp_1", turn_id=turn_id, turn_revision=rev)

    tts_input = TTSInput(text="hi", turn_id=turn_id, turn_revision=rev, response_key="resp_1")
    gen = handler._effective_cancel_generation(tts_input)
    timer = handler._latency_timer(tts_input, gen)
    assert not handler._is_synthesis_cancelled(tts_input, gen)

    reopen(tracker, turn_id, rev)
    assert handler._is_synthesis_cancelled(tts_input, gen)

    # Superseded speculative revision must not record tts_ttfa (would wrongly tag tts=cut).
    timer.first_provider_audio()
    response_tracker = store.get_response("resp_1")
    assert response_tracker is not None
    assert response_tracker.tts_ttfa_s is None
