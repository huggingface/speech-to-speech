from __future__ import annotations

import queue
import threading

import pytest

from speech_to_speech.pipeline.messages import EndOfResponse, TTSInput
from speech_to_speech.TTS.base_tts_handler import BaseTTSHandler


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
