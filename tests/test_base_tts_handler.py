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

    def wait_for_gate(self, turn_id, revision, *, commit=False, hold_for_grace=True) -> bool:
        self.calls.append((turn_id, revision, commit, hold_for_grace))
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
    assert tracker.calls == [("t", 1, False, True)]


def test_stale_end_of_response_with_key_is_forwarded_as_cleanup_only() -> None:
    end = EndOfResponse(turn_id="t", turn_revision=1, response_key="r")
    assert _handler(_Tracker(accept=False))._gate_end_of_response(end) is True
    assert end.cleanup_only is True


def test_stale_end_of_response_without_key_is_dropped() -> None:
    end = EndOfResponse(turn_id="t", turn_revision=1)
    assert _handler(_Tracker(accept=False))._gate_end_of_response(end) is False
    assert end.cleanup_only is False


@pytest.mark.parametrize("commit", [True, False])
@pytest.mark.parametrize("hold_for_grace", [True, False])
def test_admit_input_forwards_flags(commit: bool, hold_for_grace: bool) -> None:
    tracker = _Tracker(accept=True)
    assert _handler(tracker)._admit_input(_text(), commit=commit, hold_for_grace=hold_for_grace) is True
    assert tracker.calls == [("t", 1, commit, hold_for_grace)]


def test_admit_input_is_speculative_by_default_and_drops_stale() -> None:
    tracker = _Tracker(accept=False)
    assert _handler(tracker)._admit_input(_text()) is False
    assert tracker.calls == [("t", 1, False, False)]


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


def test_is_stale_and_latency_timer_honor_speculative_reopen() -> None:
    tracker = SpeculativeTurnTracker()
    turn_id, rev = tracker.start_turn()
    handler = _Handler(threading.Event(), queue.Queue(), queue.Queue())
    handler.speculative_turns = tracker
    handler.cancel_scope = CancelScope()
    store = TurnLatencyStore()
    handler.turn_latency_store = store
    store.get_or_create_response("resp_1", turn_id=turn_id, turn_revision=rev)

    tts_input = TTSInput(text="hi", turn_id=turn_id, turn_revision=rev, response_key="resp_1")
    gen = handler.cancel_scope.generation
    timer = handler._latency_timer(tts_input, gen)
    assert not handler._is_stale(gen, tts_input)

    reopen(tracker, turn_id, rev)
    assert handler._is_stale(gen, tts_input)

    # Superseded speculative revision must not record tts_ttfa (would wrongly tag tts=cut).
    timer.first_provider_audio()
    response_tracker = store.get_response("resp_1")
    assert response_tracker is not None
    assert response_tracker.tts_ttfa_s is None


def test_commit_first_audio_cancellation_during_grace_does_not_commit_and_allows_reopen() -> None:
    tracker = SpeculativeTurnTracker()
    turn_id, rev, _ = tracker.speech_started(0)
    tracker.segment_finalized(1000, output_hold_ms=300)
    scope = CancelScope()
    handler = _Handler(threading.Event(), queue.Queue(), queue.Queue())
    handler.speculative_turns = tracker
    handler.cancel_scope = scope
    tts_input = TTSInput(text="hi", turn_id=turn_id, turn_revision=rev, cancel_generation=scope.generation)

    done = threading.Event()
    committed: list[bool] = []

    def run_commit() -> None:
        committed.append(handler._commit_first_audio(tts_input, generation=scope.generation))
        done.set()

    thread = threading.Thread(target=run_commit)
    thread.start()
    assert not done.wait(0.03)
    scope.cancel()
    assert done.wait(0.2)
    thread.join(timeout=1.0)

    assert committed == [False]
    assert not tracker.is_committed(turn_id, rev)
    assert tracker.can_reopen(1100)
    assert tracker.speech_started(1100) == (turn_id, 1, True)


def test_scoped_process_commits_prints_and_publishes_deferred_latency_on_first_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    from speech_to_speech.TTS import base_tts_handler as base_module

    printed: list[str] = []
    monkeypatch.setattr(base_module.console, "print", lambda msg, *_a, **_kw: printed.append(msg))

    synth_done = threading.Event()

    class _StreamingHandler(BaseTTSHandler):
        def process(self, tts_input):
            if not self._admit_input(tts_input):
                return
            timer = self._latency_timer(tts_input)
            timer.first_provider_audio()
            synth_done.set()
            yield b"chunk-1"
            yield b"chunk-2"

    tracker = SpeculativeTurnTracker()
    turn_id, rev = tracker.start_turn()
    tracker.start_reopen_grace(turn_id, rev, grace_s=0.08)
    store = TurnLatencyStore()
    response_tracker = store.get_or_create_response("resp_deferred", turn_id=turn_id, turn_revision=rev)

    handler = _StreamingHandler(threading.Event(), queue.Queue(), queue.Queue())
    handler.speculative_turns = tracker
    handler.turn_latency_store = store

    stopped_at = time.perf_counter()
    tts_input = TTSInput(
        text="hello",
        turn_id=turn_id,
        turn_revision=rev,
        response_key="resp_deferred",
        speech_stopped_at_s=stopped_at,
    )
    done = threading.Event()
    outputs: list[bytes] = []

    def run() -> None:
        outputs.extend(handler.process(tts_input))
        done.set()

    thread = threading.Thread(target=run)
    thread.start()

    assert synth_done.wait(0.5)
    assert not done.wait(0.02)
    assert outputs == []
    assert printed == []
    assert not tracker.is_committed(turn_id, rev)
    # Provider TTFA stays deferred until first-audio commit succeeds.
    assert response_tracker.tts_ttfa_s is None
    assert response_tracker.e2e_s is None

    assert done.wait(1.0)
    thread.join(timeout=1.0)

    assert outputs == [b"chunk-1", b"chunk-2"]
    assert printed == ["[green]ASSISTANT: hello"]
    assert tracker.is_committed(turn_id, rev)
    assert response_tracker.tts_ttfa_s is not None
    assert response_tracker.tts_ttfa_s < 0.05
    assert response_tracker.e2e_s is not None
    assert response_tracker.e2e_s >= 0.07


def test_scoped_process_closes_generator_and_suppresses_ttfa_on_cancel_during_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from speech_to_speech.TTS import base_tts_handler as base_module

    printed: list[str] = []
    monkeypatch.setattr(base_module.console, "print", lambda msg, *_a, **_kw: printed.append(msg))

    first_ready = threading.Event()
    second_pulled = threading.Event()
    caught_exception = threading.Event()
    closed = threading.Event()

    class _StreamingHandler(BaseTTSHandler):
        def process(self, tts_input):
            if not self._admit_input(tts_input):
                return
            timer = self._latency_timer(tts_input)
            try:
                timer.first_provider_audio()
                first_ready.set()
                yield b"chunk-1"
                second_pulled.set()
                yield b"chunk-2"
            except Exception:
                caught_exception.set()
            finally:
                closed.set()

    tracker = SpeculativeTurnTracker()
    turn_id, rev, _ = tracker.speech_started(0)
    tracker.segment_finalized(1000, output_hold_ms=300)
    scope = CancelScope()
    store = TurnLatencyStore()
    response_tracker = store.get_or_create_response("resp_cancel", turn_id=turn_id, turn_revision=rev)

    handler = _StreamingHandler(threading.Event(), queue.Queue(), queue.Queue())
    handler.speculative_turns = tracker
    handler.cancel_scope = scope
    handler.turn_latency_store = store

    tts_input = TTSInput(
        text="hello",
        turn_id=turn_id,
        turn_revision=rev,
        response_key="resp_cancel",
        cancel_generation=scope.generation,
    )
    done = threading.Event()
    outputs: list[bytes] = []

    def run() -> None:
        outputs.extend(handler.process(tts_input))
        done.set()

    thread = threading.Thread(target=run)
    thread.start()

    assert first_ready.wait(0.5)
    assert not done.wait(0.02)
    scope.cancel()
    assert done.wait(0.2)
    thread.join(timeout=1.0)

    assert outputs == []
    assert printed == []
    assert closed.is_set()
    assert not caught_exception.is_set()
    assert not second_pulled.is_set()
    assert not tracker.is_committed(turn_id, rev)
    assert response_tracker.tts_ttfa_s is None
    assert response_tracker.e2e_s is None


def test_scoped_process_supports_nested_super_process_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    from speech_to_speech.TTS import base_tts_handler as base_module

    printed: list[str] = []
    monkeypatch.setattr(base_module.console, "print", lambda msg, *_a, **_kw: printed.append(msg))

    class _Parent(BaseTTSHandler):
        def process(self, tts_input):
            timer = self._latency_timer(tts_input)
            timer.first_provider_audio()
            yield b"audio"

    class _Child(_Parent):
        def process(self, tts_input):
            yield from super().process(tts_input)

    store = TurnLatencyStore()
    response_tracker = store.get_or_create_response("resp_nested")
    handler = _Child(threading.Event(), queue.Queue(), queue.Queue())
    handler.turn_latency_store = store

    outputs = list(handler.process(TTSInput(text="nested", response_key="resp_nested")))

    assert outputs == [b"audio"]
    assert printed == ["[green]ASSISTANT: nested"]
    assert response_tracker.tts_ttfa_s is not None
