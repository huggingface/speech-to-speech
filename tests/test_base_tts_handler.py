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


def test_scoped_process_holds_during_grace_releases_mlx_lock_and_defers_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    from speech_to_speech.TTS import base_tts_handler as base_module
    from speech_to_speech.utils.mlx_lock import MLXLockContext, acquire_mlx_lock, release_mlx_lock

    printed: list[str] = []
    monkeypatch.setattr(base_module.console, "print", lambda msg, *_a, **_kw: printed.append(msg))
    synth_done = threading.Event()

    class _StreamingHandler(BaseTTSHandler):
        def process(self, tts_input):
            if not self._admit_input(tts_input):
                return
            with MLXLockContext(handler_name="TestTTS"):
                self._latency_timer(tts_input).first_provider_audio()
                synth_done.set()
                yield b"chunk-1"
                yield b"chunk-2"

    tracker = SpeculativeTurnTracker()
    turn_id, rev = tracker.start_turn()
    tracker.start_reopen_grace(turn_id, rev, grace_s=0.1)
    store = TurnLatencyStore()
    latency = store.get_or_create_response("resp_1", turn_id=turn_id, turn_revision=rev)

    handler = _StreamingHandler(threading.Event(), queue.Queue(), queue.Queue())
    handler.speculative_turns = tracker
    handler.turn_latency_store = store
    tts_input = TTSInput(
        text="hello", turn_id=turn_id, turn_revision=rev, response_key="resp_1", speech_stopped_at_s=time.perf_counter()
    )
    done, outputs = threading.Event(), []

    thread = threading.Thread(target=lambda: (outputs.extend(handler.process(tts_input)), done.set()))
    thread.start()
    try:
        assert synth_done.wait(0.5)
        assert not done.wait(0.02)
        assert outputs == [] and printed == [] and not tracker.is_committed(turn_id, rev)
        assert latency.tts_ttfa_s is None and latency.e2e_s is None
        # MLX lock must be free while waiting in reopen grace.
        assert acquire_mlx_lock(timeout=0.05, handler_name="TestOther") is True
        release_mlx_lock(handler_name="TestOther")
        assert done.wait(1.0)
    finally:
        thread.join(timeout=1.0)

    assert outputs == [b"chunk-1", b"chunk-2"]
    assert printed == ["[green]ASSISTANT: hello"]
    assert tracker.is_committed(turn_id, rev)
    assert latency.tts_ttfa_s is not None and latency.tts_ttfa_s < 0.05
    assert latency.e2e_s is not None and latency.e2e_s >= 0.08


@pytest.mark.parametrize("abort", ["cancel", "reopen"])
def test_scoped_process_aborts_cleanly_and_suppresses_ttfa_during_grace(
    monkeypatch: pytest.MonkeyPatch, abort: str
) -> None:
    from speech_to_speech.TTS import base_tts_handler as base_module

    printed: list[str] = []
    monkeypatch.setattr(base_module.console, "print", lambda msg, *_a, **_kw: printed.append(msg))
    first_ready, second_pulled, caught, closed = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )

    class _StreamingHandler(BaseTTSHandler):
        def process(self, tts_input):
            if not self._admit_input(tts_input):
                return
            try:
                self._latency_timer(tts_input).first_provider_audio()
                first_ready.set()
                yield b"chunk-1"
                second_pulled.set()
                yield b"chunk-2"
            except Exception:
                caught.set()
            finally:
                closed.set()

    tracker = SpeculativeTurnTracker()
    turn_id, rev, _ = tracker.speech_started(0)
    tracker.segment_finalized(1000, output_hold_ms=300)
    scope, store = CancelScope(), TurnLatencyStore()
    latency = store.get_or_create_response("resp_abort", turn_id=turn_id, turn_revision=rev)
    handler = _StreamingHandler(threading.Event(), queue.Queue(), queue.Queue())
    handler.speculative_turns, handler.cancel_scope, handler.turn_latency_store = tracker, scope, store

    tts_input = TTSInput(
        text="hello", turn_id=turn_id, turn_revision=rev, response_key="resp_abort", cancel_generation=scope.generation
    )
    done, outputs = threading.Event(), []
    thread = threading.Thread(target=lambda: (outputs.extend(handler.process(tts_input)), done.set()))
    thread.start()
    try:
        assert first_ready.wait(0.5)
        assert not done.wait(0.02)
        if abort == "cancel":
            scope.cancel()
        else:
            reopen(tracker, turn_id, rev)
        assert done.wait(0.3)
    finally:
        thread.join(timeout=1.0)

    assert outputs == [] and printed == []
    assert closed.is_set() and not caught.is_set() and not second_pulled.is_set()
    assert not tracker.is_committed(turn_id, rev)
    assert latency.tts_ttfa_s is None and latency.e2e_s is None
    if abort == "cancel":
        assert tracker.can_reopen(1100) and tracker.speech_started(1100) == (turn_id, 1, True)


def test_is_stale_honors_reopen_and_nested_super_process_commits_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from speech_to_speech.TTS import base_tts_handler as base_module

    printed: list[str] = []
    monkeypatch.setattr(base_module.console, "print", lambda msg, *_a, **_kw: printed.append(msg))

    tracker, scope, store = SpeculativeTurnTracker(), CancelScope(), TurnLatencyStore()
    turn_id, rev = tracker.start_turn()
    stale_latency = store.get_or_create_response("resp_stale", turn_id=turn_id, turn_revision=rev)

    handler = _Handler(threading.Event(), queue.Queue(), queue.Queue())
    handler.speculative_turns, handler.cancel_scope, handler.turn_latency_store = tracker, scope, store
    tts_input = TTSInput(text="hi", turn_id=turn_id, turn_revision=rev, response_key="resp_stale")
    gen = scope.generation
    timer = handler._latency_timer(tts_input, gen)
    assert not handler._is_stale(gen, tts_input)
    reopen(tracker, turn_id, rev)
    assert handler._is_stale(gen, tts_input)
    timer.first_provider_audio()
    assert stale_latency.tts_ttfa_s is None

    class _Parent(BaseTTSHandler):
        def process(self, item):
            self._latency_timer(item).first_provider_audio()
            yield b"audio"

    class _Child(_Parent):
        def process(self, item):
            yield from super().process(item)

    nested_latency = store.get_or_create_response("resp_nested")
    child = _Child(threading.Event(), queue.Queue(), queue.Queue())
    child.turn_latency_store = store
    assert list(child.process(TTSInput(text="nested", response_key="resp_nested"))) == [b"audio"]
    assert printed == ["[green]ASSISTANT: nested"]
    assert nested_latency.tts_ttfa_s is not None
