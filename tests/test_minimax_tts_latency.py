"""MiniMax SSE timing through the inherited HTTP TTS decoder and block assembly."""

from types import SimpleNamespace

import numpy as np
import pytest

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import ResponseFailedEvent
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.pipeline.turn_latency import TurnLatencyStore
from tests.test_minimax_tts_handler import _event, _handler


def _set_sse_stream(monkeypatch, handler, clock, frames, *, before_first=None):
    """Stub only encoded transport bytes; keep MiniMax SSE/PCM decoding real."""

    def iter_bytes(cancel_check):
        if before_first is not None:
            before_first()
        for at_s, frame in frames:
            if cancel_check():
                return
            clock[0] = at_s
            yield frame

    operation = SimpleNamespace(iter_bytes=iter_bytes, cancel=lambda: None)
    monkeypatch.setattr(handler, "_make_operation", lambda **kwargs: operation)


def _audio_frames(samples, *, start=100.0):
    return [
        (start + 0.05, _event()),  # SSE metadata is not first audio.
        (start + 0.10, _event(samples[:3].tobytes())),
        (start + 0.40, _event(samples[3:].tobytes())),
        (start + 0.40, _event(status=2)),
    ]


@pytest.fixture
def timed_handler(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("speech_to_speech.pipeline.turn_latency.perf_counter", lambda: clock[0])
    handler = _handler(cancel_scope=CancelScope())
    handler.turn_latency_store = TurnLatencyStore()
    yield handler, clock
    handler.cleanup()


def test_provider_audio_precedes_block_assembly_and_later_segments_keep_first_values(timed_handler, monkeypatch):
    handler, clock = timed_handler
    tracker = handler.turn_latency_store.get_or_create_response("resp-1")
    samples = np.arange(768, dtype="<i2")
    _set_sse_stream(monkeypatch, handler, clock, _audio_frames(samples))
    item = TTSInput(text="First segment.", response_key="resp-1", speech_stopped_at_s=99.0)

    outputs = list(handler.process(item))

    np.testing.assert_array_equal(np.concatenate(outputs), samples)
    assert [len(chunk) for chunk in outputs] == [512, 256]
    assert tracker.tts_ttfa_s == pytest.approx(0.10)
    assert tracker.e2e_s == pytest.approx(1.40)

    clock[0] = 110.0
    _set_sse_stream(monkeypatch, handler, clock, _audio_frames(samples, start=110.0))
    assert list(handler.process(item))
    assert tracker.tts_ttfa_s == pytest.approx(0.10)
    assert tracker.e2e_s == pytest.approx(1.40)


@pytest.mark.parametrize("path", ["empty", "cancelled", "discarded"])
def test_unavailable_audio_or_tracker_cannot_contaminate_another_response(timed_handler, monkeypatch, path):
    handler, clock = timed_handler
    store = handler.turn_latency_store
    tracker = store.get_or_create_response("old")
    other = store.get_or_create_response("new")
    before_first = None
    frames = _audio_frames(np.arange(768, dtype="<i2"))
    if path == "empty":
        frames = [(100.10, _event(status=2))]
    elif path == "cancelled":
        before_first = handler.cancel_scope.cancel
    else:

        def before_first():
            store.discard_response("old")

    _set_sse_stream(monkeypatch, handler, clock, frames, before_first=before_first)

    output = list(handler.process(TTSInput(text="Hello.", response_key="old", speech_stopped_at_s=99.0)))

    assert tracker.tts_ttfa_s is None
    assert tracker.e2e_s is None
    assert other.tts_ttfa_s is None
    assert other.e2e_s is None
    if path == "discarded":
        assert output
        assert store.get_response("old") is None
    else:
        assert output == []
    if path == "empty":
        assert isinstance(handler.queue_out.get_nowait(), ResponseFailedEvent)
    else:
        assert handler.queue_out.empty()


def test_cancellation_after_first_block_preserves_first_audio_measurements(timed_handler, monkeypatch):
    handler, clock = timed_handler
    tracker = handler.turn_latency_store.get_or_create_response("resp-1")
    _set_sse_stream(monkeypatch, handler, clock, _audio_frames(np.arange(768, dtype="<i2")))
    output = handler.process(TTSInput(text="Hello.", response_key="resp-1", speech_stopped_at_s=99.0))

    assert len(next(output)) == 512
    handler.cancel_scope.cancel()
    assert list(output) == []
    assert tracker.tts_ttfa_s == pytest.approx(0.10)
    assert tracker.e2e_s == pytest.approx(1.40)
