from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.pipeline.turn_latency import TurnLatencyStore


class _FakeChat:
    class InferCodeParams:
        def __init__(self, **_kwargs: Any) -> None:
            pass


def _tone(samples: int = 24000) -> np.ndarray:
    return (0.3 * np.sin(np.arange(samples) / 50)).astype(np.float32)


@pytest.fixture
def chattts_modules(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "ChatTTS", SimpleNamespace(Chat=_FakeChat))
    from speech_to_speech.TTS import chatTTS_handler
    from speech_to_speech.TTS.chatTTS_handler import ChatTTSHandler

    monkeypatch.setattr(chatTTS_handler.console, "print", lambda *_a, **_k: None)
    return chatTTS_handler, ChatTTSHandler


def _handler(ChatTTSHandler, *, stream: bool, chunks: list[Any]):
    handler = object.__new__(ChatTTSHandler)
    handler.stream = stream
    handler.chunk_size = 512
    handler.device = "cpu"
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.params_infer_code = None
    if stream:
        handler.model = SimpleNamespace(infer=lambda *_a, **_k: iter([[chunk] for chunk in chunks]))
    else:
        handler.model = SimpleNamespace(infer=lambda *_a, **_k: [chunks[0]])
    return handler


def test_stream_records_ttfa_before_resample_and_e2e_on_first_yield(monkeypatch, chattts_modules):
    chatTTS_handler, ChatTTSHandler = chattts_modules
    clock = [100.0]
    monkeypatch.setattr(chatTTS_handler, "perf_counter", lambda: clock[0])

    def slow_resample(chunk, *, orig_sr, target_sr):
        clock[0] = 100.40
        # Keep length roughly stable for block assembly.
        return chunk[: int(len(chunk) * target_sr / orig_sr)].astype(np.float32)

    # Replace the module attribute so lazy librosa/numba is never imported in tests.
    monkeypatch.setattr(chatTTS_handler, "librosa", SimpleNamespace(resample=slow_resample))

    def infer(*_a, **_k):
        clock[0] = 100.05
        yield [_tone()]

    handler = _handler(ChatTTSHandler, stream=True, chunks=[_tone()])
    handler.model = SimpleNamespace(infer=infer)
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_response("resp-1", turn_id="turn_1", turn_revision=0)

    outputs = list(
        handler.process(
            TTSInput(
                text="hello",
                response_key="resp-1",
                turn_id="turn_1",
                turn_revision=0,
                speech_stopped_at_s=99.0,
            )
        )
    )

    assert outputs
    assert tracker.tts_ttfa_s == pytest.approx(0.05)
    # First pipeline yield happens after resample advances the clock.
    assert tracker.e2e_s == pytest.approx(1.40)


def test_stream_preserves_first_audio_across_later_segments(monkeypatch, chattts_modules):
    chatTTS_handler, ChatTTSHandler = chattts_modules
    clock = [50.0]
    monkeypatch.setattr(chatTTS_handler, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        chatTTS_handler,
        "librosa",
        SimpleNamespace(
            resample=lambda chunk, *, orig_sr, target_sr: chunk[
                : int(len(chunk) * target_sr / orig_sr)
            ].astype(np.float32)
        ),
    )

    def first_infer(*_a, **_k):
        clock[0] = 50.08
        yield [_tone()]
        clock[0] = 50.90
        yield [_tone(12000)]

    handler = _handler(ChatTTSHandler, stream=True, chunks=[_tone()])
    handler.model = SimpleNamespace(infer=first_infer)
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_response("resp-1", turn_id="turn_1", turn_revision=0)

    list(
        handler.process(
            TTSInput(
                text="one",
                response_key="resp-1",
                turn_id="turn_1",
                turn_revision=0,
                speech_stopped_at_s=49.5,
            )
        )
    )
    assert tracker.tts_ttfa_s == pytest.approx(0.08)
    first_e2e = tracker.e2e_s
    assert first_e2e is not None

    clock[0] = 80.0

    def later_infer(*_a, **_k):
        clock[0] = 80.50
        yield [_tone()]

    handler.model = SimpleNamespace(infer=later_infer)
    list(
        handler.process(
            TTSInput(
                text="two",
                response_key="resp-1",
                turn_id="turn_1",
                turn_revision=0,
                speech_stopped_at_s=49.5,
            )
        )
    )

    assert tracker.tts_ttfa_s == pytest.approx(0.08)
    assert tracker.e2e_s == pytest.approx(first_e2e)


def test_non_stream_records_ttfa_before_resample(monkeypatch, chattts_modules):
    chatTTS_handler, ChatTTSHandler = chattts_modules
    clock = [10.0]
    monkeypatch.setattr(chatTTS_handler, "perf_counter", lambda: clock[0])

    def slow_resample(chunk, *, orig_sr, target_sr):
        clock[0] = 10.30
        return chunk[: int(len(chunk) * target_sr / orig_sr)].astype(np.float32)

    monkeypatch.setattr(chatTTS_handler, "librosa", SimpleNamespace(resample=slow_resample))

    def infer(*_a, **_k):
        clock[0] = 10.04
        return [_tone()]

    handler = _handler(ChatTTSHandler, stream=False, chunks=[_tone()])
    handler.model = SimpleNamespace(infer=infer)
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_response("resp-2", turn_id="turn_2", turn_revision=0)

    outputs = list(
        handler.process(
            TTSInput(
                text="hello",
                response_key="resp-2",
                turn_id="turn_2",
                turn_revision=0,
                speech_stopped_at_s=9.5,
            )
        )
    )

    assert outputs
    assert tracker.tts_ttfa_s == pytest.approx(0.04)
    assert tracker.e2e_s == pytest.approx(0.80)


def test_empty_provider_chunk_leaves_latency_unset(monkeypatch, chattts_modules):
    chatTTS_handler, ChatTTSHandler = chattts_modules
    clock = [1.0]
    monkeypatch.setattr(chatTTS_handler, "perf_counter", lambda: clock[0])

    handler = _handler(ChatTTSHandler, stream=True, chunks=[np.empty(0, dtype=np.float32)])
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_response("resp-3", turn_id="turn_3", turn_revision=0)

    assert list(handler.process(TTSInput(text="x", response_key="resp-3", speech_stopped_at_s=0.5))) == []
    assert tracker.tts_ttfa_s is None
    assert tracker.e2e_s is None
