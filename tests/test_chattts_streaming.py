from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from speech_to_speech.pipeline.messages import TTSInput


class _FakeChat:
    class InferCodeParams:
        def __init__(self, **_kwargs: Any) -> None:
            pass


@pytest.fixture
def handler_factory(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "ChatTTS", SimpleNamespace(Chat=_FakeChat))
    from speech_to_speech.TTS.chatTTS_handler import ChatTTSHandler

    def make(chunks: list[Any]) -> ChatTTSHandler:
        handler = object.__new__(ChatTTSHandler)
        handler.stream = True
        handler.chunk_size = 512
        handler.device = "cpu"
        handler.cancel_scope = None
        handler.speculative_turns = None
        handler.params_infer_code = None
        handler.model = SimpleNamespace(infer=lambda *_a, **_k: iter([[chunk] for chunk in chunks]))
        return handler

    return make


def _tone(samples: int = 24000) -> np.ndarray:
    return (0.3 * np.sin(np.arange(samples) / 50)).astype(np.float32)


def _blocks(handler: Any) -> list[np.ndarray]:
    return list(handler.process(TTSInput(text="hello")))


def test_streaming_accepts_a_flat_chunk(handler_factory) -> None:
    """#80: a (samples,) chunk used to be reduced to one sample by a stray [0]."""
    blocks = _blocks(handler_factory([_tone()]))

    assert blocks, "a one-second chunk must produce audio"
    assert all(len(block) == 512 for block in blocks)


def test_flat_and_column_chunks_produce_identical_audio(handler_factory) -> None:
    tone = _tone()

    flat = np.concatenate(_blocks(handler_factory([tone])))
    column = np.concatenate(_blocks(handler_factory([tone.reshape(1, -1)])))

    assert np.array_equal(flat, column)


def test_streaming_stops_on_an_empty_chunk(handler_factory) -> None:
    """An empty (1, 0) chunk passes a len() check on the outer axis, not on samples."""
    assert _blocks(handler_factory([np.empty((1, 0), dtype=np.float32)])) == []
    assert _blocks(handler_factory([np.empty(0, dtype=np.float32)])) == []


def test_streaming_stops_on_a_none_chunk(handler_factory) -> None:
    assert _blocks(handler_factory([None])) == []
