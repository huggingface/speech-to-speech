from __future__ import annotations

import itertools
import re
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


def _stream_like_chattts(text: str, split_text: bool = True, **_kwargs: Any) -> Any:
    """Stream like ChatTTS >= 0.2.2, which splits the text after ". " unless split_text=False and synthesizes
    the sentences as one batch: each chunk is (sentences, samples), one zero-padded row per sentence.
    Sentence k is the constant level k / 10, so the output shows which sentences were played."""
    sentences = [s for s in re.split(r"(?<=\.\s)", text) if s]
    speech = [np.full(4000 * k, k / 10, dtype=np.float32) for k in range(1, len(sentences) + 1)]
    rows = speech if split_text else [np.concatenate(speech)]
    wavs = np.zeros((len(rows), max(len(row) for row in rows)), dtype=np.float32)
    for i, row in enumerate(rows):
        wavs[i, : len(row)] = row
    return iter([wavs[:, start : start + 6000] for start in range(0, wavs.shape[1], 6000)])


def test_streaming_speaks_every_sentence_of_the_input(handler_factory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Playing row 0 of each chunk used to drop every sentence after the first."""
    from speech_to_speech.TTS import chatTTS_handler

    # Keep the samples as they are so each sentence's level can be read back.
    monkeypatch.setattr(chatTTS_handler, "librosa", SimpleNamespace(resample=lambda chunk, **_kwargs: chunk))
    handler = handler_factory([])
    handler.model = SimpleNamespace(infer=_stream_like_chattts)

    audio = np.concatenate(list(handler.process(TTSInput(text="One sentence here. Another one here. A third one."))))
    levels = np.rint(audio[audio != 0] / 32768 * 10).astype(int).tolist()

    # Each sentence plays in full and in order, not interleaved with the others.
    runs = [(level, len(list(group))) for level, group in itertools.groupby(levels)]
    assert runs == [(1, 4000), (2, 8000), (3, 12000)]
