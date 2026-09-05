from types import SimpleNamespace

import numpy as np
import torch

from speech_to_speech.VAD.firered_vad_iterator import FireRedVadIterator
from tests.test_vad_iterator import _finish_utterance


class _FakeFireRedStream:
    def __init__(self, probs: list[float]) -> None:
        self._source = probs
        self._probs = iter(probs)
        self.reset_calls = 0
        self.chunks: list[int] = []
        self.audio: list = []

    def reset(self) -> None:
        self.reset_calls += 1
        self._probs = iter(self._source)

    def detect_chunk(self, audio_chunk) -> list[SimpleNamespace]:
        self.chunks.append(len(audio_chunk))
        self.audio.append(audio_chunk)
        return [SimpleNamespace(smoothed_prob=next(self._probs))]


def test_firered_iterator_keeps_speech_then_returns_on_silence() -> None:
    streamer = _FakeFireRedStream([0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    iterator = FireRedVadIterator(
        streamer,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=0,
    )

    first_chunk = torch.ones(512)
    second_chunk = torch.ones(512) * 2
    silence_chunk = torch.zeros(512)

    assert iterator(first_chunk) is None
    assert iterator.triggered is True
    assert iterator(second_chunk) is None
    spoken_utterance = _finish_utterance(iterator, silence_chunk)

    assert spoken_utterance is not None
    assert iterator.triggered is False
    assert torch.equal(spoken_utterance[0], first_chunk)
    assert torch.equal(spoken_utterance[1], second_chunk)


def test_firered_iterator_does_not_trigger_below_threshold() -> None:
    streamer = _FakeFireRedStream([0.2, 0.2, 0.2])
    iterator = FireRedVadIterator(
        streamer,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=0,
    )

    chunk = torch.ones(512)
    assert iterator(chunk) is None
    assert iterator.triggered is False
    assert iterator(chunk) is None
    assert iterator.triggered is False


def test_firered_prob_model_scales_silero_audio_to_int16_peak() -> None:
    streamer = _FakeFireRedStream([0.9])
    iterator = FireRedVadIterator(
        streamer,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=0,
    )

    assert iterator(torch.ones(512)) is None
    assert len(streamer.audio) == 1
    chunk = np.asarray(streamer.audio[0])
    assert chunk.dtype == np.float32
    assert float(chunk.max()) == 32768.0
    assert float(chunk.min()) == 32768.0


def test_firered_iterator_reset_states_clears_trigger_and_streamer() -> None:
    streamer = _FakeFireRedStream([0.9, 0.9])
    iterator = FireRedVadIterator(
        streamer,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=0,
    )

    assert iterator(torch.ones(512)) is None
    assert iterator.triggered is True
    resets_before = streamer.reset_calls
    iterator.reset_states()
    assert iterator.triggered is False
    assert iterator.buffer == []
    assert streamer.reset_calls == resets_before + 1
