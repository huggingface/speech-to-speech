from types import SimpleNamespace

import numpy as np
import torch

from speech_to_speech.VAD.firered_vad_iterator import FireRedVadIterator
from speech_to_speech.VAD.vad_iterator import VADIterator


def _frame(
    *,
    smoothed_prob: float = 0.9,
    is_speech_start: bool = False,
    is_speech_end: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        smoothed_prob=smoothed_prob,
        is_speech_start=is_speech_start,
        is_speech_end=is_speech_end,
    )


class _FakeFireRedStream:
    def __init__(self, frames: list[SimpleNamespace] | None = None) -> None:
        self._source = list(frames or [])
        self._frames = iter(self._source)
        self.reset_calls = 0
        self.chunks: list[int] = []
        self.audio: list = []
        self.frame_count = 0

    def reset(self) -> None:
        self.reset_calls += 1
        self._frames = iter(self._source)

    def detect_chunk(self, audio_chunk) -> list[SimpleNamespace]:
        self.chunks.append(len(audio_chunk))
        self.audio.append(audio_chunk)
        length = len(audio_chunk)
        n_frames = (length - 400) // 160 + 1 if length >= 400 else 0
        self.frame_count += n_frames
        results = []
        for _ in range(n_frames):
            results.append(next(self._frames, _frame(smoothed_prob=0.0)))
        return results


def test_firered_iterator_returns_on_firered_speech_end() -> None:
    streamer = _FakeFireRedStream(
        [
            _frame(is_speech_start=True),
            _frame(is_speech_end=True),
        ]
    )
    iterator = FireRedVadIterator(
        streamer,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=0,
    )

    first_chunk = torch.ones(512)
    second_chunk = torch.ones(512) * 2

    assert iterator(first_chunk) is None
    assert iterator.triggered is True
    spoken_utterance = iterator(second_chunk)

    assert spoken_utterance is not None
    assert iterator.triggered is False
    assert torch.equal(spoken_utterance[0], first_chunk)
    assert torch.equal(spoken_utterance[1], second_chunk)


def test_firered_iterator_does_not_trigger_without_speech_start() -> None:
    streamer = _FakeFireRedStream([_frame(), _frame(), _frame()])
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


def test_firered_iterator_scales_silero_audio_to_int16_peak() -> None:
    streamer = _FakeFireRedStream([_frame()])
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
    assert len(chunk) == 400
    assert chunk.dtype == np.float32
    assert float(chunk.max()) == 32768.0
    assert float(chunk.min()) == 32768.0


def test_firered_iterator_reset_states_clears_trigger_and_streamer() -> None:
    streamer = _FakeFireRedStream([_frame(is_speech_start=True), _frame()])
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


def test_firered_iterator_overlaps_successive_512_sample_chunks() -> None:
    streamer = _FakeFireRedStream([_frame(), _frame()])
    iterator = FireRedVadIterator(
        streamer,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=0,
    )
    first_chunk = torch.arange(512, dtype=torch.float32) / 512
    iterator(first_chunk)
    iterator(torch.ones(512, dtype=torch.float32))

    assert streamer.chunks[0] == 400
    first_scaled = first_chunk.numpy() * 32768.0
    second_feed = np.asarray(streamer.audio[1])
    assert np.array_equal(second_feed[: 512 - 160], first_scaled[160:])


def test_firered_iterator_keeps_continuous_frame_count_across_512_chunks() -> None:
    streamer = _FakeFireRedStream()
    iterator = FireRedVadIterator(
        streamer,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=0,
    )
    chunk = torch.ones(512)
    for _ in range(100):
        iterator(chunk)
    assert streamer.frame_count == 318


def test_firered_iterator_reset_states_clears_waveform_tail() -> None:
    streamer = _FakeFireRedStream([_frame(), _frame()])
    iterator = FireRedVadIterator(
        streamer,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=0,
    )
    iterator(torch.ones(512))
    iterator.reset_states()
    iterator(torch.full((512,), 0.25))

    fed = np.asarray(streamer.audio[-1])
    assert len(fed) == 400
    assert float(fed.max()) == 0.25 * 32768.0
    assert float(fed.min()) == 0.25 * 32768.0


def test_firered_iterator_is_not_vad_iterator() -> None:
    iterator = FireRedVadIterator(
        _FakeFireRedStream(),
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=0,
    )
    assert isinstance(iterator, VADIterator) is False
    iterator.threshold = 0.7
    iterator.min_silence_samples = 1600
    assert iterator.threshold == 0.7
    assert iterator.min_silence_samples == 1600
