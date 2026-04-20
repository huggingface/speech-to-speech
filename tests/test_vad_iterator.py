import torch

from VAD.vad_iterator import VADIterator


class _FakeVADModel:
    def __init__(self, probs: list[float]) -> None:
        self._probs = iter(probs)

    def reset_states(self) -> None:
        pass

    def __call__(self, x: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        return torch.tensor(next(self._probs), dtype=torch.float32)


def _finish_utterance(iterator: VADIterator, silence_chunk: torch.Tensor):
    spoken_utterance = None
    for _ in range(5):
        spoken_utterance = iterator(silence_chunk)
        if spoken_utterance is not None:
            break
    return spoken_utterance


def test_triggering_chunk_is_kept_in_buffer() -> None:
    model = _FakeVADModel([0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    iterator = VADIterator(
        model=model,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
    )

    first_chunk = torch.ones(512)
    second_chunk = torch.ones(512) * 2
    silence_chunk = torch.zeros(512)

    assert iterator(first_chunk) is None
    assert iterator(second_chunk) is None
    spoken_utterance = _finish_utterance(iterator, silence_chunk)

    assert spoken_utterance is not None
    assert len(spoken_utterance) == 2
    assert torch.equal(spoken_utterance[0], first_chunk)
    assert torch.equal(spoken_utterance[1], second_chunk)


def test_pre_speech_padding_is_prepended_to_final_utterance() -> None:
    model = _FakeVADModel([0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    iterator = VADIterator(
        model=model,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=64,
    )

    first_chunk = torch.ones(512)
    second_chunk = torch.ones(512) * 2
    third_chunk = torch.ones(512) * 3
    fourth_chunk = torch.ones(512) * 4
    silence_chunk = torch.zeros(512)

    assert iterator(first_chunk) is None
    assert iterator(second_chunk) is None
    assert iterator(third_chunk) is None
    assert iterator(fourth_chunk) is None

    spoken_utterance = _finish_utterance(iterator, silence_chunk)

    assert spoken_utterance is not None
    assert len(spoken_utterance) == 4
    assert torch.equal(spoken_utterance[0], first_chunk)
    assert torch.equal(spoken_utterance[1], second_chunk)
    assert torch.equal(spoken_utterance[2], third_chunk)
    assert torch.equal(spoken_utterance[3], fourth_chunk)


def test_padding_view_keeps_prefix_out_of_active_speech_buffer() -> None:
    model = _FakeVADModel([0.1, 0.1, 0.9])
    iterator = VADIterator(
        model=model,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=32,
    )

    older_chunk = torch.ones(512)
    latest_pre_speech_chunk = torch.ones(512) * 2
    triggering_chunk = torch.ones(512) * 3

    assert iterator(older_chunk) is None
    assert iterator(latest_pre_speech_chunk) is None
    assert iterator(triggering_chunk) is None

    assert len(iterator.buffer) == 1
    assert torch.equal(iterator.buffer[0], triggering_chunk)

    padded_buffer = iterator.buffer_with_pad()
    assert len(padded_buffer) == 2
    assert torch.equal(padded_buffer[0], latest_pre_speech_chunk)
    assert torch.equal(padded_buffer[1], triggering_chunk)
