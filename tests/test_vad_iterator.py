import torch

from VAD.vad_iterator import VADIterator


class _FakeVADModel:
    def __init__(self, probs: list[float]) -> None:
        self._probs = iter(probs)

    def reset_states(self) -> None:
        pass

    def __call__(self, x: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        return torch.tensor(next(self._probs), dtype=torch.float32)


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
    spoken_utterance = None
    for _ in range(5):
        spoken_utterance = iterator(silence_chunk)
        if spoken_utterance is not None:
            break

    assert spoken_utterance is not None
    assert len(spoken_utterance) == 2
    assert torch.equal(spoken_utterance[0], first_chunk)
    assert torch.equal(spoken_utterance[1], second_chunk)
