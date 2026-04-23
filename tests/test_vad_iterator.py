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
        speech_pad_ms=0,
        start_persistence_ms=0,
    )

    first_chunk = torch.ones(512)
    second_chunk = torch.ones(512) * 2
    silence_chunk = torch.zeros(512)

    assert iterator(first_chunk) is None
    assert iterator(second_chunk) is None
    spoken_utterance = _finish_utterance(iterator, silence_chunk)

    assert spoken_utterance is not None
    assert len(spoken_utterance) == 7
    assert torch.equal(spoken_utterance[0], first_chunk)
    assert torch.equal(spoken_utterance[1], second_chunk)
    assert all(torch.equal(chunk, silence_chunk) for chunk in spoken_utterance[2:])


def test_pre_speech_padding_is_prepended_to_final_utterance() -> None:
    model = _FakeVADModel([0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    iterator = VADIterator(
        model=model,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=64,
        start_persistence_ms=0,
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
    assert len(spoken_utterance) == 9
    assert torch.equal(spoken_utterance[0], first_chunk)
    assert torch.equal(spoken_utterance[1], second_chunk)
    assert torch.equal(spoken_utterance[2], third_chunk)
    assert torch.equal(spoken_utterance[3], fourth_chunk)
    assert all(torch.equal(chunk, silence_chunk) for chunk in spoken_utterance[4:])


def test_speech_buffer_keeps_prefix_out_of_active_speech_buffer() -> None:
    model = _FakeVADModel([0.1, 0.1, 0.9])
    iterator = VADIterator(
        model=model,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=32,
        start_persistence_ms=0,
    )

    older_chunk = torch.ones(512)
    latest_pre_speech_chunk = torch.ones(512) * 2
    triggering_chunk = torch.ones(512) * 3

    assert iterator(older_chunk) is None
    assert iterator(latest_pre_speech_chunk) is None
    assert iterator(triggering_chunk) is None

    assert len(iterator.buffer) == 1
    assert torch.equal(iterator.buffer[0], triggering_chunk)

    speech_buffer = iterator.speech_buffer()
    assert len(speech_buffer) == 2
    assert torch.equal(speech_buffer[0], latest_pre_speech_chunk)
    assert torch.equal(speech_buffer[1], triggering_chunk)


def test_final_samples_are_kept_until_vad_declares_done() -> None:
    model = _FakeVADModel([0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    iterator = VADIterator(
        model=model,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=64,
        start_persistence_ms=0,
    )

    first_chunk = torch.ones(512)
    second_chunk = torch.ones(512) * 2
    trailing_chunks = [torch.ones(512) * value for value in (10, 11, 12, 13, 14)]

    assert iterator(first_chunk) is None
    assert iterator(second_chunk) is None

    spoken_utterance = None
    for chunk in trailing_chunks:
        spoken_utterance = iterator(chunk)

    assert spoken_utterance is not None
    assert len(spoken_utterance) == 7
    assert torch.equal(spoken_utterance[0], first_chunk)
    assert torch.equal(spoken_utterance[1], second_chunk)
    assert torch.equal(spoken_utterance[2], trailing_chunks[0])
    assert torch.equal(spoken_utterance[3], trailing_chunks[1])
    assert torch.equal(spoken_utterance[4], trailing_chunks[2])
    assert torch.equal(spoken_utterance[5], trailing_chunks[3])
    assert torch.equal(spoken_utterance[6], trailing_chunks[4])


def test_brief_silence_is_preserved_when_speech_resumes() -> None:
    model = _FakeVADModel([0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    iterator = VADIterator(
        model=model,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=0,
        start_persistence_ms=0,
    )

    first_chunk = torch.ones(512)
    pause_chunks = [torch.ones(512) * value for value in (8, 9)]
    resumed_chunk = torch.ones(512) * 2
    ending_silence = torch.zeros(512)

    assert iterator(first_chunk) is None
    assert iterator(pause_chunks[0]) is None
    assert iterator(pause_chunks[1]) is None
    assert iterator(resumed_chunk) is None

    spoken_utterance = _finish_utterance(iterator, ending_silence)

    assert spoken_utterance is not None
    assert len(spoken_utterance) == 9
    assert torch.equal(spoken_utterance[0], first_chunk)
    assert torch.equal(spoken_utterance[1], pause_chunks[0])
    assert torch.equal(spoken_utterance[2], pause_chunks[1])
    assert torch.equal(spoken_utterance[3], resumed_chunk)
    assert all(torch.equal(chunk, ending_silence) for chunk in spoken_utterance[4:])


def test_start_persistence_requires_contiguous_speech_before_triggering() -> None:
    model = _FakeVADModel([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    iterator = VADIterator(
        model=model,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=0,
        start_persistence_ms=128,
    )

    speech_chunks = [torch.ones(512) * value for value in (1, 2, 3, 4)]
    silence_chunk = torch.zeros(512)

    for chunk in speech_chunks[:3]:
        assert iterator(chunk) is None
        assert not iterator.triggered
        assert iterator.buffer == []

    assert iterator(speech_chunks[3]) is None
    assert iterator.triggered
    assert len(iterator.buffer) == 4
    for buffered, expected in zip(iterator.buffer, speech_chunks):
        assert torch.equal(buffered, expected)

    spoken_utterance = _finish_utterance(iterator, silence_chunk)

    assert spoken_utterance is not None
    assert len(spoken_utterance) == 9
    for buffered, expected in zip(spoken_utterance[:4], speech_chunks):
        assert torch.equal(buffered, expected)
    assert all(torch.equal(chunk, silence_chunk) for chunk in spoken_utterance[4:])


def test_false_start_is_discarded_when_start_persistence_is_not_met() -> None:
    model = _FakeVADModel(
        [
            0.9,
            0.9,
            0.1,
            0.9,
            0.9,
            0.9,
            0.9,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
        ]
    )
    iterator = VADIterator(
        model=model,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=32,
        start_persistence_ms=128,
    )

    false_start_chunks = [torch.ones(512) * value for value in (1, 2)]
    separating_silence = torch.zeros(512)
    real_speech_chunks = [torch.ones(512) * value for value in (10, 11, 12, 13)]

    for chunk in false_start_chunks:
        assert iterator(chunk) is None
        assert not iterator.triggered
        assert iterator.buffer == []

    assert iterator(separating_silence) is None
    assert not iterator.triggered
    assert iterator.buffer == []

    for chunk in real_speech_chunks[:-1]:
        assert iterator(chunk) is None
        assert not iterator.triggered
        assert iterator.buffer == []

    assert iterator(real_speech_chunks[-1]) is None
    assert iterator.triggered

    spoken_utterance = _finish_utterance(iterator, separating_silence)

    assert spoken_utterance is not None
    assert len(spoken_utterance) == 10
    assert torch.equal(spoken_utterance[0], separating_silence)
    for buffered, expected in zip(spoken_utterance[1:5], real_speech_chunks):
        assert torch.equal(buffered, expected)
    assert all(torch.equal(chunk, separating_silence) for chunk in spoken_utterance[5:])
    assert all(
        not torch.equal(chunk, false_start_chunks[0]) for chunk in spoken_utterance
    )
    assert all(
        not torch.equal(chunk, false_start_chunks[1]) for chunk in spoken_utterance
    )
