import numpy as np
import pytest

from speech_to_speech.TTS.pcm_blocks import PCMBlockBuffer


@pytest.mark.parametrize("length", [0, 1, 512, 1537])
def test_blocks_preserve_samples_across_input_boundaries(length):
    samples = np.arange(length, dtype=np.int16)
    for parts in ([samples], np.array_split(samples, 17)):
        blocks = PCMBlockBuffer(512)
        output = [block for part in parts for block in blocks.push(part)]
        assert all(block.size == 512 for block in output)
        output.extend(blocks.flush())
        actual = np.concatenate(output) if output else np.empty(0, dtype=np.int16)
        np.testing.assert_array_equal(actual, samples)
        assert list(blocks.flush()) == []


def test_blocks_own_samples_when_provider_reuses_storage():
    samples = np.arange(700, dtype=np.int16)
    expected = samples.copy()
    blocks = PCMBlockBuffer(512)
    output = list(blocks.push(samples))
    samples[:] = -1
    output.extend(blocks.flush())
    np.testing.assert_array_equal(np.concatenate(output), expected)


@pytest.mark.parametrize("blocksize", [0, -1])
def test_blocks_reject_nonpositive_size(blocksize):
    with pytest.raises(ValueError, match="blocksize must be positive"):
        PCMBlockBuffer(blocksize)
