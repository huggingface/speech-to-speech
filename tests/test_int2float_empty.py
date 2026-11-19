"""Empty PCM chunks must not crash the VAD conversion helper."""

import numpy as np

from speech_to_speech.utils.utils import int2float


def test_int2float_accepts_empty_input():
    result = int2float(np.array([], dtype=np.int16))
    assert result.dtype == np.float32
    assert result.size == 0
