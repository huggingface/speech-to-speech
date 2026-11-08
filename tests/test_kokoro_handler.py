from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import resample_poly

from speech_to_speech.TTS.kokoro_handler import KokoroTTSHandler


@pytest.mark.parametrize("backend", ["kokoro", "mlx"])
@pytest.mark.parametrize("lengths", [(1000, 1000, 1000), (768, 3, 900)])
def test_provider_segments_preserve_pcm_without_padding(backend, lengths):
    handler = object.__new__(KokoroTTSHandler)
    handler.cancel_scope = None
    handler.voice = "bm_fable"
    handler.lang_code = "b"
    handler.speed = 1.0
    handler.blocksize = 512
    # Keep samples above the MLX silence-trimming threshold.
    segments = [np.linspace(0.1, 0.5, length, dtype=np.float32) for length in lengths]
    if backend == "kokoro":
        handler.pipeline = lambda *args, **kwargs: iter([("", "", None)] + [("", "", audio) for audio in segments])
    else:
        handler._pipeline = lambda **kwargs: iter(
            [SimpleNamespace(audio=None)] + [SimpleNamespace(audio=audio[np.newaxis, :]) for audio in segments]
        )
    chunks = list(getattr(handler, f"_process_{backend}")("Test segments."))
    expected = np.concatenate([(resample_poly(audio, up=2, down=3) * 32768).astype(np.int16) for audio in segments])
    assert all(chunk.dtype == np.int16 and 0 < len(chunk) <= 512 for chunk in chunks)
    np.testing.assert_array_equal(np.concatenate(chunks), expected)
