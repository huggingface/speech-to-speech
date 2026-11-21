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


def test_every_kokoro_voice_is_offered_once_under_the_language_it_was_made_for():
    from speech_to_speech.TTS.kokoro_handler import (
        KOKORO_LANG_DEFAULT_VOICES,
        KOKORO_LANGUAGE_NAMES,
        KOKORO_VOICES,
        kokoro_voice_choices,
    )

    choices = kokoro_voice_choices()
    ids = [choice["id"] for choice in choices]
    assert len(ids) == len(set(ids)) == 54
    assert set(KOKORO_LANG_DEFAULT_VOICES.values()) <= set(ids), "each language's default voice can be chosen"
    for code, voices in KOKORO_VOICES.items():
        assert all(voice[0] == code and voice[1] in "fm" for voice in voices)
    heart = next(choice for choice in choices if choice["id"] == "af_heart")
    assert heart == {"id": "af_heart", "name": "Heart (female)", "group": KOKORO_LANGUAGE_NAMES["a"]}


def test_kokoro_reports_the_voice_it_speaks_with_by_default():
    handler = object.__new__(KokoroTTSHandler)
    handler._initial_voice = "bm_fable"
    described = handler.voice_capabilities()
    assert described["default_voice"] == "bm_fable"
    assert any(voice["id"] == "bm_fable" for voice in described["voices"])
