import sys
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.signal import resample_poly

from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.TTS import pocket_tts_handler
from speech_to_speech.TTS.pocket_tts_handler import PocketTTSHandler


@pytest.mark.parametrize(
    ("setup_kwargs", "expected_language"),
    [
        ({}, "english"),
        ({"language": "french_24l"}, "french_24l"),
    ],
)
def test_pocket_tts_setup_loads_language(monkeypatch, setup_kwargs, expected_language):
    loaded_languages = []

    fake_model = SimpleNamespace(
        to=lambda *args, **kwargs: None,
        get_state_for_audio_prompt=lambda *args, **kwargs: object(),
        sample_rate=24000,
    )

    def fake_load_model(*, language):
        loaded_languages.append(language)
        return fake_model

    fake_pocket_tts = SimpleNamespace(
        TTSModel=SimpleNamespace(load_model=fake_load_model),
    )

    monkeypatch.setitem(sys.modules, "pocket_tts", fake_pocket_tts)
    handler = object.__new__(PocketTTSHandler)

    handler.setup(
        should_listen=Event(),
        **setup_kwargs,
    )

    assert loaded_languages == [expected_language]


def test_pocket_tts_resampled_output_is_continuous_across_generated_chunks(monkeypatch):
    # Pocket TTS streams 24 kHz audio in 80 ms frames; the pipeline plays 16 kHz.
    samples = (0.5 * np.sin(2 * np.pi * 220 * np.arange(24000) / 24000)).astype(np.float32)
    frames = [torch.from_numpy(frame) for frame in np.split(samples, 12)]

    handler = object.__new__(PocketTTSHandler)
    handler.model = SimpleNamespace(
        sample_rate=24000,
        generate_audio_stream=lambda *_args, **_kwargs: iter(frames),
    )
    handler.voice_state = object()
    handler.sample_rate = 16000
    handler.blocksize = 512
    handler.max_tokens = 50
    handler.cancel_scope = None
    handler.speculative_turns = None
    monkeypatch.setattr(pocket_tts_handler.console, "print", lambda *_args, **_kwargs: None)

    blocks = list(handler.process(TTSInput(text="Hello there.")))

    assert all(len(block) == 512 for block in blocks)
    output = np.concatenate(blocks)
    reference = np.clip(np.round(resample_poly(samples.astype(np.float64) * 32768, 2, 3)), -32768, 32767)
    assert np.abs(output[: reference.size].astype(np.int32) - reference.astype(np.int32)).max() <= 1
    assert not output[reference.size :].any()
