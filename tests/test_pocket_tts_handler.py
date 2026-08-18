from threading import Event
from types import SimpleNamespace

import pytest

from speech_to_speech.TTS.pocket_tts_handler import PocketTTSHandler


@pytest.mark.parametrize(
    ("setup_kwargs", "expected_language"),
    [
        ({}, "english"),
        ({"language": "french_24l"}, "french_24l"),
    ],
)
def test_pocket_tts_setup_loads_language(monkeypatch, setup_kwargs, expected_language):
    import pocket_tts

    loaded_languages = []

    fake_model = SimpleNamespace(
        to=lambda *args, **kwargs: None,
        get_state_for_audio_prompt=lambda *args, **kwargs: object(),
        sample_rate=24000,
    )

    def fake_load_model(*, language):
        loaded_languages.append(language)
        return fake_model

    monkeypatch.setattr(
        pocket_tts.TTSModel,
        "load_model",
        fake_load_model,
    )

    handler = object.__new__(PocketTTSHandler)

    handler.setup(
        should_listen=Event(),
        **setup_kwargs,
    )

    assert loaded_languages == [expected_language]
