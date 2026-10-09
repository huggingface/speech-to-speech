import pytest
from openai.types.realtime import RealtimeSessionCreateRequest

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.TTS.facebookmms_handler import FacebookMMSTTSHandler
from speech_to_speech.TTS.kokoro_handler import KokoroTTSHandler


def _auto_config() -> RuntimeConfig:
    return _session_config("auto")


def _session_config(language: str) -> RuntimeConfig:
    return RuntimeConfig(
        session=RealtimeSessionCreateRequest(
            type="realtime", audio={"input": {"transcription": {"language": language}}}
        )
    )


def test_kokoro_short_auto_reply_restores_setup_voice_after_spanish(monkeypatch):
    calls = []

    class FakePipeline:
        def load_voice(self, voice):
            calls.append(("voice", voice))
            return object()

    class FakeModel:
        def _get_pipeline(self, language):
            calls.append(("language", language))
            return FakePipeline()

    handler = KokoroTTSHandler.__new__(KokoroTTSHandler)
    handler.backend = "mlx"
    handler.model = FakeModel()
    handler.lang_code = "e"
    handler.voice = "ef_dora"
    handler._initial_lang_code = "b"
    handler._initial_voice = "bm_fable"
    monkeypatch.setattr(handler, "_process_mlx", lambda _text, _language, *, pinned_voice=None: iter(()))

    list(handler.process(TTSInput(text="Sure.", language_code=None, runtime_config=_auto_config())))

    assert handler.lang_code == "b"
    assert handler.voice == "bm_fable"
    assert calls == [("language", "b"), ("voice", "bm_fable")]


def test_facebook_mms_short_auto_reply_restores_setup_model_after_spanish(monkeypatch):
    calls = []

    def fake_load_model(self, language, model_name=None):
        calls.append((language, model_name))
        self.language = language
        self.model_name = model_name or f"default/{language}"

    monkeypatch.setattr(FacebookMMSTTSHandler, "load_model", fake_load_model)
    monkeypatch.setattr(FacebookMMSTTSHandler, "generate_audio", lambda self, _text: None)
    handler = FacebookMMSTTSHandler.__new__(FacebookMMSTTSHandler)
    handler._initial_language = "en"
    handler._initial_model_name = "acme/custom-mms"
    handler.language = "es"
    handler.model_name = "default/es"
    handler.cancel_scope = None

    list(handler.process(TTSInput(text="Sure.", language_code=None, runtime_config=_auto_config())))

    assert calls == [("en", "acme/custom-mms")]


@pytest.mark.parametrize(
    ("detected", "expected"),
    [("fr", "fr"), ("de", "es"), ("fi", "es")],  # German maps to English; Finnish is unmapped.
)
def test_kokoro_detected_language_without_native_voice_keeps_session_language(monkeypatch, detected, expected):
    languages = []
    handler = KokoroTTSHandler.__new__(KokoroTTSHandler)
    handler.backend = "mlx"
    handler.voice = "ef_dora"
    monkeypatch.setattr(
        handler, "_process_mlx", lambda _text, language, *, pinned_voice=None: languages.append(language) or iter(())
    )

    list(handler.process(TTSInput(text="Hallo.", tts_language_code=detected, runtime_config=_session_config("es"))))

    assert languages == [expected]


@pytest.mark.parametrize(("detected", "expected_loads"), [("fr", [("fr", None)]), ("ja", [])])
def test_facebook_mms_detected_language_without_checkpoint_keeps_session_language(
    monkeypatch, detected, expected_loads
):
    calls = []

    def fake_load_model(self, language, model_name=None):
        calls.append((language, model_name))
        self.language = language

    monkeypatch.setattr(FacebookMMSTTSHandler, "load_model", fake_load_model)
    monkeypatch.setattr(FacebookMMSTTSHandler, "generate_audio", lambda self, _text: None)
    handler = FacebookMMSTTSHandler.__new__(FacebookMMSTTSHandler)
    handler._initial_language = "en"
    handler._initial_model_name = None
    handler.language = "es"
    handler.model_name = "default/es"
    handler.cancel_scope = None

    list(handler.process(TTSInput(text="Hola.", tts_language_code=detected, runtime_config=_session_config("es"))))

    assert calls == expected_loads
