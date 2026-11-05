from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from huggingface_hub.errors import HfHubHTTPError
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.TTS import kokoro_handler
from speech_to_speech.TTS.kokoro_handler import KokoroTTSHandler


@pytest.fixture
def kokoro_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    calls: dict[str, list] = {"pipelines": [], "voices": []}

    class FakePipeline:
        def __init__(self, *, lang_code: str, device: str) -> None:
            calls["pipelines"].append(lang_code)

        def __call__(self, _text: str, *, voice: str, speed: float) -> Iterator[tuple[str, str, np.ndarray]]:
            calls["voices"].append(voice)
            return iter(())

    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KPipeline=FakePipeline))
    monkeypatch.setattr(kokoro_handler, "platform", "linux")
    monkeypatch.setattr(KokoroTTSHandler, "warmup", lambda self: None)
    monkeypatch.setattr(kokoro_handler.console, "print", lambda *_args, **_kwargs: None)
    return calls


def _handler(lang_code: str, voice: str) -> KokoroTTSHandler:
    handler = KokoroTTSHandler.__new__(KokoroTTSHandler)
    handler.setup(Event(), device="cpu", lang_code=lang_code, voice=voice)
    return handler


def test_english_reply_keeps_configured_american_voice(kokoro_calls: dict[str, list]) -> None:
    handler = _handler("a", "af_heart")

    list(handler._process_kokoro("Hello there.", language_code="en"))

    assert (handler.lang_code, handler.voice) == ("a", "af_heart")
    assert kokoro_calls["pipelines"] == ["a"]
    assert kokoro_calls["voices"] == ["af_heart"]


def test_returning_to_setup_language_restores_configured_voice(kokoro_calls: dict[str, list]) -> None:
    handler = _handler("b", "bf_emma")

    list(handler._process_kokoro("Hola.", language_code="es"))
    list(handler._process_kokoro("Hello again.", language_code="en"))

    assert (handler.lang_code, handler.voice) == ("b", "bf_emma")
    assert kokoro_calls["voices"] == ["ef_dora", "bf_emma"]


@pytest.fixture(params=["kokoro", "mlx"])
def voice_backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, list] = {"spoken": [], "loaded": []}
    pipelines: dict[str, object] = {}

    class Pipeline:
        def __init__(self, *, lang_code: str, device: str = "mps") -> None:
            self.lang_code = lang_code
            self._voice_tensor = None

        def load_voice(self, voice: str) -> None:
            self._voice_tensor = voice
            calls["loaded"].append((self.lang_code, voice))

        def __call__(self, *args, voice: str, **kwargs):
            calls["spoken"].append((self.lang_code, voice, self._voice_tensor))
            return iter(())

    def get_pipeline(lang_code: str):
        if lang_code not in pipelines:
            pipelines[lang_code] = Pipeline(lang_code=lang_code)
        return pipelines[lang_code]

    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KPipeline=Pipeline))
    monkeypatch.setitem(
        sys.modules,
        "mlx_audio.tts.utils",
        SimpleNamespace(load_model=lambda _: SimpleNamespace(_get_pipeline=get_pipeline)),
    )
    monkeypatch.setattr(KokoroTTSHandler, "warmup", lambda self: None)
    monkeypatch.setattr(kokoro_handler.console, "print", lambda *args, **kwargs: None)
    handler = KokoroTTSHandler.__new__(KokoroTTSHandler)
    handler.setup(Event(), device="mps" if request.param == "mlx" else "cpu", lang_code="a", voice="af_heart")
    calls["loaded"].clear()
    return handler, calls


def _session(voice: str) -> RuntimeConfig:
    config = RuntimeConfig()
    config.session.audio.output.voice = voice
    return config


def _speak(handler, config, language="en", response=None):
    return list(
        handler.process(TTSInput(text="Hello", runtime_config=config, tts_language_code=language, response=response))
    )


def test_session_voice_survives_language_switch(voice_backend):
    handler, calls = voice_backend
    config = _session("am_puck")

    _speak(handler, config, "es")
    _speak(handler, config, "en")
    _speak(handler, config, "en")

    assert [voice for _, voice, _ in calls["spoken"]] == ["ef_dora", "am_puck", "am_puck"]
    if handler.backend == "mlx":
        assert calls["loaded"] == [("e", "ef_dora"), ("a", "am_puck")]
        assert handler._pipeline._voice_tensor == "am_puck"


def test_response_voice_overrides_session_only_for_that_response(voice_backend):
    handler, calls = voice_backend
    config = _session("am_puck")
    response = RealtimeResponseCreateParams(audio={"output": {"voice": "am_michael"}})

    _speak(handler, config, "es")
    _speak(handler, config, response=response)
    _speak(handler, config)

    assert [voice for _, voice, _ in calls["spoken"]] == ["ef_dora", "am_michael", "am_puck"]
    if handler.backend == "mlx":
        assert [tensor for _, _, tensor in calls["spoken"]] == ["ef_dora", "am_michael", "am_puck"]


def test_voice_update_without_language_change_reloads_mlx_voice(voice_backend):
    handler, calls = voice_backend
    config = _session("am_puck")
    _speak(handler, config, language=None)
    config.session.audio.output.voice = "am_michael"
    _speak(handler, config, language=None)

    assert [voice for _, voice, _ in calls["spoken"]] == ["am_puck", "am_michael"]
    if handler.backend == "mlx":
        assert calls["loaded"] == [("a", "am_puck"), ("a", "am_michael")]


def test_session_end_restores_cli_voice(voice_backend):
    handler, calls = voice_backend
    _speak(handler, _session("am_puck"))
    handler.on_session_end()
    list(handler.process(TTSInput(text="New session", tts_language_code="en")))

    assert (handler.lang_code, handler.voice) == ("a", "af_heart")
    assert calls["spoken"][-1][1] == "af_heart"
    if handler.backend == "mlx":
        assert handler._pipeline._voice_tensor == "af_heart"


def test_language_switch_without_client_voice_restores_cli_voice(voice_backend):
    handler, calls = voice_backend
    _speak(handler, None, "es")
    _speak(handler, None, "en")
    assert [voice for _, voice, _ in calls["spoken"]] == ["ef_dora", "af_heart"]
    if handler.backend == "mlx":
        assert handler._pipeline._voice_tensor == "af_heart"


def test_client_voice_for_another_language_is_used_in_that_language(voice_backend):
    handler, calls = voice_backend
    _speak(handler, _session("em_alex"), "es")
    assert calls["spoken"][-1][1] == "em_alex"
    if handler.backend == "mlx":
        assert handler._pipeline._voice_tensor == "em_alex"


@pytest.mark.parametrize("language", ["en", "de", "nl", "pl", "ru", "uk"])
def test_english_fallback_uses_configured_variant(kokoro_calls, language):
    handler = _handler("a", "af_heart")
    list(handler._process_kokoro("Hello", language_code=language))
    assert (handler.lang_code, handler.voice) == ("a", "af_heart")


@pytest.fixture
def voice_audio_backend(voice_backend, monkeypatch):
    handler, calls = voice_backend
    pipeline = handler._pipeline if handler.backend == "mlx" else handler.pipeline
    pipeline_type = type(pipeline)
    load_voice = pipeline_type.load_voice
    available = {"af_heart", "af_alloy", "am_puck", "am_michael", "ef_dora"}

    def checked_load(self, voice):
        suffix = ".safetensors" if handler.backend == "mlx" else ".pt"
        if voice.endswith(suffix):
            Path(voice).read_bytes()
        elif not all(part in available for part in voice.split(",")):
            response = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co/missing"))
            raise HfHubHTTPError(f"Missing voice {voice}", response=response)
        load_voice(self, voice)

    def generate(self, *args, voice, **kwargs):
        if handler.backend == "kokoro":
            self.load_voice(voice)
        calls["spoken"].append((self.lang_code, voice, self._voice_tensor))
        audio = np.full(24, 0.1, dtype=np.float32)
        if handler.backend == "mlx":
            yield SimpleNamespace(audio=audio.reshape(1, -1))
        else:
            yield ("Hello", "", audio)

    monkeypatch.setattr(pipeline_type, "load_voice", checked_load)
    monkeypatch.setattr(pipeline_type, "__call__", generate)
    return handler, calls


@pytest.mark.parametrize("requested", ["af_typo", "xx_name", "unknown"])
def test_missing_client_voice_falls_back_and_emits_audio(voice_audio_backend, requested, caplog):
    handler, calls = voice_audio_backend

    chunks = _speak(handler, _session(requested))

    assert calls["spoken"][-1][1:] == ("af_heart", "af_heart")
    assert any(np.any(chunk) for chunk in chunks)
    assert requested in caplog.text
    assert "falling back to 'af_heart'" in caplog.text


def test_missing_voice_restores_setup_voice_after_language_switch(voice_audio_backend):
    handler, calls = voice_audio_backend
    config = _session("unknown")

    _speak(handler, config, "es")
    chunks = _speak(handler, config, "en")

    assert [voice for _, voice, _ in calls["spoken"]] == ["ef_dora", "af_heart"]
    assert any(np.any(chunk) for chunk in chunks)


def test_missing_voice_does_not_reuse_previous_response_voice(voice_audio_backend):
    handler, calls = voice_audio_backend
    config = _session("unknown")
    response = RealtimeResponseCreateParams(audio={"output": {"voice": "am_michael"}})

    _speak(handler, config, response=response)
    _speak(handler, config)

    assert [voice for _, voice, _ in calls["spoken"]] == ["am_michael", "af_heart"]


@pytest.mark.parametrize("relative", [False, True])
def test_client_voice_file_is_preserved(voice_audio_backend, tmp_path, monkeypatch, relative):
    handler, calls = voice_audio_backend
    monkeypatch.chdir(tmp_path)
    suffix = ".safetensors" if handler.backend == "mlx" else ".pt"
    voice_path = Path(f"custom{suffix}")
    voice_path.write_bytes(b"voice fixture")
    requested = str(voice_path if relative else voice_path.resolve())

    chunks = _speak(handler, _session(requested))

    assert calls["spoken"][-1][1] == requested
    assert any(np.any(chunk) for chunk in chunks)


def test_missing_voice_file_falls_back(voice_audio_backend):
    handler, calls = voice_audio_backend
    suffix = ".safetensors" if handler.backend == "mlx" else ".pt"

    chunks = _speak(handler, _session(f"missing{suffix}"))

    assert calls["spoken"][-1][1] == "af_heart"
    assert any(np.any(chunk) for chunk in chunks)


def test_client_voice_blend_is_preserved(voice_audio_backend):
    handler, calls = voice_audio_backend

    chunks = _speak(handler, _session("af_heart,am_puck"))

    assert calls["spoken"][-1][1] == "af_heart,am_puck"
    assert any(np.any(chunk) for chunk in chunks)


@pytest.mark.parametrize("response_override", [False, True])
def test_alloy_alias_emits_audio(voice_audio_backend, response_override):
    handler, calls = voice_audio_backend
    response = RealtimeResponseCreateParams(audio={"output": {"voice": "alloy"}}) if response_override else None
    config = _session("am_puck" if response_override else "alloy")

    chunks = _speak(handler, config, response=response)

    assert calls["spoken"][-1][1] == "af_alloy"
    assert any(np.any(chunk) for chunk in chunks)


def test_failed_fallback_load_is_not_retried(voice_backend):
    handler, _ = voice_backend
    attempted = []

    def fail(voice):
        attempted.append(voice)
        raise FileNotFoundError(voice)

    with pytest.raises(FileNotFoundError, match="af_heart"):
        handler._load_requested_voice(SimpleNamespace(load_voice=fail), "af_typo")
    assert attempted == ["af_typo", "af_heart"]

    attempted.clear()
    with pytest.raises(FileNotFoundError, match="af_heart"):
        handler._load_requested_voice(SimpleNamespace(load_voice=fail), "af_heart")
    assert attempted == ["af_heart"]


def test_synthesis_failure_does_not_retry_with_another_voice(voice_audio_backend, monkeypatch):
    handler, calls = voice_audio_backend
    pipeline = handler._pipeline if handler.backend == "mlx" else handler.pipeline

    def fail(self, *args, **kwargs):
        raise RuntimeError("synthesis failed")

    monkeypatch.setattr(type(pipeline), "__call__", fail)

    with pytest.raises(RuntimeError, match="synthesis failed"):
        _speak(handler, _session("am_puck"))
    assert calls["loaded"] == [("a", "am_puck")]
