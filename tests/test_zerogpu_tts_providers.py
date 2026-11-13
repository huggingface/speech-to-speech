"""Provider selection uses native inference without loading weights or CUDA."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from speech_to_speech.zerogpu import _tts_provider


def test_unknown_provider_fails_before_model_import():
    with pytest.raises(ValueError, match="Unsupported ZeroGPU TTS backend"):
        _tts_provider.create_provider("missing", "model", "voice")


def test_qwen_provider_delegates_native_language_voice_and_stream(monkeypatch):
    from speech_to_speech.zerogpu import _model_qwen_tts

    calls = []

    class Handler:
        speaker = "Aiden"

        def _normalize_language(self, language):
            return language

        def _apply_session_voice_override(self, mode, response):
            requested = response.audio.output.voice
            self.speaker = {"aiden": "Aiden", "vivian": "Vivian"}.get(requested.lower(), self.speaker)

        def _language_for_utterance(self, language, selected):
            calls.append((language, selected))
            return "french" if language == "fr" else "english"

    handler = Handler()
    monkeypatch.setattr(_model_qwen_tts, "load_tts", lambda model, voice: handler)
    monkeypatch.setattr(_model_qwen_tts, "initialize_tts_graphs", lambda value: calls.append("graphs"))

    def synthesize(value, text, language, speaker):
        calls.append((text, language, speaker))
        yield np.ones(512, dtype=np.int16)
        calls.append("second")
        yield np.zeros(512, dtype=np.int16)

    monkeypatch.setattr(_model_qwen_tts, "synthesize_chunks16k", synthesize)
    provider = _tts_provider.create_provider("qwen3", "model", "Aiden")
    assert provider.resolve_voice("vivian", "Aiden") == "Vivian"
    assert provider.resolve_voice("invalid", "Vivian") == "Vivian"
    assert handler.speaker == "Aiden", "Metadata resolution cannot mutate shared handler"
    provider.initialize_lease()
    chunks = provider.synthesize_chunks("Bonjour", "fr", "Vivian")
    assert next(chunks).dtype == np.int16
    assert "second" not in calls
    assert ("Bonjour", "french", "Vivian") in calls
    assert len(list(chunks)) == 1


def test_kokoro_provider_preserves_native_process_and_shares_language_models(monkeypatch):
    import sys

    from speech_to_speech.zerogpu import _model_kokoro as module

    created, inference = [], []

    class Model:
        def __init__(self, repo_id):
            self.repo_id = repo_id

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def parameters(self):
            yield SimpleNamespace(device=SimpleNamespace(type=self.device))

    class Pipeline:
        def __init__(self, lang_code, repo_id, model):
            self.lang_code, self.model = lang_code, model
            self.voices = []
            created.append(self)

        def load_voice(self, voice):
            self.voices.append(voice)

        def __call__(self, text, voice, speed):
            inference.append((self.lang_code, text, voice, speed))
            yield text, "phonemes", np.ones(768, dtype=np.float32) * 0.1

    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KModel=Model, KPipeline=Pipeline))
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch, "inference_mode", nullcontext)
    provider = _tts_provider.create_provider("kokoro", "hexgrad/Kokoro-82M", "bm_fable")
    assert inference == [], "No inference before GPU allocation"
    assert provider.resolve_voice("alloy", "bm_fable") == "af_alloy"
    assert provider.default_language == "en"
    assert provider.gpu_models()["tts"] is created[0].model
    first = list(provider.synthesize_chunks("Hello", "en", "alloy"))
    assert first[0].dtype == np.int16
    assert first[0].shape == (512,)
    assert inference[-1][2] == "af_alloy"
    list(provider.synthesize_chunks("Bonjour", "fr", "ff_siwis"))
    assert len(created) == 2
    assert created[0].model is created[1].model
    assert inference[-1][0:3] == ("f", "Bonjour", "ff_siwis")
    list(provider.synthesize_chunks("Back again", "en"))
    assert len(created) == 2, "Returning to a language reuses its pipeline"
    assert inference[-1][2] == "bm_fable", "Prior request voice must not leak"
    provider.warmup()
    assert inference[-1][1] == "Hello."


def test_kokoro_rejects_cpu_fallback(monkeypatch):
    from speech_to_speech.zerogpu import _model_kokoro as module

    provider = object.__new__(module.KokoroTTSProvider)
    provider.handler = SimpleNamespace()
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="allocated CUDA"):
        provider.initialize_lease()


def test_provider_catalog_defaults_and_imports_are_lightweight(monkeypatch):
    import os
    import subprocess
    import sys
    from unittest.mock import Mock

    from speech_to_speech.zerogpu import _model_kokoro

    qwen, kokoro = Mock(), Mock()
    monkeypatch.setattr(_tts_provider, "QwenTTSProvider", qwen)
    monkeypatch.setattr(_model_kokoro, "KokoroTTSProvider", kokoro)
    _tts_provider.create_provider("qwen3")
    qwen.assert_called_once_with("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "Aiden")
    _tts_provider.create_provider("kokoro")
    kokoro.assert_called_once_with("hexgrad/Kokoro-82M", "bm_fable")
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from speech_to_speech.zerogpu._tts_provider import SUPPORTED_BACKENDS
assert SUPPORTED_BACKENDS == {'qwen3', 'kokoro'}
assert not {'torch', 'spaces', 'kokoro', 'transformers'}.intersection(sys.modules)
""",
        ],
        check=True,
        env=os.environ.copy(),
    )


def test_empty_audio_is_not_successful_warmup():
    provider = object.__new__(_tts_provider.QwenTTSProvider)
    provider.handler = SimpleNamespace(_language_for_utterance=lambda *args: "english")
    provider.native = SimpleNamespace(synthesize_chunks16k=lambda *args: iter(()))
    provider.default_voice = "Aiden"
    with pytest.raises(RuntimeError, match="returned no audio"):
        provider.warmup()
