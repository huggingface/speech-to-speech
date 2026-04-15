from pathlib import Path
from threading import Event
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import TTS.qwen3_tts_handler as qwen3_tts_module
from TTS.qwen3_tts_handler import Qwen3TTSHandler


def test_setup_uses_mlx_backend_on_darwin_and_maps_qwen_repo_ids(monkeypatch):
    recorded = {}

    def _setup_mlx(self, model_name):
        recorded["model_name"] = model_name

    def _setup_faster(self, *args, **kwargs):
        raise AssertionError("Darwin setup should not use the faster-qwen3-tts backend")

    monkeypatch.setattr(qwen3_tts_module, "platform", "darwin")
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_mlx", _setup_mlx)
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_faster", _setup_faster)
    monkeypatch.setattr(Qwen3TTSHandler, "warmup", lambda self: None)

    handler = object.__new__(Qwen3TTSHandler)
    handler.setup(
        Event(),
        model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device="cuda",
    )

    assert handler.backend == "mlx"
    assert handler.device == "mps"
    assert (
        recorded["model_name"]
        == "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
    )


def test_setup_preserves_faster_backend_off_darwin(monkeypatch):
    recorded = {}

    def _setup_mlx(self, *args, **kwargs):
        raise AssertionError("Non-Darwin setup should not use the mlx backend")

    def _setup_faster(self, model_name, dtype, attn_implementation):
        recorded["model_name"] = model_name
        recorded["dtype"] = dtype
        recorded["attn_implementation"] = attn_implementation

    monkeypatch.setattr(qwen3_tts_module, "platform", "linux")
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_mlx", _setup_mlx)
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_faster", _setup_faster)
    monkeypatch.setattr(Qwen3TTSHandler, "warmup", lambda self: None)

    handler = object.__new__(Qwen3TTSHandler)
    handler.setup(
        Event(),
        model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device="cuda",
        dtype="float16",
        attn_implementation="sdpa",
    )

    assert handler.backend == "faster_qwen3_tts"
    assert recorded == {
        "model_name": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "dtype": "float16",
        "attn_implementation": "sdpa",
    }


def test_mlx_helper_methods_use_model_config_and_streaming_conversion():
    handler = object.__new__(Qwen3TTSHandler)
    handler.backend = "mlx"
    handler.model_name = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
    handler.speaker = None
    handler.streaming_chunk_size = 8
    handler.model = SimpleNamespace(
        config=SimpleNamespace(tts_model_type="custom_voice"),
        get_supported_speakers=lambda: ["Vivian", "Ryan"],
    )

    assert handler._model_type() == "custom_voice"
    assert handler._resolve_speaker() == "Vivian"
    assert handler._mlx_streaming_interval() == pytest.approx(0.64)
