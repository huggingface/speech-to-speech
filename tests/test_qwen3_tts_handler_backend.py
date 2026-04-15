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


def test_prepare_mlx_ref_audio_normalizes_file_and_caches_result(monkeypatch, tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"fake")

    save_calls = []

    class FakeTensor:
        ndim = 2
        shape = (2, 3)

        def unsqueeze(self, _dim):
            return self

        def mean(self, dim=0, keepdim=True):
            assert dim == 0
            assert keepdim is True
            return self

        def to(self, dtype=None):
            return self

        def cpu(self):
            return self

    fake_torchaudio = SimpleNamespace(
        load=lambda path: (FakeTensor(), 44100),
        functional=SimpleNamespace(resample=lambda waveform, src, dst: waveform),
        save=lambda path, waveform, sample_rate, format=None: (
            save_calls.append((path, sample_rate, format)),
            Path(path).write_bytes(b"RIFF"),
        ),
    )
    fake_torch = SimpleNamespace(float32="float32")

    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    handler = object.__new__(Qwen3TTSHandler)
    handler.backend = "mlx"
    handler.model = SimpleNamespace(sample_rate=24000)
    handler._mlx_ref_audio_cache = {}
    handler._mlx_temp_ref_audio_files = set()

    normalized = handler._prepare_mlx_ref_audio(str(source))
    normalized_again = handler._prepare_mlx_ref_audio(str(source))

    assert normalized == normalized_again
    assert Path(normalized).exists()
    assert save_calls == [(normalized, 24000, "wav")]


def test_apply_session_voice_override_ignores_non_file_for_base_model():
    handler = object.__new__(Qwen3TTSHandler)
    handler.runtime_config = SimpleNamespace(
        session=SimpleNamespace(
            audio=SimpleNamespace(
                output=SimpleNamespace(voice="alloy")
            )
        )
    )
    handler.ref_audio = "TTS/ref_audio.wav"
    handler.speaker = None

    handler._apply_session_voice_override("base")

    assert handler.ref_audio == "TTS/ref_audio.wav"
    assert handler.speaker is None
