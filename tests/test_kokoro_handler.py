from pathlib import Path
from types import ModuleType
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TTS.kokoro_handler import KokoroTTSHandler


def test_setup_mlx_surfaces_missing_tts_dependency(monkeypatch):
    handler = KokoroTTSHandler.__new__(KokoroTTSHandler)
    handler.lang_code = "b"
    handler.voice = "bm_fable"

    def fake_load_model(_model_name):
        raise ImportError("No module named 'misaki'")

    mlx_audio_module = ModuleType("mlx_audio")
    tts_module = ModuleType("mlx_audio.tts")
    utils_module = ModuleType("mlx_audio.tts.utils")
    utils_module.load_model = fake_load_model
    tts_module.utils = utils_module
    mlx_audio_module.tts = tts_module

    monkeypatch.setitem(sys.modules, "mlx_audio", mlx_audio_module)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", tts_module)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.utils", utils_module)

    try:
        handler._setup_mlx("mlx-community/Kokoro-82M-bf16")
    except ImportError as exc:
        message = str(exc)
        assert "additional mlx-audio TTS dependencies" in message
        assert "misaki" in message
        assert "pip install misaki espeakng-loader" in message
    else:
        raise AssertionError("Expected _setup_mlx to raise ImportError")
