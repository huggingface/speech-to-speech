from queue import Queue
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

import speech_to_speech.TTS.qwen3_tts_handler as module
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.s2s_pipeline import parse_arguments


@pytest.mark.parametrize("preset", [False, True])
def test_mac_defaults_load_ggml_and_stream_selected_speaker(monkeypatch, preset):
    import sys

    def unexpected_mlx(*args, **kwargs):
        raise AssertionError("macOS must load GGML, not MLX")

    monkeypatch.setattr(module.Qwen3TTSHandler, "_setup_mlx", unexpected_mlx, raising=False)
    loads, calls = [], []
    model = SimpleNamespace(
        warmup=lambda **kwargs: None,
        get_supported_speakers=lambda: ["aiden", "ryan"],
        generate_custom_voice_streaming=lambda **kwargs: (calls.append(kwargs), iter([(np.full(768, 0.1), 24000, {})]))[
            1
        ],
    )
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(
        sys.modules,
        "faster_qwen3_tts",
        SimpleNamespace(
            FasterQwen3TTS=SimpleNamespace(
                from_pretrained=lambda name, **kwargs: (loads.append((name, kwargs)), model)[1]
            )
        ),
    )
    args = parse_arguments(["--mac-optimal-settings"] if preset else [])
    handler = object.__new__(module.Qwen3TTSHandler)
    handler.queue_in = Queue()
    handler.setup(Event(), **args.tts_backend.config)
    assert loads[0][0] == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert loads[0][1]["backend"] == "ggml"
    assert loads[0][1]["quant"] == "Q8_0"
    assert loads[0][1]["gguf_talker_path"] is None
    assert loads[0][1]["gguf_codec_path"] is None
    assert calls[0]["speaker"] == "Aiden"
    handler.speaker = "Ryan"
    chunks = list(handler.process(TTSInput(text="Hello from the Mac.")))
    assert calls[-1]["speaker"] == "Ryan"
    assert chunks and all(chunk.dtype == np.int16 and chunk.size == 512 for chunk in chunks)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"backend": "mlx"}, "no longer supported"),
        ({"mlx_quantization": "6bit"}, "qwen3_tts_ggml_quantization"),
        ({"model_name": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit"}, "Qwen/"),
    ],
)
def test_old_mlx_configuration_has_migration_error(monkeypatch, kwargs, message):
    monkeypatch.setattr(module.Qwen3TTSHandler, "_setup_mlx", lambda *args: None, raising=False)
    monkeypatch.setattr(module.Qwen3TTSHandler, "warmup", lambda *args: None)
    handler = object.__new__(module.Qwen3TTSHandler)
    with pytest.raises(ValueError, match=message):
        flags = [item for key, value in kwargs.items() for item in (f"--qwen3_tts_{key}", value)]
        args = parse_arguments(flags)
        handler.setup(Event(), **args.tts_backend.config)


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
@pytest.mark.parametrize("backend", ["ggml", "torch"])
def test_explicit_backend_is_honored(monkeypatch, platform, backend):
    import sys

    loaded = []
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(module.Qwen3TTSHandler, "_setup_faster", lambda self, **kw: loaded.append(kw))
    monkeypatch.setattr(module.Qwen3TTSHandler, "warmup", lambda self: None)
    args = parse_arguments(["--qwen3_tts_backend", backend, "--qwen3_tts_streaming_chunk_size", "4"])
    handler = object.__new__(module.Qwen3TTSHandler)
    handler.setup(Event(), **args.tts_backend.config)
    assert loaded[0]["backend"] == backend
    assert handler.streaming_chunk_size == 4


def test_mac_clone_uses_shared_reference_audio_path(monkeypatch, tmp_path):
    import sys

    captured = []
    reference = tmp_path / "reference.wav"
    reference.touch()
    model = SimpleNamespace(
        warmup=lambda **kwargs: None,
        generate_voice_clone_streaming=lambda **kwargs: (
            captured.append(kwargs),
            iter([(np.full(768, 0.1), 24000, {})]),
        )[1],
    )
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(module.Qwen3TTSHandler, "_setup_faster", lambda self, **kw: setattr(self, "model", model))
    handler = object.__new__(module.Qwen3TTSHandler)
    handler.queue_in = Queue()
    handler.setup(Event(), model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base", ref_audio=str(reference), ref_text="Reference.")
    assert list(handler.process(TTSInput(text="Clone this sentence.")))
    assert captured[-1]["ref_audio"] == str(reference)
    assert captured[-1]["ref_text"] == "Reference."
    assert captured[-1]["chunk_size"] == 8
    assert captured[-1]["non_streaming_mode"] is True


def test_clone_preserves_packaged_relative_reference_path(monkeypatch, tmp_path):
    captured = []
    handler = object.__new__(module.Qwen3TTSHandler)
    model = SimpleNamespace(
        warmup=lambda **kw: None, generate_voice_clone_streaming=lambda **kw: (captured.append(kw), iter([]))[1]
    )
    monkeypatch.setattr(module.Qwen3TTSHandler, "_setup_faster", lambda self, **kw: setattr(self, "model", model))
    monkeypatch.chdir(tmp_path)
    handler.setup(Event(), model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base", ref_audio="TTS/ref_audio.wav")
    from pathlib import Path

    assert captured and Path(captured[0]["ref_audio"]).is_absolute()
    assert Path(captured[0]["ref_audio"]).is_file()


@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_torch_device_validation_survives_macos_migration(device):
    handler = object.__new__(module.Qwen3TTSHandler)
    with pytest.raises(ValueError, match="Qwen3-TTS torch backend supports device"):
        handler.setup(Event(), backend="torch", device=device)


def test_torch_auto_uses_main_device_resolver(monkeypatch):
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.Qwen3TTSHandler, "_setup_faster", lambda self, **kwargs: None)
    monkeypatch.setattr(module.Qwen3TTSHandler, "warmup", lambda self: None)
    handler = object.__new__(module.Qwen3TTSHandler)
    handler.setup(Event(), backend="torch", device="auto")
    assert handler.device == "cuda"


@pytest.mark.parametrize(
    "platform, backend, quantization, expected",
    [
        ("darwin", "ggml", None, "Q8_0"),
        ("linux", "ggml", None, "BF16"),
        ("win32", "ggml", None, "BF16"),
        ("darwin", "torch", None, "BF16"),
        ("darwin", "ggml", "BF16", "BF16"),
        ("darwin", "ggml", "Q4_K_M", "Q4_K_M"),
    ],
)
def test_quantization_defaults_preserve_platforms_and_overrides(monkeypatch, platform, backend, quantization, expected):
    import sys

    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(module.Qwen3TTSHandler, "_setup_faster", lambda self, **kw: None)
    monkeypatch.setattr(module.Qwen3TTSHandler, "warmup", lambda self: None)
    flags = ["--qwen3_tts_backend", backend]
    if quantization is not None:
        flags.extend(["--qwen3_tts_ggml_quantization", quantization])
    args = parse_arguments(flags)
    handler = object.__new__(module.Qwen3TTSHandler)
    handler.setup(Event(), **args.tts_backend.config)
    assert handler.ggml_quantization == expected
