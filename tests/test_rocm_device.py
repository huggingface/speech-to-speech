from __future__ import annotations

import sys
from threading import Event
from types import SimpleNamespace

import pytest
import torch

from speech_to_speech.utils import utils
from speech_to_speech.utils.utils import (
    TORCH_DEVICES,
    is_rocm,
    normalize_device,
    resolve_attn_implementation,
    resolve_device,
    validate_device,
)


@pytest.mark.parametrize(
    ("given", "expected"),
    [("rocm", "cuda"), ("ROCm:1", "cuda:1"), ("hip", "cuda"), ("cuda:0", "cuda:0"), ("cpu", "cpu"), ("auto", "auto")],
)
def test_rocm_and_hip_are_spellings_of_cuda(given: str, expected: str) -> None:
    assert normalize_device(given) == expected


def test_explicit_rocm_resolves_to_the_cuda_namespace() -> None:
    assert resolve_device("rocm:1", TORCH_DEVICES, "Test") == "cuda:1"
    validate_device("rocm", ("cpu", "cuda"), "Faster Whisper STT")


def test_rocm_is_still_rejected_where_cuda_is_unsupported() -> None:
    with pytest.raises(ValueError, match="got 'cuda'"):
        validate_device("rocm", ("mps", "cpu"), "Test")


def test_is_rocm_reads_torch_version_hip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.version, "hip", "7.0.0", raising=False)
    assert is_rocm()
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert not is_rocm()


def test_flash_attention_falls_back_to_sdpa_without_flash_attn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils.importlib.util, "find_spec", lambda name: None)
    assert resolve_attn_implementation("flash_attention_2", "cuda") == "sdpa"
    assert resolve_attn_implementation("flash_attention_2", "cpu") == "sdpa"
    assert resolve_attn_implementation("eager", "cuda") == "eager"


def test_flash_attention_kept_when_installed_on_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils.importlib.util, "find_spec", lambda name: object())
    assert resolve_attn_implementation("flash_attention_2", "cuda:0") == "flash_attention_2"


@pytest.fixture
def faster_whisper_handler(monkeypatch: pytest.MonkeyPatch):
    # Stub the optional dependency so these tests run without faster-whisper installed.
    if "faster_whisper" not in sys.modules:
        monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=None))
    from speech_to_speech.STT import faster_whisper_handler

    return faster_whisper_handler


def test_faster_whisper_uses_cpu_when_ctranslate2_has_no_cuda(
    monkeypatch: pytest.MonkeyPatch, faster_whisper_handler
) -> None:
    FasterWhisperSTTHandler = faster_whisper_handler.FasterWhisperSTTHandler
    monkeypatch.setattr(faster_whisper_handler, "is_rocm", lambda: True)
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: 0))
    assert FasterWhisperSTTHandler._ctranslate2_device("cuda") == "cpu"
    assert FasterWhisperSTTHandler._ctranslate2_device("auto") == "auto"
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: 1))
    assert FasterWhisperSTTHandler._ctranslate2_device("cuda:0") == "cuda:0"


def test_faster_whisper_keeps_explicit_cuda_off_rocm(monkeypatch: pytest.MonkeyPatch, faster_whisper_handler) -> None:
    monkeypatch.setattr(faster_whisper_handler, "is_rocm", lambda: False)
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: 0))
    assert faster_whisper_handler.FasterWhisperSTTHandler._ctranslate2_device("cuda") == "cuda"


@pytest.mark.parametrize(
    ("device", "requested", "expected_device", "expected_compute_type"),
    [
        ("rocm", "float16", "cpu", "float32"),
        ("rocm", "bfloat16", "cpu", "float32"),
        ("rocm", "int8_float16", "cpu", "int8"),
        ("rocm", "auto", "cpu", "auto"),
        ("rocm", "int8", "cpu", "int8"),
        ("cpu", "float16", "cpu", "float16"),
    ],
)
def test_faster_whisper_cpu_fallback_picks_a_cpu_compute_type(
    monkeypatch: pytest.MonkeyPatch,
    faster_whisper_handler,
    device: str,
    requested: str,
    expected_device: str,
    expected_compute_type: str,
) -> None:
    loaded = {}
    monkeypatch.setattr(faster_whisper_handler, "is_rocm", lambda: True)
    monkeypatch.setattr(faster_whisper_handler, "WhisperModel", lambda name, **kwargs: loaded.update(kwargs))
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: 0))

    handler = object.__new__(faster_whisper_handler.FasterWhisperSTTHandler)
    handler.setup(device=device, compute_type=requested)

    assert loaded == {"device": expected_device, "compute_type": expected_compute_type}


def test_chattts_hands_torch_a_normalized_rocm_device(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = {}

    class _FakeChat:
        device = "cuda"

        def load(self, compile: bool, device: torch.device | None) -> bool:
            loaded["device"] = device
            return True

        def sample_random_speaker(self) -> None:
            return None

        class InferCodeParams:
            def __init__(self, **_kwargs: object) -> None:
                pass

    fake_chattts = SimpleNamespace(Chat=_FakeChat)
    monkeypatch.setitem(sys.modules, "ChatTTS", fake_chattts)
    from speech_to_speech.TTS import chatTTS_handler

    # Another test may have imported the handler module against a different stub.
    monkeypatch.setattr(chatTTS_handler, "ChatTTS", fake_chattts)
    monkeypatch.setattr(chatTTS_handler.ChatTTSHandler, "warmup", lambda self: None)
    handler = object.__new__(chatTTS_handler.ChatTTSHandler)
    handler.setup(Event(), device="rocm:1")

    assert loaded["device"] == torch.device("cuda:1")
