from __future__ import annotations

import sys
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


def test_faster_whisper_uses_cpu_when_ctranslate2_has_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    from speech_to_speech.STT.faster_whisper_handler import FasterWhisperSTTHandler

    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: 0))
    assert FasterWhisperSTTHandler._ctranslate2_device("cuda") == "cpu"
    assert FasterWhisperSTTHandler._ctranslate2_device("auto") == "auto"
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: 1))
    assert FasterWhisperSTTHandler._ctranslate2_device("cuda:0") == "cuda:0"
