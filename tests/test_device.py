from __future__ import annotations

import sys
from threading import Event
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from speech_to_speech.utils import utils
from speech_to_speech.utils.utils import TORCH_DEVICES, resolve_device, validate_device


def _available(monkeypatch: pytest.MonkeyPatch, *devices: str) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: "cuda" in devices)
    monkeypatch.setattr(utils, "is_npu_available", lambda: "npu" in devices)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: "xpu" in devices)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: "mps" in devices)


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        (("cuda", "npu", "xpu", "mps"), "cuda"),
        (("npu", "xpu", "mps"), "npu"),
        (("xpu", "mps"), "xpu"),
        (("mps",), "mps"),
        ((), "cpu"),
    ],
)
def test_auto_picks_first_available_supported_device(
    monkeypatch: pytest.MonkeyPatch, available: tuple[str, ...], expected: str
) -> None:
    _available(monkeypatch, *available)
    assert resolve_device("auto", TORCH_DEVICES, "Test") == expected


def test_auto_skips_available_devices_the_handler_does_not_support(monkeypatch: pytest.MonkeyPatch) -> None:
    _available(monkeypatch, "xpu", "mps")
    assert resolve_device("auto", ("cuda", "npu", "cpu"), "Test") == "cpu"


def test_explicit_supported_device_is_kept_with_its_index(monkeypatch: pytest.MonkeyPatch) -> None:
    _available(monkeypatch)
    assert resolve_device("cuda:1", TORCH_DEVICES, "Test") == "cuda:1"


def test_explicit_unsupported_device_names_the_supported_devices() -> None:
    with pytest.raises(ValueError, match=r"NeMo ASR supports device 'auto' or one of: cuda, npu, cpu; got 'mps'"):
        resolve_device("mps", ("cuda", "npu", "cpu"), "NeMo ASR")


def test_auto_without_an_available_supported_device_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _available(monkeypatch)
    with pytest.raises(ValueError, match="none of its supported devices"):
        resolve_device("auto", ("cuda",), "Qwen3-TTS torch backend")


def test_validate_device_leaves_auto_to_the_backend() -> None:
    validate_device("auto", ("cpu", "cuda"), "Test")
    with pytest.raises(ValueError, match="got 'mps'"):
        validate_device("mps", ("cpu", "cuda"), "Test")


def test_paraformer_resolves_auto_before_calling_funasr(monkeypatch: pytest.MonkeyPatch) -> None:
    from speech_to_speech.STT.paraformer_handler import ParaformerSTTHandler

    _available(monkeypatch, "npu")
    fake_model = MagicMock()
    fake_model.generate.return_value = [{"text": ""}]
    fake_auto_model = MagicMock(return_value=fake_model)
    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=fake_auto_model))

    handler = object.__new__(ParaformerSTTHandler)
    handler.setup(model_name="paraformer-zh", device="auto")

    fake_auto_model.assert_called_once_with(model="paraformer-zh", device="npu")


def test_faster_whisper_rejects_a_device_ctranslate2_cannot_use(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_whisper_model = MagicMock()
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=fake_whisper_model))
    sys.modules.pop("speech_to_speech.STT.faster_whisper_handler", None)
    from speech_to_speech.STT.faster_whisper_handler import FasterWhisperSTTHandler

    handler = object.__new__(FasterWhisperSTTHandler)
    with pytest.raises(ValueError, match="Faster Whisper STT supports device 'auto' or one of: cpu, cuda"):
        handler.setup(device="mps")
    fake_whisper_model.assert_not_called()


class _FakeChat:
    loads: list[dict[str, Any]] = []

    class InferCodeParams:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    def load(self, **kwargs: Any) -> bool:
        self.loads.append(kwargs)
        self.device = kwargs["device"] if kwargs["device"] is not None else torch.device("cpu")
        return True

    def sample_random_speaker(self) -> str:
        return "speaker"

    def infer(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return []


@pytest.mark.parametrize(
    ("device", "passed", "used"),
    [("auto", None, "cpu"), ("cpu", torch.device("cpu"), "cpu"), ("mps", torch.device("mps"), "mps")],
)
def test_chattts_passes_its_device_to_the_model(
    monkeypatch: pytest.MonkeyPatch, device: str, passed: torch.device | None, used: str
) -> None:
    monkeypatch.setitem(sys.modules, "ChatTTS", SimpleNamespace(Chat=_FakeChat))
    sys.modules.pop("speech_to_speech.TTS.chatTTS_handler", None)
    from speech_to_speech.TTS.chatTTS_handler import ChatTTSHandler

    _FakeChat.loads = []
    handler = object.__new__(ChatTTSHandler)
    handler.setup(Event(), device=device)

    assert _FakeChat.loads == [{"compile": False, "device": passed}]
    assert handler.device == used


def test_chattts_rejects_an_unsupported_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ChatTTS", SimpleNamespace(Chat=_FakeChat))
    sys.modules.pop("speech_to_speech.TTS.chatTTS_handler", None)
    from speech_to_speech.TTS.chatTTS_handler import ChatTTSHandler

    _FakeChat.loads = []
    handler = object.__new__(ChatTTSHandler)
    with pytest.raises(ValueError, match="ChatTTS supports device 'auto' or one of: cuda, npu, mps, cpu; got 'xpu'"):
        handler.setup(Event(), device="xpu")
    assert _FakeChat.loads == []
