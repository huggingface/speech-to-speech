from __future__ import annotations

import subprocess
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


def test_importing_device_helpers_does_not_import_torch() -> None:
    # The CLI, Realtime API and LLM helpers import this module without touching a
    # device, so it must not pay the multi-second torch import at module load.
    code = (
        "import sys\n"
        "import speech_to_speech.utils.utils\n"
        "import speech_to_speech.cli\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# --- accelerator cache clearing -----------------------------------------------------------
#
# `resolve_device` keeps an explicit device as requested without checking that it exists
# (see test_explicit_supported_device_is_kept_with_its_index), so `self.device == "mps"`
# records the request. Calling into `torch.mps` on that basis raises
# `RuntimeError: Cannot execute emptyCache() without MPS backend.`


def _recording_caches(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    cleared: list[str] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cleared.append("cuda"))
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: cleared.append("mps"))
    return cleared


def test_empty_mps_cache_skips_mps_when_it_is_not_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _available(monkeypatch)
    cleared = _recording_caches(monkeypatch)

    utils.empty_mps_cache("mps")

    assert cleared == []


def test_empty_mps_cache_clears_mps_when_it_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _available(monkeypatch, "mps")
    cleared = _recording_caches(monkeypatch)

    utils.empty_mps_cache("mps")

    assert cleared == ["mps"]


@pytest.mark.parametrize("device", ["cuda", "cuda:1", "cpu", "xpu", "npu"])
def test_empty_mps_cache_never_touches_other_devices(monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    """Clearing the CUDA cache after every generation would discard reusable blocks."""
    _available(monkeypatch, "cuda", "xpu", "npu", "mps")
    cleared = _recording_caches(monkeypatch)

    utils.empty_mps_cache(device)

    assert cleared == []


def test_device_is_available_reports_the_fact_not_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    _available(monkeypatch, "cuda")

    assert resolve_device("mps", TORCH_DEVICES, "Test") == "mps"
    assert utils.device_is_available("mps") is False
    assert utils.device_is_available("cuda:1") is True


def test_no_handler_guards_an_mps_call_on_the_requested_device_alone() -> None:
    """`self.device == "mps"` is the request; a torch.mps call needs availability too."""
    import ast
    from pathlib import Path

    def guards_on_requested_device(test: ast.expr) -> bool:
        # Matches `self.device == "mps"` but not `... and device_is_available(...)`.
        return (
            isinstance(test, ast.Compare)
            and isinstance(test.ops[0], ast.Eq)
            and isinstance(test.left, ast.Attribute)
            and test.left.attr == "device"
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "mps"
        )

    def calls_torch_mps(node: ast.AST) -> bool:
        return any(
            isinstance(inner, ast.Attribute) and isinstance(inner.value, ast.Attribute) and inner.value.attr == "mps"
            for inner in ast.walk(node)
        )

    offenders = []
    for path in sorted(Path("src/speech_to_speech").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and guards_on_requested_device(node.test):
                if any(calls_torch_mps(stmt) for stmt in node.body):
                    offenders.append(f"{path}:{node.lineno}")

    assert offenders == [], (
        "these blocks call into torch.mps guarded only by the requested device; "
        f"use empty_device_cache()/device_is_available(): {offenders}"
    )


def test_paraformer_does_not_clear_an_mps_cache_it_was_only_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    """The #145 symptom: `--paraformer_stt_device mps` without an MPS build."""
    import numpy as np

    from speech_to_speech.pipeline.messages import VADAudio
    from speech_to_speech.STT.paraformer_handler import ParaformerSTTHandler

    _available(monkeypatch, "cpu")
    monkeypatch.setattr(
        torch.mps,
        "empty_cache",
        lambda: (_ for _ in ()).throw(RuntimeError("Cannot execute emptyCache() without MPS backend.")),
    )
    fake_model = MagicMock()
    fake_model.generate.return_value = [{"text": "你好"}]
    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=MagicMock(return_value=fake_model)))

    handler = object.__new__(ParaformerSTTHandler)
    handler.setup(model_name="paraformer-zh", device="mps")
    assert handler.device == "mps"

    audio = VADAudio(audio=np.zeros(16000, dtype=np.float32), turn_id="turn_1", turn_revision=0)
    assert [out.text for out in handler.process(audio)] == ["你好"]


def test_cuda_paraformer_does_not_clear_the_cuda_cache_per_transcription(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cache clear was MPS-only before; CUDA must keep its allocator cache."""
    import numpy as np

    from speech_to_speech.pipeline.messages import VADAudio
    from speech_to_speech.STT.paraformer_handler import ParaformerSTTHandler

    _available(monkeypatch, "cuda")
    cleared = _recording_caches(monkeypatch)
    fake_model = MagicMock()
    fake_model.generate.return_value = [{"text": "你好"}]
    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=MagicMock(return_value=fake_model)))

    handler = object.__new__(ParaformerSTTHandler)
    handler.setup(model_name="paraformer-zh", device="cuda")
    for revision in range(3):
        audio = VADAudio(
            audio=np.zeros(16000, dtype=np.float32), turn_id="turn_1", turn_revision=revision, mode="progressive"
        )
        list(handler.process(audio))

    assert cleared == []
