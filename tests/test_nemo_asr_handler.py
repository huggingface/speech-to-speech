from __future__ import annotations

import sys
import types
from queue import Queue
from threading import Event
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from speech_to_speech.backend_registry import HandlerContext, create_backend_handler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.s2s_pipeline import parse_arguments
from speech_to_speech.STT import nemo_asr_handler
from speech_to_speech.STT.nemo_asr_handler import NemoASRSTTHandler


class _FakeTranscribeModel:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.transcribe_calls: list[Any] = []

    def transcribe(self, audio: Any) -> list[Any]:
        self.transcribe_calls.append(audio)
        return [self.result]


def _handler(*, language: str = "en", result: Any = "hello") -> NemoASRSTTHandler:
    handler = object.__new__(NemoASRSTTHandler)
    handler.model = _FakeTranscribeModel(result)
    handler.language = language
    handler.device = "cpu"
    return handler


def _vad_audio(mode: str = "final") -> VADAudio:
    return VADAudio(
        audio=np.zeros(16000, dtype=np.float32),
        mode=mode,
        turn_id="turn_1",
        turn_revision=2,
        created_at_s=123.0,
    )


def _context() -> HandlerContext:
    return HandlerContext(
        stop_event=Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        text_output_queue=Queue(),
        should_listen=Event(),
        cancel_scope=CancelScope(),
        speculative_turns=SpeculativeTurnTracker(),
        pipeline_index=0,
        sample_rate=16000,
        enable_live_transcription=False,
        live_transcription_update_interval=0.5,
    )


def _install_fake_nemo(monkeypatch: pytest.MonkeyPatch, asr_model_cls: type[Any]) -> None:
    for name in ("nemo", "nemo.collections", "nemo.collections.asr"):
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, module)
    models = types.ModuleType("nemo.collections.asr.models")
    models.ASRModel = asr_model_cls
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", models)
    sys.modules["nemo"].collections = sys.modules["nemo.collections"]
    sys.modules["nemo.collections"].asr = sys.modules["nemo.collections.asr"]
    sys.modules["nemo.collections.asr"].models = models


@pytest.fixture(autouse=True)
def _quiet_console(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nemo_asr_handler.console, "print", lambda *args, **kwargs: None)


@pytest.mark.parametrize("result", ["hello", SimpleNamespace(text="hello")])
def test_final_transcription_copies_text_language_and_turn_fields(result: Any) -> None:
    handler = _handler(language="fr", result=result)

    events = list(handler.process(_vad_audio("final")))

    assert len(events) == 1
    assert isinstance(events[0], Transcription)
    assert events[0].text == "hello"
    assert events[0].language_code == "fr"
    assert events[0].turn_id == "turn_1"
    assert events[0].turn_revision == 2
    assert events[0].speech_stopped_at_s == 123.0
    called = handler.model.transcribe_calls[0]
    assert len(called) == 1
    assert isinstance(called[0], np.ndarray)
    assert called[0].dtype == np.float32


@pytest.mark.parametrize("result", ["hello", SimpleNamespace(text="hello")])
def test_progressive_transcription_is_partial(result: Any) -> None:
    events = list(_handler(result=result).process(_vad_audio("progressive")))

    assert len(events) == 1
    assert isinstance(events[0], PartialTranscription)
    assert events[0].text == "hello"
    assert events[0].turn_id == "turn_1"
    assert events[0].turn_revision == 2
    assert not hasattr(events[0], "language_code")


def test_setup_calls_from_pretrained_with_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeASRModel:
        loaded: list[Any] = []

        def __init__(self) -> None:
            self.transcribe_calls: list[Any] = []

        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> FakeASRModel:
            model_name = kwargs.get("model_name", args[0] if args else None)
            cls.loaded.append(model_name)
            return cls()

        def to(self, device: str) -> FakeASRModel:
            return self

        def transcribe(self, audio: Any) -> list[str]:
            self.transcribe_calls.append(audio)
            return ["warmup"]

    _install_fake_nemo(monkeypatch, FakeASRModel)
    handler = object.__new__(NemoASRSTTHandler)
    handler.setup(model_name="nvidia/parakeet-unified-en-0.6b", device="cpu", language="en")

    assert "nvidia/parakeet-unified-en-0.6b" in FakeASRModel.loaded
    assert handler.language == "en"
    assert handler.device == "cpu"
    assert handler.model_name == "nvidia/parakeet-unified-en-0.6b"


def test_missing_nemo_names_the_nemo_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "nemo",
        "nemo.collections",
        "nemo.collections.asr",
        "nemo.collections.asr.models",
    ):
        monkeypatch.setitem(sys.modules, name, None)

    args = parse_arguments(["--stt", "parakeet-unified"])
    with pytest.raises(ImportError, match=r"speech-to-speech\[nemo\]"):
        create_backend_handler(args.stt_backend, _context())


def test_cli_parakeet_unified_defaults() -> None:
    args = parse_arguments(["--stt", "parakeet-unified"])

    assert args.stt_backend.name == "parakeet-unified"
    assert args.stt_backend.config["model_name"] == "nvidia/parakeet-unified-en-0.6b"
    assert args.stt_backend.config["device"] == "auto"
    assert args.stt_backend.config["language"] == "en"
    assert args.stt_backend.spec.required_extra == "nemo"


def test_cli_nemotron_streaming_model_name_override() -> None:
    args = parse_arguments(
        [
            "--stt",
            "nemotron-streaming",
            "--nemotron_streaming_model_name",
            "nvidia/nemotron-3.5-asr-streaming-0.6b",
        ]
    )

    assert args.stt_backend.name == "nemotron-streaming"
    assert args.stt_backend.config["model_name"] == "nvidia/nemotron-3.5-asr-streaming-0.6b"
    assert args.stt_backend.spec.required_extra == "nemo"
