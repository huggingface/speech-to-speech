from __future__ import annotations

from copy import deepcopy
from queue import Queue
from threading import Event
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from openai.types.realtime import RealtimeErrorEvent, SessionUpdateEvent
from transformers.feature_extraction_utils import BatchFeature

import speech_to_speech.s2s_pipeline as s2s_pipeline
from speech_to_speech.backend_registry import (
    STT_BACKENDS,
    BackendSelection,
    HandlerContext,
    create_backend_handler,
)
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.s2s_pipeline import parse_arguments
from speech_to_speech.STT import granite_speech_ctc_handler
from speech_to_speech.STT.granite_speech_ctc_handler import GraniteSpeechCTCSTTHandler


class _FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, audio: Any, sampling_rate: int, return_tensors: str) -> BatchFeature:
        self.calls.append({"audio": audio, "sampling_rate": sampling_rate, "return_tensors": return_tensors})
        return BatchFeature(
            {
                "input_features": torch.zeros(1, 4, 80),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            }
        )

    def batch_decode(self, ids: Any, skip_special_tokens: bool) -> list[str]:
        assert skip_special_tokens
        return [" Hello, world. "]


class _FakeModel:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.moved_to: str | None = None

    def to(self, device: str) -> _FakeModel:
        self.moved_to = device
        return self

    def eval(self) -> _FakeModel:
        return self

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.generate_calls.append(kwargs)
        return torch.zeros((1, 3), dtype=torch.long)


class _Loader:
    def __init__(self, model: _FakeModel) -> None:
        self.model = model
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def from_pretrained(self, name: str, **kwargs: Any) -> _FakeModel:
        self.calls.append((name, kwargs))
        return self.model


def _install_granite(monkeypatch) -> tuple[_Loader, list[tuple[str, dict[str, Any]]], _FakeProcessor]:
    processor = _FakeProcessor()
    loader = _Loader(_FakeModel())
    processor_calls: list[tuple[str, dict[str, Any]]] = []

    def load_processor(name: str, **kwargs: Any) -> _FakeProcessor:
        processor_calls.append((name, kwargs))
        return processor

    monkeypatch.setattr(granite_speech_ctc_handler.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        granite_speech_ctc_handler,
        "transformers",
        SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=load_processor), AutoModelForCTC=loader),
    )
    return loader, processor_calls, processor


def _vad(mode: str) -> VADAudio:
    return VADAudio(
        audio=np.zeros(16000, dtype=np.float32),
        mode=mode,
        turn_id="turn",
        turn_revision=2,
        speech_end_at_s=5.0,
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


def test_granite_ctc_loads_pinned_revision_and_warms_up(monkeypatch):
    loader, processor_calls, processor = _install_granite(monkeypatch)

    handler = GraniteSpeechCTCSTTHandler.__new__(GraniteSpeechCTCSTTHandler)
    handler.setup(device="cpu", revision="abc123")

    name = "ibm-granite/granite-speech-5.0-470m-turboctc"
    assert processor_calls == [(name, {"revision": "abc123"})]
    assert loader.calls == [(name, {"revision": "abc123", "dtype": torch.float32})]
    assert loader.model.moved_to == "cpu"
    assert len(loader.model.generate_calls) == 1
    assert processor.calls[0]["sampling_rate"] == 16000
    assert processor.calls[0]["return_tensors"] == "pt"
    assert len(processor.calls[0]["audio"]) == 1
    assert processor.calls[0]["audio"][0].shape == (16000,)
    assert loader.model.generate_calls[0]["input_features"].dtype == torch.float32
    assert loader.model.generate_calls[0]["attention_mask"].dtype == torch.long


def test_granite_ctc_rejects_unsupported_language(monkeypatch):
    loader, processor_calls, _processor = _install_granite(monkeypatch)
    handler = GraniteSpeechCTCSTTHandler.__new__(GraniteSpeechCTCSTTHandler)

    with pytest.raises(ValueError, match="supports"):
        handler.setup(device="cpu", language="zh")

    assert loader.calls == []
    assert processor_calls == []


@pytest.mark.parametrize(("mode", "output_type"), [("progressive", PartialTranscription), ("final", Transcription)])
def test_granite_ctc_emits_pipeline_transcription_events(monkeypatch, mode, output_type):
    loader, _processor_calls, _processor = _install_granite(monkeypatch)
    handler = GraniteSpeechCTCSTTHandler.__new__(GraniteSpeechCTCSTTHandler)
    generation = {"output_hidden_states": True}
    handler.setup(device="cpu", language="fr", gen_kwargs=generation)
    recorded = []
    handler._record_final_stt = lambda item, seconds: recorded.append((item.turn_id, seconds))

    [output] = list(handler.process(_vad(mode)))

    assert type(output) is output_type
    assert output.text == "Hello, world."
    assert output.turn_id == "turn"
    assert output.turn_revision == 2
    assert handler.gen_kwargs == generation
    assert handler.gen_kwargs is not generation
    assert loader.model.generate_calls[-1]["output_hidden_states"] is True
    if mode == "final":
        assert output.language_code == "fr"
        assert output.speech_stopped_at_s == 5.0
        assert [turn for turn, _seconds in recorded] == ["turn"]
    else:
        assert recorded == []


def test_granite_ctc_cli_config_is_normalized_for_handler():
    args = parse_arguments(["--stt", "granite-ctc", "--granite_ctc_language", "de"])

    assert args.stt_backend.name == "granite-ctc"
    assert args.stt_backend.config == {
        "model_name": "ibm-granite/granite-speech-5.0-470m-turboctc",
        "revision": None,
        "device": "auto",
        "torch_dtype": "auto",
        "language": "de",
        "num_threads": None,
        "gen_kwargs": {},
    }


def test_granite_ctc_constructs_through_registry(monkeypatch):
    _install_granite(monkeypatch)
    args = parse_arguments(["--stt", "granite-ctc", "--granite_ctc_device", "cpu"])
    context = _context()

    handler = create_backend_handler(args.stt_backend, context)

    assert type(handler) is GraniteSpeechCTCSTTHandler
    assert handler.speculative_turns is context.speculative_turns


def test_granite_ctc_session_language_selection_remains_unsupported():
    selection = BackendSelection(STT_BACKENDS["granite-ctc"], {})

    assert s2s_pipeline._stt_session_languages(selection, SimpleNamespace()) == set()


def test_granite_ctc_rejects_session_language_update(monkeypatch):
    args = parse_arguments(["--stt", "granite-ctc", "--granite_ctc_language", "de", "--tts", "openai"])
    setup_config = deepcopy(args.stt_backend.config)
    monkeypatch.setattr(s2s_pipeline, "_build_handlers", lambda **_kwargs: [SimpleNamespace() for _ in range(5)])
    unit = s2s_pipeline._build_pipeline_unit(
        index=0,
        stop_event=Event(),
        module_kwargs=args.module_kwargs,
        vad_handler_kwargs=args.vad_handler_kwargs,
        stt_backend=args.stt_backend,
        llm_backend=args.llm_backend,
        tts_backend=args.tts_backend,
    )
    unit.service.tts_supported_languages = {"en", "fr", "de"}
    conn_id = unit.service.register()
    update = SessionUpdateEvent.model_validate(
        {
            "type": "session.update",
            "session": {"type": "realtime", "audio": {"input": {"transcription": {"language": "fr"}}}},
        }
    )

    error = unit.service.handle_session_update(conn_id, update)

    assert isinstance(error, RealtimeErrorEvent)
    assert "STT" in error.error.message
    assert unit.service._state(conn_id).runtime_config.selected_language is None
    assert args.stt_backend.config == setup_config
