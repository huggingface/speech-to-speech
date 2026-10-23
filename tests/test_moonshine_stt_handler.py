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

import speech_to_speech.s2s_pipeline as s2s_pipeline
from speech_to_speech.backend_registry import HandlerContext, create_backend_handler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.s2s_pipeline import parse_arguments
from speech_to_speech.STT import moonshine_handler
from speech_to_speech.STT.moonshine_handler import MoonshineSTTHandler, max_length_for, model_class_for


class _FakeInputs(dict):
    def to(self, *args: Any, **kwargs: Any) -> "_FakeInputs":
        self["moved_to"] = (args, kwargs)
        return self


class _FakeProcessor:
    def __init__(self, text: str = "hello world") -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def __call__(self, audio: Any, sampling_rate: int, return_tensors: str) -> _FakeInputs:
        self.calls.append({"audio": audio, "sampling_rate": sampling_rate})
        return _FakeInputs(input_values=torch.zeros(1, 4))

    def decode(self, ids: Any, skip_special_tokens: bool) -> str:
        return f" {self.text} "


class _FakeModel:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.moved_to: str | None = None

    def to(self, device: str) -> "_FakeModel":
        self.moved_to = device
        return self

    def eval(self) -> "_FakeModel":
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


def _vad(mode: str, seconds: float = 1.0) -> VADAudio:
    return VADAudio(
        audio=np.zeros(int(16000 * seconds), dtype=np.float32),
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


def _install_moonshine(monkeypatch, model_type: str = "moonshine") -> tuple[_Loader, _Loader, _FakeProcessor]:
    processor = _FakeProcessor()
    model = _FakeModel()
    regular, streaming = _Loader(model), _Loader(model)
    monkeypatch.setattr(moonshine_handler.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        moonshine_handler,
        "transformers",
        SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=lambda name: processor),
            AutoConfig=SimpleNamespace(from_pretrained=lambda name: SimpleNamespace(model_type=model_type)),
            MoonshineForConditionalGeneration=regular,
            MoonshineStreamingForConditionalGeneration=streaming,
        ),
    )
    return regular, streaming, processor


@pytest.mark.parametrize(("model_type", "streaming"), [("moonshine", False), ("moonshine_streaming", True)])
def test_moonshine_loads_the_class_matching_the_checkpoint(monkeypatch, model_type, streaming):
    regular, streaming_loader, _processor = _install_moonshine(monkeypatch, model_type)

    handler = MoonshineSTTHandler.__new__(MoonshineSTTHandler)
    handler.setup(model_name="moonshine-ai/checkpoint", device="cpu")

    used, unused = (streaming_loader, regular) if streaming else (regular, streaming_loader)
    assert used.calls == [("moonshine-ai/checkpoint", {"dtype": torch.float32})]
    assert unused.calls == []
    assert used.model.moved_to == "cpu"
    # Warmup ran one bounded generation.
    assert used.model.generate_calls[0]["max_length"] == max_length_for(16000, 6.5)


def test_moonshine_model_class_selection_uses_transformers_classes():
    assert model_class_for(SimpleNamespace(model_type="moonshine")).__name__ == "MoonshineForConditionalGeneration"
    assert (
        model_class_for(SimpleNamespace(model_type="moonshine_streaming")).__name__
        == "MoonshineStreamingForConditionalGeneration"
    )


def test_moonshine_bounds_decoding_by_audio_length():
    assert max_length_for(16000 * 10, 6.5) == 65
    # Single-word turns keep enough room for a word.
    assert max_length_for(1600, 6.5) == 8


def test_moonshine_rejects_nonpositive_token_rate(monkeypatch):
    _install_moonshine(monkeypatch)
    handler = MoonshineSTTHandler.__new__(MoonshineSTTHandler)
    with pytest.raises(ValueError, match="tokens_per_second"):
        handler.setup(device="cpu", tokens_per_second=0)


@pytest.mark.parametrize(("mode", "output_type"), [("progressive", PartialTranscription), ("final", Transcription)])
def test_moonshine_emits_pipeline_transcription_events(monkeypatch, mode, output_type):
    regular, _streaming, _processor = _install_moonshine(monkeypatch)
    handler = MoonshineSTTHandler.__new__(MoonshineSTTHandler)
    handler.setup(device="cpu", gen_kwargs={"num_beams": 1})
    recorded = []
    handler._record_final_stt = lambda item, seconds: recorded.append((item.turn_id, seconds))

    [output] = list(handler.process(_vad(mode, seconds=4.0)))

    assert type(output) is output_type
    assert output.text == "hello world"
    assert output.turn_id == "turn"
    assert output.turn_revision == 2
    call = regular.model.generate_calls[-1]
    assert call["max_length"] == 26
    assert call["num_beams"] == 1
    if mode == "final":
        assert output.language_code == "en"
        assert output.speech_stopped_at_s == 5.0
        assert [turn for turn, _seconds in recorded] == ["turn"]
    else:
        assert recorded == []


def test_moonshine_cli_config_is_normalized_for_handler():
    args = parse_arguments(
        [
            "--stt",
            "moonshine",
            "--moonshine_stt_model_name",
            "moonshine-ai/moonshine-tiny",
            "--moonshine_stt_device",
            "cpu",
            "--moonshine_stt_num_threads",
            "2",
        ]
    )
    assert args.stt_backend.name == "moonshine"
    assert args.stt_backend.config == {
        "model_name": "moonshine-ai/moonshine-tiny",
        "device": "cpu",
        "torch_dtype": "auto",
        "tokens_per_second": 6.5,
        "num_threads": 2,
        "gen_kwargs": {},
    }


def test_moonshine_constructs_through_registry(monkeypatch):
    _install_moonshine(monkeypatch)
    args = parse_arguments(["--stt", "moonshine", "--moonshine_stt_device", "cpu"])
    context = _context()

    handler = create_backend_handler(args.stt_backend, context)

    assert type(handler) is MoonshineSTTHandler
    assert handler.speculative_turns is context.speculative_turns


def test_moonshine_rejects_session_language_update(monkeypatch):
    args = parse_arguments(["--stt", "moonshine", "--tts", "openai"])
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
    unit.service.tts_supported_languages = {"en", "fr"}
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
