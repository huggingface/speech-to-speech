from __future__ import annotations

import logging
from queue import Queue
from threading import Event
from typing import Any

import numpy as np
import pytest
import torch

from speech_to_speech.backend_registry import HandlerContext, create_backend_handler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.s2s_pipeline import parse_arguments
from speech_to_speech.STT import qwen3_asr_handler
from speech_to_speech.STT.qwen3_asr_handler import (
    Qwen3ASRSTTHandler,
    language_to_code,
    resolve_device,
    resolve_torch_dtype,
)

_PROMPT_TOKENS = 3


class _FakeInputs(dict):
    """Mimics the BatchFeature returned by ``apply_transcription_request``."""

    def to(self, *args: Any, **kwargs: Any) -> "_FakeInputs":
        return self


class _FakeProcessor:
    """Mirrors the real Qwen3ASRProcessor: ``parsed`` carries a language only in auto mode."""

    def __init__(self, language: str | None = "English", text: str = "hello world") -> None:
        self.language = language
        self.text = text
        self.requests: list[dict[str, Any]] = []

    def apply_transcription_request(
        self, audio: Any, language: str | None = None, prompt: str | None = None
    ) -> _FakeInputs:
        self.requests.append({"language": language, "prompt": prompt, "samples": len(audio)})
        return _FakeInputs(input_ids=torch.zeros((1, _PROMPT_TOKENS), dtype=torch.long))

    def decode(self, generated_ids: Any, return_format: str = "raw") -> list[Any]:
        forced = self.requests[-1]["language"] is not None
        language = None if forced else self.language
        if return_format == "transcription_only":
            return [self.text]
        if return_format == "parsed":
            return [{"language": language, "transcription": self.text}]
        return [f"language {language}<asr_text>{self.text}"]


class _FakeModel:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []

    def to(self, device: str) -> "_FakeModel":
        return self

    def eval(self) -> "_FakeModel":
        return self

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.generate_calls.append(kwargs)
        return torch.zeros((1, _PROMPT_TOKENS + 4), dtype=torch.long)


def _handler(
    *,
    language: str = "auto",
    processor: _FakeProcessor | None = None,
    prompt: str | None = None,
    gen_kwargs: dict[str, Any] | None = None,
) -> Qwen3ASRSTTHandler:
    handler = object.__new__(Qwen3ASRSTTHandler)
    handler.processor = processor or _FakeProcessor()
    handler.model = _FakeModel()
    handler.device = "cpu"
    handler.torch_dtype = torch.float32
    handler.prompt = prompt
    handler.gen_kwargs = dict(gen_kwargs or {})
    handler.configure_language(language)
    return handler


def _vad_audio(mode: str = "final", seconds: float = 2.0, revision: int = 1) -> VADAudio:
    return VADAudio(
        audio=np.zeros(int(16000 * seconds), dtype=np.float32),
        mode=mode,
        turn_id="turn_1",
        turn_revision=revision,
        created_at_s=123.0,
    )


@pytest.fixture(autouse=True)
def _quiet_console(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qwen3_asr_handler.console, "print", lambda *args, **kwargs: None)


# ── language and device helpers ──────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("English", "en"),
        ("english", "en"),
        ("Cantonese", "yue"),
        ("Chinese", "zh"),
        ("zh", "zh"),
        (" French ", "fr"),
        ("Klingon", None),
        ("None", None),
        (None, None),
        ("", None),
    ],
)
def test_language_to_code(value: str | None, expected: str | None) -> None:
    assert language_to_code(value) == expected


def _hardware(monkeypatch: pytest.MonkeyPatch, *, cuda: bool, mps: bool = False, bf16: bool = True) -> None:
    monkeypatch.setattr(qwen3_asr_handler.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(qwen3_asr_handler.torch.cuda, "is_bf16_supported", lambda: bf16)
    monkeypatch.setattr(qwen3_asr_handler.torch.backends.mps, "is_available", lambda: mps)


def test_resolve_device_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    _hardware(monkeypatch, cuda=True, mps=True)
    assert resolve_device("auto") == "cuda"


def test_resolve_device_auto_uses_mps_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _hardware(monkeypatch, cuda=False, mps=True)
    assert resolve_device("auto") == "mps"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _hardware(monkeypatch, cuda=False, mps=False)
    assert resolve_device("auto") == "cpu"


def test_resolve_device_keeps_explicit_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    _hardware(monkeypatch, cuda=True)
    assert resolve_device("cpu") == "cpu"


@pytest.mark.parametrize(
    ("dtype", "device", "bf16", "expected"),
    [
        ("auto", "cuda", True, torch.bfloat16),
        ("auto", "cuda", False, torch.float16),
        ("auto", "mps", True, torch.float16),
        ("auto", "cpu", True, torch.float32),
        ("float16", "cpu", True, torch.float16),
        ("bfloat16", "cuda", False, torch.bfloat16),
    ],
)
def test_resolve_torch_dtype(
    monkeypatch: pytest.MonkeyPatch, dtype: str, device: str, bf16: bool, expected: torch.dtype
) -> None:
    _hardware(monkeypatch, cuda=True, bf16=bf16)
    assert resolve_torch_dtype(dtype, device) == expected


# ── transcription events ─────────────────────────────────────────────


def test_final_transcription_reports_text_and_detected_language() -> None:
    result = list(_handler().process(_vad_audio("final")))

    assert len(result) == 1
    assert isinstance(result[0], Transcription)
    assert result[0].text == "hello world"
    assert result[0].language_code == "en-auto"
    assert result[0].turn_id == "turn_1"
    assert result[0].turn_revision == 1
    assert result[0].speech_stopped_at_s == 123.0


def test_final_transcription_in_auto_mode_lets_the_model_detect_language() -> None:
    handler = _handler()

    list(handler.process(_vad_audio("final")))

    assert handler.processor.requests[0]["language"] is None


def test_progressive_transcription_is_partial() -> None:
    result = list(_handler().process(_vad_audio("progressive")))

    assert len(result) == 1
    assert isinstance(result[0], PartialTranscription)
    assert result[0].text == "hello world"
    assert result[0].turn_id == "turn_1"
    assert result[0].turn_revision == 1
    assert not hasattr(result[0], "language_code")


def test_progressive_reuses_last_final_language_in_auto_mode() -> None:
    """Short progressive windows fool the language ID, so partials stick to the last final language."""
    processor = _FakeProcessor(language="French", text="bonjour")
    handler = _handler(processor=processor)

    list(handler.process(_vad_audio("final")))
    list(handler.process(_vad_audio("progressive", seconds=0.5)))
    list(handler.process(_vad_audio("final")))

    assert [request["language"] for request in processor.requests] == [None, "fr", None]


def test_progressive_before_any_final_uses_auto_detection() -> None:
    handler = _handler()

    list(handler.process(_vad_audio("progressive")))

    assert handler.processor.requests[0]["language"] is None


def test_forced_language_is_passed_on_every_request_and_reported() -> None:
    processor = _FakeProcessor()
    handler = _handler(language="de", processor=processor)

    list(handler.process(_vad_audio("progressive")))
    result = list(handler.process(_vad_audio("final")))

    assert [request["language"] for request in processor.requests] == ["de", "de"]
    assert isinstance(result[0], Transcription)
    assert result[0].language_code == "de"


def test_forced_language_accepts_full_name() -> None:
    handler = _handler(language="German")

    result = list(handler.process(_vad_audio("final")))

    assert handler.processor.requests[0]["language"] == "de"
    assert result[0].language_code == "de"


def test_forced_language_outside_the_map_is_passed_through() -> None:
    """The processor validates the checkpoint's languages, so the handler does not gatekeep."""
    handler = _handler(language="sw")

    result = list(handler.process(_vad_audio("final")))

    assert handler.processor.requests[0]["language"] == "sw"
    assert result[0].language_code == "sw"


def test_unknown_detected_language_is_not_sticky_and_falls_back() -> None:
    processor = _FakeProcessor(language="English")
    handler = _handler(processor=processor)
    list(handler.process(_vad_audio("final")))

    processor.language = "Klingon"
    result = list(handler.process(_vad_audio("final")))
    list(handler.process(_vad_audio("progressive")))

    assert result[0].language_code == "en-auto"
    assert processor.requests[-1]["language"] == "en"


def test_silence_reports_no_language_and_defaults_to_english() -> None:
    """On silence the model emits ``language None<asr_text>`` with an empty transcription."""
    handler = _handler(processor=_FakeProcessor(language=None, text=""))

    result = list(handler.process(_vad_audio("final")))

    assert result[0].text == ""
    assert result[0].language_code == "en-auto"


def test_prompt_and_generation_kwargs_are_forwarded() -> None:
    handler = _handler(prompt="Vocabulary: Quilter.", gen_kwargs={"max_new_tokens": 64})

    list(handler.process(_vad_audio("final")))

    assert handler.processor.requests[0]["prompt"] == "Vocabulary: Quilter."
    assert handler.model.generate_calls[0]["max_new_tokens"] == 64


class _LegacyProcessor(_FakeProcessor):
    """Transformers 5.14.1: ``apply_transcription_request`` takes no ``prompt`` and rejects unknown kwargs."""

    def apply_transcription_request(self, audio: Any, language: str | None = None) -> _FakeInputs:  # type: ignore[override]
        return super().apply_transcription_request(audio, language=language)


def test_requests_omit_unset_language_and_prompt_for_older_transformers() -> None:
    handler = _handler(processor=_LegacyProcessor())

    result = list(handler.process(_vad_audio("final")))

    assert isinstance(result[0], Transcription)
    assert handler.processor.requests[0] == {"language": None, "prompt": None, "samples": 32000}


def test_setup_drops_the_prompt_with_a_warning_when_the_processor_lacks_it(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = _FakeTransformers()
    fake.processor = _LegacyProcessor()
    monkeypatch.setattr(qwen3_asr_handler, "transformers", fake)

    handler = object.__new__(Qwen3ASRSTTHandler)
    with caplog.at_level(logging.WARNING, logger="speech_to_speech.STT.qwen3_asr_handler"):
        handler.setup(device="cpu", prompt="Vocabulary: Quilter.")

    assert handler.prompt is None
    assert "transformers>=5.15.1" in caplog.text
    assert len(fake.model.generate_calls) == 1


def test_generation_strips_the_prompt_tokens_before_decoding() -> None:
    class _RecordingProcessor(_FakeProcessor):
        decoded_length = 0

        def decode(self, generated_ids: Any, return_format: str = "raw") -> list[Any]:
            self.decoded_length = generated_ids.shape[1]
            return super().decode(generated_ids, return_format)

    processor = _RecordingProcessor()
    handler = _handler(processor=processor)

    list(handler.process(_vad_audio("final")))

    assert processor.decoded_length == 4


# ── setup ────────────────────────────────────────────────────────────


class _FakeTransformers:
    """Stands in for the ``transformers`` module inside the handler."""

    def __init__(self) -> None:
        self.processor = _FakeProcessor()
        self.model = _FakeModel()
        self.loaded: list[tuple[str, Any]] = []

        outer = self

        class AutoProcessor:
            @staticmethod
            def from_pretrained(name: str) -> _FakeProcessor:
                outer.loaded.append(("processor", name))
                return outer.processor

        class AutoModelForMultimodalLM:
            @staticmethod
            def from_pretrained(name: str, **kwargs: Any) -> _FakeModel:
                outer.loaded.append(("model", kwargs.get("dtype")))
                return outer.model

        self.AutoProcessor = AutoProcessor
        self.AutoModelForMultimodalLM = AutoModelForMultimodalLM


def test_setup_loads_model_warms_up_and_logs(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    fake = _FakeTransformers()
    monkeypatch.setattr(qwen3_asr_handler, "transformers", fake)

    handler = object.__new__(Qwen3ASRSTTHandler)
    with caplog.at_level(logging.INFO, logger="speech_to_speech.STT.qwen3_asr_handler"):
        handler.setup(model_name="Qwen/Qwen3-ASR-0.6B-hf", device="cpu", torch_dtype="float32", language="auto")

    assert ("processor", "Qwen/Qwen3-ASR-0.6B-hf") in fake.loaded
    assert ("model", torch.float32) in fake.loaded
    assert len(fake.model.generate_calls) == 1, "setup must run one warmup generation"
    assert "Loading Qwen3-ASR STT model: Qwen/Qwen3-ASR-0.6B-hf" in caplog.text
    assert handler.last_language is None


def test_setup_does_not_share_generation_kwargs_between_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qwen3_asr_handler, "transformers", _FakeTransformers())

    first = object.__new__(Qwen3ASRSTTHandler)
    first.setup(device="cpu")
    first.gen_kwargs["max_new_tokens"] = 1
    second = object.__new__(Qwen3ASRSTTHandler)
    second.setup(device="cpu")

    assert second.gen_kwargs.get("max_new_tokens") != 1


# ── registry and CLI wiring ──────────────────────────────────────────


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


def test_cli_builds_a_qwen3_asr_handler_from_its_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeTransformers()
    monkeypatch.setattr(qwen3_asr_handler, "transformers", fake)
    args = parse_arguments(["--stt", "qwen3-asr", "--qwen3_asr_language", "fr", "--qwen3_asr_gen_max_new_tokens", "64"])
    context = _context()

    handler = create_backend_handler(args.stt_backend, context)

    assert isinstance(handler, Qwen3ASRSTTHandler)
    assert handler.speculative_turns is context.speculative_turns
    assert handler.forced_language == "fr"
    assert handler.gen_kwargs == {"max_new_tokens": 64}
    assert ("processor", "Qwen/Qwen3-ASR-0.6B-hf") in fake.loaded


def test_registry_normalizes_qwen3_asr_arguments() -> None:
    from speech_to_speech.arguments_classes.qwen3_asr_stt_arguments import Qwen3ASRSTTHandlerArguments
    from speech_to_speech.backend_registry import STT_BACKENDS, select_backend

    config = Qwen3ASRSTTHandlerArguments(qwen3_asr_language="fr", qwen3_asr_gen_max_new_tokens=64)
    selection = select_backend(STT_BACKENDS, "qwen3-asr", config)

    assert selection.config["model_name"] == "Qwen/Qwen3-ASR-0.6B-hf"
    assert selection.config["device"] == "auto"
    assert selection.config["torch_dtype"] == "auto"
    assert selection.config["language"] == "fr"
    assert selection.config["prompt"] is None
    assert selection.config["gen_kwargs"] == {"max_new_tokens": 64}
    assert selection.spec.required_extra is None
