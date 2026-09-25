from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from speech_to_speech.pipeline.messages import Transcription, VADAudio
from speech_to_speech.s2s_pipeline import parse_arguments
from speech_to_speech.STT import nemo_asr_handler
from speech_to_speech.STT.nemo_asr_handler import NemoASRSTTHandler
from tests.test_nemo_asr_handler import _install_fake_nemo


@pytest.fixture(autouse=True)
def _quiet_console(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nemo_asr_handler.console, "print", lambda *args, **kwargs: None)


def _vad_audio(*, turn_id: str = "turn_1") -> VADAudio:
    return VADAudio(
        audio=np.zeros(16000, dtype=np.float32),
        mode="final",
        turn_id=turn_id,
        turn_revision=0,
        created_at_s=1.0,
    )


def test_cli_nemotron_streaming_defaults() -> None:
    args = parse_arguments(["--stt", "nemotron-streaming"])

    assert args.stt_backend.name == "nemotron-streaming"
    assert args.stt_backend.config["model_name"] == "nvidia/nemotron-speech-streaming-en-0.6b"
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


def test_multilingual_nemotron_uses_detected_language_per_utterance(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = ["warmup", "Bonjour. <fr-FR>", "Hello. <en-US>"]
    prompts: list[str] = []
    transcribe_kwargs: list[dict[str, Any]] = []

    class FakeASRModel:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> FakeASRModel:
            return cls()

        def to(self, device: str) -> FakeASRModel:
            return self

        def set_inference_prompt(self, lang: str) -> None:
            prompts.append(lang)

        def transcribe(self, audio: Any, **kwargs: Any) -> list[str]:
            transcribe_kwargs.append(kwargs)
            return [outputs.pop(0)]

    _install_fake_nemo(monkeypatch, FakeASRModel)
    handler = object.__new__(NemoASRSTTHandler)
    handler.setup(
        model_name="nvidia/nemotron-3.5-asr-streaming-0.6b",
        device="cpu",
        language="en",
    )

    french = list(handler.process(_vad_audio(turn_id="turn_1")))
    english = list(handler.process(_vad_audio(turn_id="turn_2")))

    assert isinstance(french[0], Transcription)
    assert french[0].text == "Bonjour."
    assert french[0].language_code == "fr"
    assert isinstance(english[0], Transcription)
    assert english[0].text == "Hello."
    assert english[0].language_code == "en"
    assert handler.last_language == "en"
    assert "auto" in prompts
    assert any(kwargs.get("target_lang") == "auto" for kwargs in transcribe_kwargs)


def test_english_nemotron_keeps_configured_language(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = ["warmup", "Hello. <fr-FR>"]

    class FakeASRModel:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> FakeASRModel:
            return cls()

        def to(self, device: str) -> FakeASRModel:
            return self

        def transcribe(self, audio: Any, **kwargs: Any) -> list[str]:
            assert kwargs == {}
            return [outputs.pop(0)]

    _install_fake_nemo(monkeypatch, FakeASRModel)
    handler = object.__new__(NemoASRSTTHandler)
    handler.setup(
        model_name="nvidia/nemotron-speech-streaming-en-0.6b",
        device="cpu",
        language="en",
    )

    events = list(handler.process(_vad_audio()))

    assert events[0].text == "Hello. <fr-FR>"
    assert events[0].language_code == "en"
