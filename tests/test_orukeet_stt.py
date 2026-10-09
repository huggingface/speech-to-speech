from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from speech_to_speech.backend_registry import create_backend_handler
from speech_to_speech.pipeline.messages import Transcription, VADAudio
from speech_to_speech.s2s_pipeline import parse_arguments
from speech_to_speech.STT import nemo_asr_handler
from speech_to_speech.STT.nemo_asr_handler import NemoASRSTTHandler
from tests.test_nemo_asr_handler import _context, _install_fake_nemo


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


def test_cli_orukeet_defaults() -> None:
    args = parse_arguments(["--stt", "orukeet"])

    assert args.stt_backend.name == "orukeet"
    assert args.stt_backend.config["model_name"] == "oruk/orukeet"
    assert args.stt_backend.config["checkpoint_filename"] == "orukeet-v0.1.0.nemo"
    assert args.stt_backend.config["checkpoint_revision"] == "555136b50265a132d4cea0d35560c26fc4f657ab"
    assert args.stt_backend.config["device"] == "auto"
    assert args.stt_backend.config["language"] == "auto"
    assert args.stt_backend.spec.required_extra == "nemo"
    assert "detect_language_from_text" not in args.stt_backend.config


def test_orukeet_setup_downloads_and_restore_from(monkeypatch: pytest.MonkeyPatch) -> None:
    downloads: list[tuple[Any, ...]] = []
    restored: list[str] = []
    pretrained: list[Any] = []
    warmed: list[Any] = []

    def fake_download(repo_id: str, filename: str, revision: str | None = None) -> str:
        downloads.append((repo_id, filename, revision))
        return "/tmp/orukeet-v0.1.0.nemo"

    class FakeASRModel:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> FakeASRModel:
            pretrained.append((args, kwargs))
            return cls()

        @classmethod
        def restore_from(cls, path: str) -> FakeASRModel:
            restored.append(path)
            return cls()

        def to(self, device: str) -> FakeASRModel:
            return self

        def transcribe(self, audio: Any) -> list[str]:
            return ["warmup"]

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fake_download))
    _install_fake_nemo(monkeypatch, FakeASRModel)
    monkeypatch.setattr(
        nemo_asr_handler,
        "warm_language_detector",
        lambda languages=None: warmed.append(languages) or object(),
    )

    args = parse_arguments(["--stt", "orukeet", "--orukeet_device", "cpu"])
    handler = create_backend_handler(args.stt_backend, _context())

    assert downloads == [("oruk/orukeet", "orukeet-v0.1.0.nemo", "555136b50265a132d4cea0d35560c26fc4f657ab")]
    assert restored == ["/tmp/orukeet-v0.1.0.nemo"]
    assert pretrained == []
    assert handler._detect_language_from_text is True
    assert warmed == [tuple(nemo_asr_handler.TEXT_DETECTION_LANGUAGES)]
    assert handler.start_language == "auto"
    assert handler.language == "en"
    assert handler.last_language == "en"
    assert handler.model_name == "oruk/orukeet"


@pytest.mark.parametrize("filename", ["", "   "])
def test_orukeet_empty_checkpoint_filename_errors(monkeypatch: pytest.MonkeyPatch, filename: str) -> None:
    pretrained: list[Any] = []

    class FakeASRModel:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> FakeASRModel:
            pretrained.append((args, kwargs))
            return cls()

        @classmethod
        def restore_from(cls, path: str) -> FakeASRModel:
            raise AssertionError("restore_from should not run for an empty checkpoint filename")

        def to(self, device: str) -> FakeASRModel:
            return self

        def transcribe(self, audio: Any) -> list[str]:
            return ["warmup"]

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda *args, **kwargs: "/tmp/orukeet-v0.1.0.nemo"),
    )
    _install_fake_nemo(monkeypatch, FakeASRModel)
    monkeypatch.setattr(nemo_asr_handler, "warm_language_detector", lambda languages=None: object())

    args = parse_arguments(["--stt", "orukeet", "--orukeet_device", "cpu", "--orukeet_checkpoint_filename", filename])
    with pytest.raises(ValueError, match="checkpoint_filename is empty"):
        create_backend_handler(args.stt_backend, _context())
    assert pretrained == []


def test_orukeet_missing_nemo_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "nemo",
        "nemo.collections",
        "nemo.collections.asr",
        "nemo.collections.asr.models",
    ):
        monkeypatch.setitem(sys.modules, name, None)

    args = parse_arguments(["--stt", "orukeet"])
    with pytest.raises(ImportError, match=r"speech-to-speech\[nemo\]"):
        create_backend_handler(args.stt_backend, _context())


def test_orukeet_reports_detected_language_per_final(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    outputs = ["warmup", "Bonjour, comment allez-vous aujourd'hui?", "Hello, how are you doing today?"]
    detected = iter(["fr", "en"])

    class FakeASRModel:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> FakeASRModel:
            raise AssertionError("from_pretrained should not run for Orukeet")

        @classmethod
        def restore_from(cls, path: str) -> FakeASRModel:
            return cls()

        def to(self, device: str) -> FakeASRModel:
            return self

        def transcribe(self, audio: Any) -> list[str]:
            return [outputs.pop(0)]

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda *args, **kwargs: "/tmp/orukeet-v0.1.0.nemo"),
    )
    _install_fake_nemo(monkeypatch, FakeASRModel)
    monkeypatch.setattr(nemo_asr_handler, "warm_language_detector", lambda languages=None: object())
    monkeypatch.setattr(
        nemo_asr_handler,
        "detect_language_from_text",
        lambda *args, **kwargs: next(detected),
    )

    handler = object.__new__(NemoASRSTTHandler)
    with caplog.at_level("WARNING"):
        handler.setup(
            model_name="oruk/orukeet",
            device="cpu",
            language="auto",
            checkpoint_filename="orukeet-v0.1.0.nemo",
            checkpoint_revision="555136b50265a132d4cea0d35560c26fc4f657ab",
            detect_language_from_text=True,
        )

    assert "not a code" not in caplog.text

    french = list(handler.process(_vad_audio(turn_id="turn_1")))
    english = list(handler.process(_vad_audio(turn_id="turn_2")))

    assert isinstance(french[0], Transcription)
    assert french[0].text == "Bonjour, comment allez-vous aujourd'hui?"
    assert french[0].language_code == "fr"
    assert isinstance(english[0], Transcription)
    assert english[0].text == "Hello, how are you doing today?"
    assert english[0].language_code == "en"
    assert handler.last_language == "en"


def test_orukeet_reports_greek_with_real_language_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "Καλημέρα σας. Θα ήθελα να μάθω περισσότερα για αυτό το προϊόν."

    class FakeASRModel:
        @classmethod
        def restore_from(cls, path: str) -> FakeASRModel:
            return cls()

        def to(self, device: str) -> FakeASRModel:
            return self

        def transcribe(self, audio: Any) -> list[str]:
            return [text]

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda *args, **kwargs: "/tmp/orukeet-v0.1.0.nemo"),
    )
    _install_fake_nemo(monkeypatch, FakeASRModel)

    # Keep the real detector so missing candidates cannot be hidden by a mock.
    args = parse_arguments(["--stt", "orukeet", "--orukeet_device", "cpu"])
    handler = create_backend_handler(args.stt_backend, _context())

    events = list(handler.process(_vad_audio()))

    assert len(events) == 1
    assert isinstance(events[0], Transcription)
    assert events[0].text == text
    assert events[0].language_code == "el"
    assert handler.last_language == "el"
