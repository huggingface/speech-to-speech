"""The whisper STT handler must not write its language into the shared default.

`setup` takes `gen_kwargs: dict[str, Any] = {}`, so every call that omits the
argument receives the same dict object. `setup` then stores `language` in it,
which means the value survives in the default for the life of the process and is
handed to the next handler built without an explicit `gen_kwargs`.
"""

from typing import Any

import pytest
import torch

from speech_to_speech.STT import whisper_stt_handler
from speech_to_speech.STT.whisper_stt_handler import WhisperSTTHandler


class _FakeConfig:
    num_mel_bins = 80


class _FakeModel:
    config = _FakeConfig()

    def to(self, device: str) -> "_FakeModel":
        return self

    def generate(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.zeros((1, 4), dtype=torch.long)


@pytest.fixture
def stub_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the real `setup` run without downloading or loading a model."""
    monkeypatch.setattr(
        whisper_stt_handler.AutoProcessor, "from_pretrained", classmethod(lambda cls, *a, **k: object())
    )
    monkeypatch.setattr(
        whisper_stt_handler.AutoModelForSpeechSeq2Seq,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: _FakeModel()),
    )


def _make_handler(**setup_kwargs: Any) -> WhisperSTTHandler:
    handler = object.__new__(WhisperSTTHandler)
    handler.setup(device="cpu", torch_dtype="float32", **setup_kwargs)
    return handler


def test_setup_does_not_mutate_the_shared_default(stub_transformers: None) -> None:
    default_gen_kwargs = WhisperSTTHandler.setup.__defaults__[-1]
    assert default_gen_kwargs == {}, "precondition: the default starts empty"

    _make_handler(language="fr")

    assert default_gen_kwargs == {}


def test_language_does_not_leak_into_a_later_handler(stub_transformers: None) -> None:
    _make_handler(language="fr")

    without_language = _make_handler()

    assert "language" not in without_language.gen_kwargs
    assert without_language.last_language is None


def test_handlers_do_not_share_one_gen_kwargs_dict(stub_transformers: None) -> None:
    first = _make_handler(language="fr")
    second = _make_handler(language="ja")

    assert first.gen_kwargs is not second.gen_kwargs
    assert first.gen_kwargs["language"] == "fr"
    assert second.gen_kwargs["language"] == "ja"


def test_caller_supplied_gen_kwargs_is_left_alone(stub_transformers: None) -> None:
    caller_dict: dict[str, Any] = {"max_new_tokens": 128}

    handler = _make_handler(language="fr", gen_kwargs=caller_dict)

    assert caller_dict == {"max_new_tokens": 128}
    assert handler.gen_kwargs == {"max_new_tokens": 128, "language": "fr"}


def test_explicit_gen_kwargs_still_carries_the_language(stub_transformers: None) -> None:
    handler = _make_handler(language="es", gen_kwargs={"num_beams": 1})

    assert handler.gen_kwargs == {"num_beams": 1, "language": "es"}


def test_auto_language_is_not_written_into_gen_kwargs(stub_transformers: None) -> None:
    handler = _make_handler(language="auto")

    assert handler.gen_kwargs == {}
    assert handler.start_language == "auto"
    assert handler.last_language is None
