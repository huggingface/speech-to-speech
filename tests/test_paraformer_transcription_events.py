import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio
from speech_to_speech.STT import paraformer_handler
from speech_to_speech.STT.paraformer_handler import ParaformerSTTHandler


class _FakeParaformerModel:
    def generate(self, audio):
        return [{"text": " 今 天 天 气 不 错 "}]


def _handler(*, language: str = "zh", device: str = "cpu"):
    handler = object.__new__(ParaformerSTTHandler)
    handler.model = _FakeParaformerModel()
    handler.language = language
    handler.device = device
    return handler


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("paraformer-zh", "zh"),
        ("paraformer-en", "en"),
        ("funasr/paraformer-en", "en"),
        ("paraformer-zh-streaming", "zh"),
        ("paraformer", "zh"),
    ],
)
def test_setup_extracts_language_from_model_name(monkeypatch, model_name, expected):
    fake_model = MagicMock()
    fake_model.generate.return_value = [{"text": "warmup"}]
    fake_funasr = MagicMock()
    fake_funasr.AutoModel = MagicMock(return_value=fake_model)
    monkeypatch.setitem(sys.modules, "funasr", fake_funasr)
    monkeypatch.setattr(paraformer_handler.torch.mps, "empty_cache", lambda: None)

    handler = object.__new__(ParaformerSTTHandler)
    handler.setup(model_name=model_name, device="cpu")

    assert handler.language == expected


def test_progressive_paraformer_transcription_is_partial(monkeypatch):
    monkeypatch.setattr(paraformer_handler.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(paraformer_handler.torch.mps, "empty_cache", lambda: None)

    result = list(
        _handler().process(
            VADAudio(
                audio=np.zeros(16000, dtype=np.float32),
                mode="progressive",
                turn_id="turn_1",
                turn_revision=2,
            )
        )
    )

    assert len(result) == 1
    assert isinstance(result[0], PartialTranscription)
    assert result[0].text == "今天天气不错"
    assert result[0].turn_id == "turn_1"
    assert result[0].turn_revision == 2
    assert not hasattr(result[0], "language_code")


def test_final_paraformer_transcription_is_final(monkeypatch):
    monkeypatch.setattr(paraformer_handler.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(paraformer_handler.torch.mps, "empty_cache", lambda: None)

    result = list(
        _handler().process(
            VADAudio(
                audio=np.zeros(16000, dtype=np.float32),
                mode="final",
                turn_id="turn_1",
                turn_revision=2,
                created_at_s=124.0,
                speech_end_at_s=123.0,
            )
        )
    )

    assert len(result) == 1
    assert isinstance(result[0], Transcription)
    assert result[0].text == "今天天气不错"
    assert result[0].language_code == "zh"
    assert result[0].turn_id == "turn_1"
    assert result[0].turn_revision == 2
    assert result[0].speech_stopped_at_s == 123.0


def test_final_paraformer_transcription_uses_english_checkpoint_language(monkeypatch):
    monkeypatch.setattr(paraformer_handler.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(paraformer_handler.torch.mps, "empty_cache", lambda: None)

    result = list(
        _handler(language="en").process(
            VADAudio(
                audio=np.zeros(16000, dtype=np.float32),
                mode="final",
            )
        )
    )

    assert result[0].language_code == "en"


def test_paraformer_skips_mps_cache_clear_on_cpu(monkeypatch):
    monkeypatch.setattr(paraformer_handler.console, "print", lambda *args, **kwargs: None)

    def fail_if_called():
        raise AssertionError("torch.mps.empty_cache should not run on cpu")

    monkeypatch.setattr(paraformer_handler.torch.mps, "empty_cache", fail_if_called)

    result = list(
        _handler(device="cpu").process(
            VADAudio(
                audio=np.zeros(16000, dtype=np.float32),
                mode="final",
            )
        )
    )

    assert len(result) == 1
    assert isinstance(result[0], Transcription)


@pytest.mark.parametrize(
    ("model_text", "expected"),
    [
        # FunASR's timestamp path joins every token with a space. Chinese text
        # should come out as its plain path returns it; English words keep theirs.
        (" 今 天 天 气 不 错 ", "今天天气不错"),
        ("hello world how are you", "hello world how are you"),
        ("我 喜 欢 python 编 程", "我喜欢python编程"),
        ("我 有 3 个", "我有3个"),
        ("今 天 ， 好 。", "今天，好。"),
        ("i have 3 apples", "i have 3 apples"),
    ],
)
def test_paraformer_keeps_spaces_only_between_latin_or_digit_words(monkeypatch, model_text, expected):
    monkeypatch.setattr(paraformer_handler.console, "print", lambda *args, **kwargs: None)
    handler = _handler()
    handler.model = MagicMock()
    handler.model.generate.return_value = [{"text": model_text}]

    result = list(handler.process(VADAudio(audio=np.zeros(16000, dtype=np.float32), mode="final")))

    assert result[0].text == expected
