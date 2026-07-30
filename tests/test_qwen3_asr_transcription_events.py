import numpy as np

from speech_to_speech.pipeline.messages import Transcription, VADAudio
from speech_to_speech.STT import qwen3_asr_handler
from speech_to_speech.STT.qwen3_asr_handler import Qwen3ASRSTTHandler


class _FakeQwen3ASRResult:
    def __init__(self, text: str, language: str) -> None:
        self.text = text
        self.language = language


class _FakeQwen3ASRModel:
    def __init__(self, language: str = "English") -> None:
        self.language = language

    def transcribe(self, audio, language=None):
        return [_FakeQwen3ASRResult(text="hello there", language=self.language)]


def _handler(start_language=None, model_language="English"):
    handler = object.__new__(Qwen3ASRSTTHandler)
    handler.model = _FakeQwen3ASRModel(language=model_language)
    handler.start_language = start_language
    handler.last_language_code = start_language if start_language != "auto" else None
    handler.language_name = None
    return handler


def test_qwen3_asr_transcription_is_final(monkeypatch):
    monkeypatch.setattr(qwen3_asr_handler.console, "print", lambda *args, **kwargs: None)

    result = list(
        _handler().process(
            VADAudio(
                audio=np.zeros(16000, dtype=np.float32),
                mode="final",
                turn_id="turn_1",
                turn_revision=2,
                created_at_s=123.0,
            )
        )
    )

    assert len(result) == 1
    assert isinstance(result[0], Transcription)
    assert result[0].text == "hello there"
    assert result[0].language_code == "en"
    assert result[0].turn_id == "turn_1"
    assert result[0].turn_revision == 2
    assert result[0].speech_stopped_at_s == 123.0


def test_qwen3_asr_auto_language_appends_suffix(monkeypatch):
    monkeypatch.setattr(qwen3_asr_handler.console, "print", lambda *args, **kwargs: None)

    result = list(
        _handler(start_language="auto", model_language="Chinese").process(
            VADAudio(audio=np.zeros(16000, dtype=np.float32), mode="final", turn_id="turn_2", turn_revision=1)
        )
    )

    assert result[0].language_code == "zh-auto"
