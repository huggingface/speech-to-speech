import numpy as np

from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio
from speech_to_speech.STT import sense_voice_handler
from speech_to_speech.STT.sense_voice_handler import SenseVoiceSTTHandler


class RecordingModel:
    def __init__(self, text: str) -> None:
        self.text = text
        self.audio = None
        self.kwargs = None

    def generate(self, audio, **kwargs):
        self.audio = audio
        self.kwargs = kwargs
        return [{"text": self.text}]


def make_handler(text: str, *, language: str = "auto") -> tuple[SenseVoiceSTTHandler, RecordingModel, list[bool]]:
    handler = object.__new__(SenseVoiceSTTHandler)
    model = RecordingModel(text)
    cache_clears: list[bool] = []
    handler.model = model
    handler.language = language
    handler.device = "cpu"
    handler._postprocess = lambda value: value.replace("<tags>", "").strip()
    handler._empty_cache = lambda: cache_clears.append(True)
    return handler, model, cache_clears


def test_process_yields_final_transcription_with_turn_metadata(monkeypatch):
    handler, model, cache_clears = make_handler(" <tags>Hello world ")
    audio = np.linspace(-1.0, 1.0, 16000, dtype=np.float32)
    monkeypatch.setattr(sense_voice_handler.console, "print", lambda *args, **kwargs: None)

    result = list(
        handler.process(
            VADAudio(
                audio=audio,
                mode="final",
                turn_id="turn-7",
                turn_revision=3,
                created_at_s=123.5,
            )
        )
    )

    assert len(result) == 1
    assert isinstance(result[0], Transcription)
    assert result[0].text == "Hello world"
    assert result[0].turn_id == "turn-7"
    assert result[0].turn_revision == 3
    assert result[0].speech_stopped_at_s == 123.5
    assert model.audio is audio
    assert model.kwargs == {"cache": {}, "language": "auto", "use_itn": True}
    assert cache_clears == [True]


def test_process_yields_partial_transcription_for_progressive_audio(monkeypatch):
    handler, model, cache_clears = make_handler("<tags>Ni hao", language="zh")
    audio = np.zeros(8000, dtype=np.float32)
    monkeypatch.setattr(sense_voice_handler.console, "print", lambda *args, **kwargs: None)

    result = list(
        handler.process(
            VADAudio(
                audio=audio,
                mode="progressive",
                turn_id="turn-8",
                turn_revision=1,
            )
        )
    )

    assert len(result) == 1
    assert isinstance(result[0], PartialTranscription)
    assert result[0].text == "Ni hao"
    assert result[0].turn_id == "turn-8"
    assert result[0].turn_revision == 1
    assert model.audio is audio
    assert model.kwargs == {"cache": {}, "language": "zh", "use_itn": True}
    assert cache_clears == [True]
