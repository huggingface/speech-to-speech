from types import SimpleNamespace

import numpy as np

from speech_to_speech.pipeline.messages import Transcription, VADAudio
from speech_to_speech.STT import mlx_audio_whisper_handler
from speech_to_speech.STT.mlx_audio_whisper_handler import MLXAudioWhisperSTTHandler


class _FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return SimpleNamespace(text=" hello ", language="de")


class _RecordingMLXLock:
    handler_names = []

    def __init__(self, handler_name):
        self.handler_name = handler_name

    def __enter__(self):
        self.handler_names.append(self.handler_name)
        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _handler():
    handler = object.__new__(MLXAudioWhisperSTTHandler)
    handler.model = _FakeModel()
    handler.start_language = "auto"
    handler.last_language = None
    return handler


def test_mlx_audio_whisper_warmup_uses_global_mlx_lock(monkeypatch):
    _RecordingMLXLock.handler_names = []
    handler = _handler()
    monkeypatch.setattr(mlx_audio_whisper_handler, "MLXLockContext", _RecordingMLXLock)

    handler.warmup()

    assert _RecordingMLXLock.handler_names == ["MLXAudioWhisperSTTHandler"]
    assert len(handler.model.calls) == 1
    assert handler.model.calls[0][1] == {"verbose": False}


def test_mlx_audio_whisper_process_uses_global_mlx_lock(monkeypatch):
    _RecordingMLXLock.handler_names = []
    handler = _handler()
    monkeypatch.setattr(mlx_audio_whisper_handler, "MLXLockContext", _RecordingMLXLock)
    monkeypatch.setattr(mlx_audio_whisper_handler.console, "print", lambda *args, **kwargs: None)

    result = list(
        handler.process(
            VADAudio(
                audio=np.zeros(16000, dtype=np.float32),
                mode="final",
                turn_id="turn_1",
                turn_revision=2,
                created_at_s=123.0,
            )
        )
    )

    assert _RecordingMLXLock.handler_names == ["MLXAudioWhisperSTTHandler"]
    assert len(handler.model.calls) == 1
    assert handler.model.calls[0][1] == {"verbose": False}
    assert len(result) == 1
    assert isinstance(result[0], Transcription)
    assert result[0].text == "hello"
    assert result[0].language_code == "de-auto"
    assert result[0].turn_id == "turn_1"
    assert result[0].turn_revision == 2
    assert result[0].speech_stopped_at_s == 123.0
