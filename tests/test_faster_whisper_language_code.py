"""FasterWhisper must put detected language on Transcription like other STT backends.

``model.transcribe()`` already returns ``info.language``, but the handler used to omit
``language_code`` on ``Transcription``. Downstream lang-prompt / multilingual TTS then
saw ``None`` and could not switch language.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio


def _import_handler(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", MagicMock())
    sys.modules.pop("speech_to_speech.STT.faster_whisper_handler", None)
    from speech_to_speech.STT.faster_whisper_handler import FasterWhisperSTTHandler

    return FasterWhisperSTTHandler


def _make_handler(monkeypatch, *, segments, language: str | None):
    FasterWhisperSTTHandler = _import_handler(monkeypatch)
    handler = object.__new__(FasterWhisperSTTHandler)
    handler.gen_kwargs = {}
    handler.start_language = language
    handler.model = MagicMock()
    handler.model.transcribe.return_value = (segments, SimpleNamespace(language=language))
    return handler


def test_faster_whisper_propagates_language_code(monkeypatch) -> None:
    segment = SimpleNamespace(start=0.0, end=1.0, text=" Jaka pogoda?")
    handler = _make_handler(monkeypatch, segments=[segment], language="pl")

    outputs = list(
        handler.process(
            VADAudio(
                audio=np.zeros(1600, dtype=np.float32),
                turn_id="turn-1",
                turn_revision=1,
            )
        )
    )

    assert len(outputs) == 1
    assert isinstance(outputs[0], Transcription)
    assert outputs[0].text == "Jaka pogoda?"
    assert outputs[0].language_code == "pl"
    assert outputs[0].turn_id == "turn-1"
    assert outputs[0].turn_revision == 1


def test_faster_whisper_skips_empty_transcript(monkeypatch) -> None:
    segment = SimpleNamespace(start=0.0, end=0.1, text="   ")
    handler = _make_handler(monkeypatch, segments=[segment], language="en")

    outputs = list(handler.process(VADAudio(audio=np.zeros(1600, dtype=np.float32))))

    assert outputs == []


def test_faster_whisper_progressive_partial_has_no_language_field(monkeypatch) -> None:
    """PartialTranscription has no language_code field; only finals carry it."""
    segment = SimpleNamespace(start=0.0, end=0.5, text=" Hello")
    handler = _make_handler(monkeypatch, segments=[segment], language="en")

    outputs = list(
        handler.process(
            VADAudio(
                audio=np.zeros(1600, dtype=np.float32),
                mode="progressive",
                turn_id="turn-1",
                turn_revision=1,
            )
        )
    )

    assert len(outputs) == 1
    assert isinstance(outputs[0], PartialTranscription)
    assert outputs[0].text == "Hello"
    assert not hasattr(outputs[0], "language_code")
