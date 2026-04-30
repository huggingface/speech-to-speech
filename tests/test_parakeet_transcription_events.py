from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np

from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio
from speech_to_speech.STT import parakeet_tdt_handler
from speech_to_speech.STT.parakeet_tdt_handler import ParakeetTDTSTTHandler
from speech_to_speech.STT.smart_progressive_streaming import SmartProgressiveStreamingHandler


def test_show_progressive_transcription_returns_combined_text(monkeypatch):
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.streaming_handler = SimpleNamespace(
        transcribe_incremental=lambda audio: SimpleNamespace(
            fixed_text="I just wanted",
            active_text="to check in",
        )
    )
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: None)

    result = handler._show_progressive_transcription(np.zeros(16000, dtype=np.float32))

    assert result == "I just wanted to check in"


def test_process_yields_partial_tagged_tuple(monkeypatch):
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.enable_live_transcription = True
    handler.processing_final = False

    @contextmanager
    def fake_lock(*args, **kwargs):
        yield True

    handler._compute_lock_context = fake_lock
    handler._show_progressive_transcription = lambda audio: "partial text"
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: None)

    result = list(handler.process(VADAudio(audio=np.zeros(16000, dtype=np.float32), mode="progressive")))

    assert len(result) == 1
    assert isinstance(result[0], PartialTranscription)
    assert result[0].text == "partial text"


def test_process_yields_final_transcript(monkeypatch):
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.enable_live_transcription = False
    handler.backend = "nano_parakeet"
    handler.last_language = "en"
    handler.start_language = None

    @contextmanager
    def fake_lock(*args, **kwargs):
        yield True

    handler._compute_lock_context = fake_lock
    handler._process_nano_parakeet = lambda audio_input: ("I am here.", "en")
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: None)

    result = list(handler.process(VADAudio(audio=np.zeros(16000, dtype=np.float32))))

    assert len(result) == 1
    assert isinstance(result[0], Transcription)
    assert result[0].text == "I am here."
    assert result[0].language_code == "en"


def test_final_transcription_resets_live_streaming_state(monkeypatch):
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.enable_live_transcription = True
    handler.backend = "nano_parakeet"
    handler.last_language = "en"
    handler.start_language = None
    handler.processing_final = False
    reset_calls = []
    handler.streaming_handler = SimpleNamespace(reset=lambda: reset_calls.append(True))

    @contextmanager
    def fake_lock(*args, **kwargs):
        yield True

    handler._compute_lock_context = fake_lock
    handler._process_nano_parakeet = lambda audio_input: ("I am here.", "en")
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: None)

    result = list(handler.process(VADAudio(audio=np.zeros(16000, dtype=np.float32), mode="final")))

    assert len(result) == 1
    assert isinstance(result[0], Transcription)
    assert handler.processing_final is False
    assert reset_calls == [True]


def test_progressive_stream_preserves_fixed_text_when_fixed_end_exceeds_audio():
    class Model:
        def __init__(self):
            self.lengths = []

        def transcribe(self, audio, timestamps=True):
            self.lengths.append(len(audio))
            return SimpleNamespace(text="fresh partial", timestamp={"segment": []})

    model = Model()
    handler = SmartProgressiveStreamingHandler(model)
    handler.fixed_sentences = ["stale text"]
    handler.fixed_end_time = 10.0

    result = handler.transcribe_incremental(np.zeros(16000, dtype=np.float32))

    assert model.lengths == []
    assert result.fixed_text == "stale text"
    assert result.active_text == ""
    assert handler.fixed_end_time == 10.0


def test_progressive_stream_keeps_fixed_text_at_exact_boundary():
    class Model:
        def __init__(self):
            self.lengths = []

        def transcribe(self, audio, timestamps=True):
            self.lengths.append(len(audio))
            return SimpleNamespace(text="active", timestamp={"segment": []})

    model = Model()
    handler = SmartProgressiveStreamingHandler(model)
    handler.fixed_sentences = ["kept fixed"]
    handler.fixed_end_time = 1.0

    result = handler.transcribe_incremental(np.zeros(16000, dtype=np.float32))

    assert model.lengths == []
    assert result.fixed_text == "kept fixed"
    assert result.active_text == ""
    assert handler.fixed_end_time == 1.0


def test_on_session_end_resets_streaming_state():
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.start_language = "en"
    handler.enable_live_transcription = True
    handler.processing_final = True
    reset_calls = []
    handler.streaming_handler = SimpleNamespace(reset=lambda: reset_calls.append(True))

    handler.on_session_end()

    assert handler.processing_final is False
    assert handler.last_language == "en"
    assert reset_calls == [True]
