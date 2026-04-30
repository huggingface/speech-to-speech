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


def test_final_transcription_prevents_stale_fixed_window_on_next_progressive(monkeypatch):
    class Model:
        def __init__(self):
            self.progressive_window_lengths = []

        def transcribe(self, audio, timestamps=True):
            self.progressive_window_lengths.append(len(audio))
            return SimpleNamespace(text="new partial", timestamp={"segment": []})

    model = Model()
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.enable_live_transcription = True
    handler.backend = "nano_parakeet"
    handler.last_language = "en"
    handler.start_language = None
    handler.processing_final = False
    handler.streaming_handler = SmartProgressiveStreamingHandler(model)
    handler.streaming_handler.fixed_sentences = ["previous fixed sentence"]
    handler.streaming_handler.fixed_end_time = 10.0
    handler.streaming_handler.last_transcribed_length = 20 * 16000

    @contextmanager
    def fake_lock(*args, **kwargs):
        yield True

    handler._compute_lock_context = fake_lock
    handler._process_nano_parakeet = lambda audio_input: ("previous final", "en")
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: None)

    final_result = list(handler.process(VADAudio(audio=np.zeros(16000, dtype=np.float32), mode="final")))
    progressive_audio = np.zeros(852 * 16, dtype=np.float32)
    progressive_result = list(handler.process(VADAudio(audio=progressive_audio, mode="progressive")))

    assert len(final_result) == 1
    assert isinstance(final_result[0], Transcription)
    assert model.progressive_window_lengths == [len(progressive_audio)]
    assert len(progressive_result) == 1
    assert isinstance(progressive_result[0], PartialTranscription)
    assert progressive_result[0].text == "new partial"


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
