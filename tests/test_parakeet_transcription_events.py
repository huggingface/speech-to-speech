from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from STT import parakeet_tdt_handler
from STT.parakeet_tdt_handler import ParakeetTDTSTTHandler


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

    result = list(handler.process(("progressive", np.zeros(16000, dtype=np.float32))))

    assert result == [("__PARTIAL__", "partial text")]


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

    result = list(handler.process(np.zeros(16000, dtype=np.float32)))

    assert result == [("I am here.", "en")]


def test_on_session_end_resets_streaming_state():
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.enable_live_transcription = True
    handler.processing_final = True
    reset_calls = []
    handler.streaming_handler = SimpleNamespace(reset=lambda: reset_calls.append(True))

    handler.on_session_end()

    assert handler.processing_final is False
    assert reset_calls == [True]
