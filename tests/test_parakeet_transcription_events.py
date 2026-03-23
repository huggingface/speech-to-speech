from contextlib import contextmanager
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from STT import parakeet_tdt_handler
from STT.parakeet_tdt_handler import ParakeetTDTSTTHandler


def test_emit_user_transcript_keeps_repeated_partial_updates():
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.text_output_queue = Queue()

    handler._emit_user_transcript("Hello there", is_final=False)
    handler._emit_user_transcript("Hello there", is_final=False)
    handler._emit_user_transcript("Hello there again", is_final=False)

    messages = [
        handler.text_output_queue.get_nowait(),
        handler.text_output_queue.get_nowait(),
        handler.text_output_queue.get_nowait(),
    ]

    assert messages == [
        {"type": "user_text", "text": "Hello there", "is_final": False},
        {"type": "user_text", "text": "Hello there", "is_final": False},
        {"type": "user_text", "text": "Hello there again", "is_final": False},
    ]


def test_show_progressive_transcription_enqueues_combined_text(monkeypatch):
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.text_output_queue = Queue()
    handler.streaming_handler = SimpleNamespace(
        transcribe_incremental=lambda audio: SimpleNamespace(
            fixed_text="I just wanted",
            active_text="to check in",
        )
    )
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: None)

    handler._show_progressive_transcription(np.zeros(16000, dtype=np.float32))

    assert handler.text_output_queue.get_nowait() == {
        "type": "user_text",
        "text": "I just wanted to check in",
        "is_final": False,
    }


def test_process_emits_final_transcript_event_with_language(monkeypatch):
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.enable_live_transcription = False
    handler.backend = "nano_parakeet"
    handler.last_language = "en"
    handler.start_language = None
    handler.text_output_queue = Queue()

    @contextmanager
    def fake_lock(*args, **kwargs):
        yield True

    handler._compute_lock_context = fake_lock
    handler._process_nano_parakeet = lambda audio_input: ("I am here.", "en")
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: None)

    result = list(handler.process(np.zeros(16000, dtype=np.float32)))

    assert result == [("I am here.", "en")]
    assert handler.text_output_queue.get_nowait() == {
        "type": "user_text",
        "text": "I am here.",
        "is_final": True,
        "language": "en",
    }


def test_on_session_end_resets_streaming_state():
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.enable_live_transcription = True
    handler.processing_final = True
    reset_calls = []
    handler.streaming_handler = SimpleNamespace(reset=lambda: reset_calls.append(True))

    handler.on_session_end()

    assert handler.processing_final is False
    assert reset_calls == [True]
