import io
import logging
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from openai.types.realtime.session_update_event import SessionUpdateEvent
from rich.text import Text

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
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


def test_live_transcription_clears_terminal_line_before_each_update(monkeypatch):
    calls = []

    class FakeConsole:
        is_terminal = True
        width = 80

        def __init__(self):
            self.file = io.StringIO()

        def print(self, *args, **kwargs):
            calls.append((args, kwargs))

    handler = object.__new__(ParakeetTDTSTTHandler)
    handler._live_transcription_active = False
    fake_console = FakeConsole()
    monkeypatch.setattr(parakeet_tdt_handler, "console", fake_console)

    handler._print_live_transcription(Text("Live: first"), "first")
    handler._print_live_transcription(Text("Live: second"), "second")
    handler._clear_live_transcription_line()

    assert [args[0].plain for args, _ in calls] == ["Live: first", "Live: second"]
    assert [kwargs for _, kwargs in calls] == [{"end": ""}, {"end": ""}]
    assert fake_console.file.getvalue() == "\r\x1b[2K\r\r\x1b[2K\r\r\x1b[2K"
    assert handler._live_transcription_active is False


def test_live_transcription_truncates_terminal_updates(monkeypatch):
    calls = []

    class FakeConsole:
        is_terminal = True
        width = 14

        def __init__(self):
            self.file = io.StringIO()

        def print(self, *args, **kwargs):
            calls.append((args, kwargs))

    handler = object.__new__(ParakeetTDTSTTHandler)
    handler._live_transcription_active = False
    monkeypatch.setattr(parakeet_tdt_handler, "console", FakeConsole())

    handler._print_live_transcription(Text("Live: abcdefghijklmnopqrstuvwxyz"), "abcdefghijklmnopqrstuvwxyz")

    printed_text = calls[0][0][0]
    assert printed_text.plain == "Live: abcdef\u2026"
    assert len(printed_text.plain) == 13


def test_live_transcription_uses_lines_for_non_terminal_logs(monkeypatch):
    calls = []

    class FakeConsole:
        is_terminal = False

        def print(self, *args, **kwargs):
            calls.append((args, kwargs))

    handler = object.__new__(ParakeetTDTSTTHandler)
    handler._live_transcription_active = False
    monkeypatch.setattr(parakeet_tdt_handler, "console", FakeConsole())

    handler._print_live_transcription(Text("Live: first"), "first")

    assert calls == [((Text("Live: first"),), {})]
    assert handler._live_transcription_active is False


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
    handler._process_nano_parakeet = lambda audio_input: "I am here."
    handler._detect_language_from_text = lambda text: "en"
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: None)

    result = list(handler.process(VADAudio(audio=np.zeros(16000, dtype=np.float32))))

    assert len(result) == 1
    assert isinstance(result[0], Transcription)
    assert result[0].text == "I am here."
    assert result[0].language_code == "en-auto"


@pytest.mark.parametrize("backend", ["nano_parakeet", "mlx"])
@pytest.mark.parametrize("detected_language", ["en", None])
def test_configured_language_does_not_relabel_decoded_text(monkeypatch, backend, detected_language):
    calls = []
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.enable_live_transcription = False
    handler.backend = backend
    handler.streaming_handler = None
    handler.start_language = "es"
    handler.last_language = "es"
    handler._detect_language_from_text = lambda text: detected_language

    @contextmanager
    def fake_lock(*args, **kwargs):
        yield True

    handler._compute_lock_context = fake_lock
    text = "This is a complete English sentence."
    if backend == "nano_parakeet":

        def transcribe(audio):
            calls.append(("transcribe", audio))
            return text

        handler.model = SimpleNamespace(transcribe=transcribe)
    else:
        fake_mx = SimpleNamespace(array=lambda audio, dtype: audio, float32="float32")
        monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

        def decode_chunk(audio, verbose=False):
            calls.append(("decode_chunk", audio, verbose))
            return SimpleNamespace(text=text)

        handler.model = SimpleNamespace(decode_chunk=decode_chunk)
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: None)

    result = list(handler.process(VADAudio(audio=np.zeros(16000, dtype=np.float32), mode="final")))

    assert len(calls) == 1
    assert calls[0][0] == ("transcribe" if backend == "nano_parakeet" else "decode_chunk")
    assert result[0].text == text
    assert result[0].language_code == (f"{detected_language}-auto" if detected_language else None)


def test_mlx_live_final_detects_combined_text_without_using_configured_language(monkeypatch):
    calls = []
    fake_mx = SimpleNamespace(array=lambda audio, dtype: audio, float32="float32")
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.backend = "mlx"
    handler.enable_live_transcription = True
    handler.processing_final = False
    handler._live_turn_key = ("turn_1", 0)
    handler.start_language = "es"
    handler.last_language = None
    handler.streaming_handler = SimpleNamespace(fixed_sentences=["This is"], fixed_end_time=0.5, reset=lambda: None)
    handler._detect_language_from_text = lambda text: "en"

    @contextmanager
    def fake_lock(*args, **kwargs):
        yield True

    handler._compute_lock_context = fake_lock

    def decode_chunk(audio, verbose=False):
        calls.append((len(audio), verbose))
        return SimpleNamespace(text="a complete English sentence.")

    handler.model = SimpleNamespace(decode_chunk=decode_chunk)
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: None)

    result = list(
        handler.process(
            VADAudio(audio=np.zeros(16000, dtype=np.float32), mode="final", turn_id="turn_1", turn_revision=0)
        )
    )

    assert calls == [(8000, False)]
    assert result[0].text == "This is a complete English sentence."
    assert result[0].language_code == "en-auto"


@pytest.mark.parametrize("backend", ["nano_parakeet", "mlx"])
@pytest.mark.parametrize(
    ("selected_language", "detected_language", "expected_language"),
    [(None, "en", "en"), ("auto", "en", "en"), ("auto", None, None)],
)
def test_session_auto_reset_does_not_report_setup_language(
    monkeypatch, backend, selected_language, detected_language, expected_language
):
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.enable_live_transcription = False
    handler.backend = backend
    handler.start_language = "fr"
    handler.last_language = None

    @contextmanager
    def fake_lock(*args, **kwargs):
        yield True

    handler._compute_lock_context = fake_lock
    handler._process_nano_parakeet = lambda audio: "This is a complete English sentence."
    handler._process_mlx_final = lambda audio: "This is a complete English sentence."
    handler._detect_language_from_text = lambda text: detected_language
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: None)

    runtime_config = RuntimeConfig()
    if selected_language is not None:
        update = SessionUpdateEvent.model_validate(
            {
                "type": "session.update",
                "session": {"type": "realtime", "audio": {"input": {"transcription": {"language": selected_language}}}},
            }
        )
        runtime_config.apply_session_update(update.session)

    result = list(
        handler.process(VADAudio(audio=np.zeros(16000, dtype=np.float32), runtime_config=runtime_config, mode="final"))
    )

    assert result[0].language_code == (f"{expected_language}-auto" if expected_language else None)
    assert handler.last_language == expected_language


def test_session_auto_reset_keeps_transcript_when_language_detection_fails(monkeypatch):
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.enable_live_transcription = False
    handler.backend = "nano_parakeet"
    handler.start_language = "fr"
    handler.last_language = None

    @contextmanager
    def fake_lock(*args, **kwargs):
        yield True

    handler._compute_lock_context = fake_lock
    handler._process_nano_parakeet = lambda audio: "This is a complete English sentence."

    def failed_detection(text):
        raise RuntimeError("language detector unavailable")

    handler._detect_language_from_text = failed_detection
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: None)

    runtime_config = RuntimeConfig()
    update = SessionUpdateEvent.model_validate(
        {
            "type": "session.update",
            "session": {"type": "realtime", "audio": {"input": {"transcription": {"language": "auto"}}}},
        }
    )
    runtime_config.apply_session_update(update.session)

    result = list(
        handler.process(VADAudio(audio=np.zeros(16000, dtype=np.float32), runtime_config=runtime_config, mode="final"))
    )

    assert result[0].text == "This is a complete English sentence."
    assert result[0].language_code is None
    assert handler.last_language is None


def test_parakeet_timing_logs_only_final_transcriptions():
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler._times = [0.01]

    assert handler.timing_log_level == logging.INFO
    assert handler.should_log_timing(Transcription(text="I am here.", language_code="en"))
    assert not handler.should_log_timing(PartialTranscription(text="I am"))


def test_final_transcription_resets_live_streaming_state(monkeypatch):
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.enable_live_transcription = True
    handler.backend = "nano_parakeet"
    handler.last_language = "en"
    handler.start_language = None
    handler.processing_final = False
    handler._live_turn_key = (None, None)
    reset_calls = []
    handler.streaming_handler = SimpleNamespace(reset=lambda: reset_calls.append(True))

    @contextmanager
    def fake_lock(*args, **kwargs):
        yield True

    handler._compute_lock_context = fake_lock
    handler._process_nano_parakeet = lambda audio_input: "I am here."
    handler._detect_language_from_text = lambda text: "en"
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: None)

    result = list(handler.process(VADAudio(audio=np.zeros(16000, dtype=np.float32), mode="final")))

    assert len(result) == 1
    assert isinstance(result[0], Transcription)
    assert handler.processing_final is False
    assert reset_calls == [True]


def test_turn_change_resets_live_streaming_state_before_progressive(monkeypatch):
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.enable_live_transcription = True
    handler.processing_final = False
    handler._live_turn_key = ("turn_1", 0)
    reset_calls = []
    handler.streaming_handler = SimpleNamespace(reset=lambda: reset_calls.append(True))

    @contextmanager
    def fake_lock(*args, **kwargs):
        yield True

    handler._compute_lock_context = fake_lock
    handler._show_progressive_transcription = lambda audio: "new partial"
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: None)

    result = list(
        handler.process(
            VADAudio(
                audio=np.zeros(16000, dtype=np.float32),
                mode="progressive",
                turn_id="turn_2",
                turn_revision=0,
            )
        )
    )

    assert reset_calls == [True]
    assert len(result) == 1
    assert isinstance(result[0], PartialTranscription)
    assert result[0].text == "new partial"


def test_mlx_final_ignores_fixed_text_that_exceeds_current_audio(monkeypatch):
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.enable_live_transcription = True
    handler.backend = "mlx"
    handler.last_language = "en"
    handler.start_language = None
    handler.processing_final = False
    handler._live_turn_key = ("turn_3", 0)
    handler.streaming_handler = SimpleNamespace(
        fixed_sentences=["stale previous transcript"],
        fixed_end_time=10.0,
        reset=lambda: None,
    )

    @contextmanager
    def fake_lock(*args, **kwargs):
        yield True

    handler._compute_lock_context = fake_lock
    handler._process_mlx = lambda audio_input: "new short turn"
    handler._detect_language_from_text = lambda text: "en"
    monkeypatch.setattr(parakeet_tdt_handler.console, "print", lambda *args, **kwargs: None)

    result = list(
        handler.process(
            VADAudio(
                audio=np.zeros(16000, dtype=np.float32),
                mode="final",
                turn_id="turn_3",
                turn_revision=0,
            )
        )
    )

    assert len(result) == 1
    assert isinstance(result[0], Transcription)
    assert result[0].text == "new short turn"


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
    handler._process_nano_parakeet = lambda audio_input: "previous final"
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
    assert handler.last_language is None
    assert reset_calls == [True]
