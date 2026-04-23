from queue import Queue
from threading import Event

from speech_to_speech.pipeline.control import SESSION_END
from speech_to_speech.pipeline.messages import EndOfResponse, TTSInput
from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler


def _make_handler():
    handler = object.__new__(Qwen3TTSHandler)
    handler.queue_in = Queue()
    handler.queue_out = Queue()
    handler.stop_event = Event()
    return handler


def test_coalesce_pending_tts_input_merges_ready_sentences_and_absorbs_response_end():
    handler = _make_handler()

    handler.queue_in.put(TTSInput(text="Second sentence.", language_code="en"))
    handler.queue_in.put(TTSInput(text="Third sentence.", language_code="en"))
    handler.queue_in.put(EndOfResponse())

    text, lang, saw_end = handler._coalesce_pending_tts_input(TTSInput(text="First sentence.", language_code="en"))

    assert text == "First sentence. Second sentence. Third sentence."
    assert lang == "en"
    assert saw_end is True
    remaining = handler.queue_in.get_nowait()
    assert isinstance(remaining, EndOfResponse)


def test_coalesce_pending_tts_input_stops_before_control_messages():
    handler = _make_handler()

    handler.queue_in.put(SESSION_END)
    text, lang, saw_end = handler._coalesce_pending_tts_input(TTSInput(text="Hello.", language_code="en"))

    assert text == "Hello."
    assert lang == "en"
    assert saw_end is False
    assert handler.queue_in.get_nowait() == SESSION_END
