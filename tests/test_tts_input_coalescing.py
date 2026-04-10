from pathlib import Path
from queue import Queue
import sys
from threading import Event

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_control import SESSION_END
from pipeline_messages import MessageTag
from TTS.qwen3_tts_handler import Qwen3TTSHandler


def _make_handler():
    handler = object.__new__(Qwen3TTSHandler)
    handler.queue_in = Queue()
    handler.queue_out = Queue()
    handler.stop_event = Event()
    return handler


def test_coalesce_pending_tts_input_merges_ready_sentences_and_absorbs_response_end():
    handler = _make_handler()

    handler.queue_in.put(("Second sentence.", "en"))
    handler.queue_in.put(("Third sentence.", "en"))
    handler.queue_in.put((MessageTag.END_OF_RESPONSE, None))

    combined, saw_end = handler._coalesce_pending_tts_input(("First sentence.", "en"))

    assert combined == ("First sentence. Second sentence. Third sentence.", "en")
    assert saw_end is True
    assert handler.queue_in.get_nowait() == (MessageTag.END_OF_RESPONSE, None)


def test_coalesce_pending_tts_input_stops_before_control_messages():
    handler = _make_handler()

    handler.queue_in.put(SESSION_END)
    combined, saw_end = handler._coalesce_pending_tts_input(("Hello.", "en"))

    assert combined == ("Hello.", "en")
    assert saw_end is False
    assert handler.queue_in.get_nowait() == SESSION_END
