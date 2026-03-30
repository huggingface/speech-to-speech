from pathlib import Path
from queue import Queue
import sys
from threading import Event

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseHandler import BaseHandler
from pipeline_control import SESSION_END


class ProbeHandler(BaseHandler):
    def setup(self):
        pass

    def process(self, _item):
        yield _item


def test_coalesce_pending_tts_input_merges_ready_sentences_and_absorbs_response_end():
    handler = ProbeHandler(Event(), queue_in=Queue(), queue_out=Queue())

    handler.queue_in.put(("Second sentence.", "en"))
    handler.queue_in.put(("Third sentence.", "en"))
    handler.queue_in.put(("__END_OF_RESPONSE__", None))

    combined, saw_end = handler._coalesce_pending_tts_input(("First sentence.", "en"))

    assert combined == ("First sentence. Second sentence. Third sentence.", "en")
    assert saw_end is True
    assert handler.queue_in.get_nowait() == ("__END_OF_RESPONSE__", None)


def test_coalesce_pending_tts_input_stops_before_control_messages():
    handler = ProbeHandler(Event(), queue_in=Queue(), queue_out=Queue())

    handler.queue_in.put(SESSION_END)
    combined, saw_end = handler._coalesce_pending_tts_input(("Hello.", "en"))

    assert combined == ("Hello.", "en")
    assert saw_end is False
    assert handler.queue_in.get_nowait() == SESSION_END
