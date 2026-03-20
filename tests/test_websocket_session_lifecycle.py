import asyncio
from pathlib import Path
from queue import Queue
import sys
from threading import Event, Thread

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseHandler import BaseHandler
from connections.websocket_streamer import WebSocketStreamer
from pipeline_control import SESSION_END, is_control_message


class EchoHandler(BaseHandler):
    def setup(self):
        self.processed = []
        self.session_end_calls = 0

    def process(self, item):
        self.processed.append(item)
        yield item.upper()

    def on_session_end(self):
        self.session_end_calls += 1


class FakeWebSocket:
    def __init__(self, messages):
        self._messages = iter(messages)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._messages)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def test_base_handler_session_end_resets_without_stopping():
    stop_event = Event()
    queue_in = Queue()
    queue_out = Queue()
    handler = EchoHandler(stop_event, queue_in=queue_in, queue_out=queue_out)

    thread = Thread(target=handler.run)
    thread.start()

    queue_in.put(SESSION_END)
    queue_in.put("hello")
    queue_in.put(b"END")

    thread.join(timeout=2)
    assert not thread.is_alive()

    outputs = [queue_out.get(timeout=1) for _ in range(3)]
    assert is_control_message(outputs[0], SESSION_END.kind)
    assert outputs[1] == "HELLO"
    assert outputs[2] == b"END"
    assert handler.processed == ["hello"]
    assert handler.session_end_calls == 1


def test_websocket_streamer_last_disconnect_queues_session_end():
    streamer = WebSocketStreamer(
        stop_event=Event(),
        input_queue=Queue(),
        output_queue=Queue(),
        should_listen=Event(),
    )

    asyncio.run(streamer._handle_client(FakeWebSocket([])))

    queued = streamer.input_queue.get_nowait()
    assert is_control_message(queued, SESSION_END.kind)
    assert streamer.should_listen.is_set()


def test_websocket_send_loop_ignores_session_end_until_stop():
    stop_event = Event()
    streamer = WebSocketStreamer(
        stop_event=stop_event,
        input_queue=Queue(),
        output_queue=Queue(),
        should_listen=Event(),
    )

    async def exercise_send_loop():
        task = asyncio.create_task(streamer._send_loop())
        streamer.output_queue.put(SESSION_END)
        await asyncio.sleep(0.05)
        assert not task.done()
        stop_event.set()
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(exercise_send_loop())
