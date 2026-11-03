import asyncio
import json
from threading import Event

from speech_to_speech.zerogpu.readiness import ModelReadinessMiddleware


def frame(kind, event):
    return {"type": kind, "text": json.dumps({"type": event})}


def test_gate_keeps_session_drops_early_speech_and_allows_ready_speech():
    release = Event()
    started = Event()
    calls = []

    def prepare():
        calls.append(1)
        started.set()
        assert release.wait(2)

    async def run():
        sent, received = [], []
        incoming = asyncio.Queue()
        finished = asyncio.Event()

        async def app(scope, receive, send):
            await send({"type": "websocket.accept"})
            await send(frame("websocket.send", "session.created"))
            while True:
                message = await receive()
                received.append(message)
                if message["type"] == "websocket.disconnect":
                    break

        async def send(message):
            sent.append(message)
            if json.loads(message.get("text", "{}")).get("type") == "speech_to_speech.models.ready":
                finished.set()

        task = asyncio.create_task(
            ModelReadinessMiddleware(app, prepare)({"type": "websocket", "path": "/v1/realtime"}, incoming.get, send)
        )
        try:
            while not started.is_set():
                await asyncio.sleep(0.001)
            for event in ["input_audio_buffer.append", "response.create", "session.update"]:
                await incoming.put(frame("websocket.receive", event))
            await asyncio.sleep(0.01)
            assert [json.loads(m["text"])["type"] for m in received] == ["session.update"]
            release.set()
            await asyncio.wait_for(finished.wait(), 2)
            await incoming.put(frame("websocket.receive", "input_audio_buffer.append"))
            await incoming.put({"type": "websocket.disconnect"})
            await task
            assert [json.loads(m["text"])["type"] for m in sent if "text" in m] == [
                "session.created",
                "speech_to_speech.models.preparing",
                "speech_to_speech.models.ready",
            ]
            assert json.loads(received[1]["text"])["type"] == "input_audio_buffer.append"
            assert calls == [1]
        finally:
            release.set()
            task.cancel()

    asyncio.run(run())


def test_preparation_error_is_visible_and_closes_connection():
    async def run():
        sent = []
        closed = asyncio.Event()

        async def app(scope, receive, send):
            await send(frame("websocket.send", "session.created"))
            await closed.wait()

        async def send(message):
            sent.append(message)
            if message["type"] == "websocket.close":
                closed.set()

        def fail():
            raise RuntimeError("GPU warmup failed")

        await ModelReadinessMiddleware(app, fail)({"type": "websocket", "path": "/v1/realtime"}, None, send)
        assert json.loads(sent[-2]["text"]) == {
            "type": "speech_to_speech.models.error",
            "error": {"message": "GPU warmup failed"},
        }
        assert sent[-1]["code"] == 1011

    asyncio.run(run())


def test_busy_rejection_and_other_paths_never_allocate():
    async def run(scope, admitted):
        calls, sent = [], []
        messages = [
            {"type": "websocket.accept"},
            frame("websocket.send", "session.created" if admitted else "error"),
            {"type": "websocket.close", "code": 1008},
        ]

        async def app(scope, receive, send):
            for message in messages:
                await send(message)

        async def send(message):
            sent.append(message)

        await ModelReadinessMiddleware(app, lambda: calls.append(1))(scope, None, send)
        assert sent == messages
        assert calls == []

    for scope, admitted in [
        ({"type": "websocket", "path": "/v1/realtime"}, False),
        ({"type": "websocket", "path": "/gpu"}, True),
        ({"type": "http", "path": "/v1/realtime"}, True),
    ]:
        asyncio.run(run(scope, admitted))


def test_disconnect_during_preparation_never_emits_late_ready():
    started, release = Event(), Event()

    def prepare():
        started.set()
        release.wait(2)

    async def run():
        sent = []

        async def app(scope, receive, send):
            await send(frame("websocket.send", "session.created"))
            assert (await receive())["type"] == "websocket.disconnect"

        async def receive():
            while not started.is_set():
                await asyncio.sleep(0.001)
            return {"type": "websocket.disconnect"}

        async def send(message):
            sent.append(message)

        await ModelReadinessMiddleware(app, prepare)({"type": "websocket", "path": "/v1/realtime"}, receive, send)
        release.set()
        await asyncio.sleep(0.01)
        assert [json.loads(m["text"])["type"] for m in sent] == ["session.created", "speech_to_speech.models.preparing"]

    try:
        asyncio.run(run())
    finally:
        release.set()
