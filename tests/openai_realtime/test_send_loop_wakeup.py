"""Regression tests for the event-driven realtime send loop (#692).

Covers OutputSignal / NotifyingQueue primitives, producer/wait races, held
output becoming deliverable without another queue write, and teardown while
the consumer is waiting.
"""

from __future__ import annotations

import asyncio
import threading
import time
from queue import Empty, Queue
from threading import Event as ThreadingEvent
from unittest.mock import Mock

import pytest
from starlette.testclient import TestClient

import speech_to_speech.api.openai_realtime.websocket_router as router_module
from speech_to_speech.api.openai_realtime.pipeline_unit import NotifyingQueue, OutputSignal, SessionState
from speech_to_speech.api.openai_realtime.transports import SessionTransport
from speech_to_speech.api.openai_realtime.websocket_router import (
    SEND_LOOP_HOLD_RETRY_S,
    SEND_LOOP_WAIT_TIMEOUT_S,
    _dispatch_client_event,
    _flush_queue,
    _send_loop_wait_timeout,
    _wait_for_pipeline_output,
    create_app,
)
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.configured_runtime import ConfiguredRuntime
from speech_to_speech.pipeline.control import SESSION_END, is_control_message
from speech_to_speech.pipeline.events import SpeechStartedEvent
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END, AudioOutput
from tests.openai_realtime.conftest import make_pipeline_unit


def _pcm_bytes(n_samples: int) -> bytes:
    return b"\x00" * (n_samples * 2)


class RecordingTransport(SessionTransport):
    kind = "test"

    def __init__(self):
        self.audio = asyncio.Queue()
        self.events = asyncio.Queue()

    async def send_events(self, events):
        for event in events:
            self.events.put_nowait(event)

    async def send_audio_chunk(self, service, session_id, pcm, response_key=None):
        self.audio.put_nowait(pcm)

    def discard_pending_audio(self):
        pass

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_output_signal_replays_notify_before_bind_loop():
    signal = OutputSignal()
    signal.notify_threadsafe()
    signal.bind_loop(asyncio.get_running_loop())

    await asyncio.wait_for(signal.wait(), timeout=0.1)
    signal.clear()
    wait_task = asyncio.create_task(signal.wait())
    await asyncio.sleep(0)
    assert not wait_task.done()
    signal.notify_threadsafe()
    await asyncio.wait_for(wait_task, timeout=0.1)


def test_output_signal_rebinds_to_a_new_event_loop():
    """A unit reused by a later server lifespan must wait, not spin."""
    signal = OutputSignal()

    async def wait_then_wake():
        signal.bind_loop(asyncio.get_running_loop())
        signal.clear()
        # The first run binds the event to its loop by blocking on it.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(signal.wait(), timeout=0.01)
        signal.notify_threadsafe()
        await asyncio.wait_for(signal.wait(), timeout=0.5)

    asyncio.run(wait_then_wake())
    asyncio.run(wait_then_wake())


@pytest.mark.asyncio
async def test_output_signal_notify_from_worker_thread_wakes_waiter():
    signal = OutputSignal()
    signal.bind_loop(asyncio.get_running_loop())
    wait_task = asyncio.create_task(signal.wait())
    await asyncio.sleep(0)

    threading.Thread(target=signal.notify_threadsafe, daemon=True).start()
    await asyncio.wait_for(wait_task, timeout=0.5)


@pytest.mark.parametrize("close_during_notify", [False, True])
def test_notifying_queue_put_survives_loop_close(monkeypatch, close_during_notify):
    signal = OutputSignal()
    loop = asyncio.new_event_loop()
    signal.bind_loop(loop)
    queue = NotifyingQueue(signal=signal)
    schedule = loop.call_soon_threadsafe

    if close_during_notify:

        def close_then_schedule(callback):
            loop.close()
            return schedule(callback)

        monkeypatch.setattr(loop, "call_soon_threadsafe", close_then_schedule)
    else:
        loop.close()

    try:
        queue.put(PIPELINE_END)
        assert queue.get_nowait() == PIPELINE_END
        assert queue.empty()
    finally:
        loop.close()


def test_output_signal_preserves_runtime_errors_from_open_loop(monkeypatch):
    signal = OutputSignal()
    loop = asyncio.new_event_loop()
    signal.bind_loop(loop)

    def fail_to_schedule(callback):
        raise RuntimeError("scheduling failed")

    monkeypatch.setattr(loop, "call_soon_threadsafe", fail_to_schedule)
    try:
        with pytest.raises(RuntimeError, match="scheduling failed"):
            signal.notify_threadsafe()
    finally:
        loop.close()


def test_configured_shutdown_allows_worker_cleanup_after_server_loop_closes():
    stop_event = ThreadingEvent()
    ready = ThreadingEvent()
    loop_closed = ThreadingEvent()
    unit = make_pipeline_unit()
    app = create_app(pool=[unit], stop_event=stop_event)

    class LateWorker(BaseHandler):
        def cleanup(self):
            # BaseHandler.run must still enqueue PIPELINE_END after cleanup.
            assert loop_closed.wait(2.0)

    class Server:
        def run(self):
            async def serve():
                async with app.router.lifespan_context(app):
                    # Let the send loop bind its signals before stopping.
                    await asyncio.sleep(0)
                    ready.set()
                    await asyncio.to_thread(stop_event.wait)

            try:
                asyncio.run(serve())
            finally:
                loop_closed.set()

    worker = LateWorker(stop_event, Queue(), unit.output_queue)
    server = Server()
    runtime = ConfiguredRuntime(ThreadingEvent())
    runtime.server = server
    runtime._events = [stop_event]
    runtime._handlers = [worker, server]
    try:
        runtime.start()
        assert ready.wait(2.0)
    finally:
        runtime.stop()

    assert runtime.state == "stopped"
    assert runtime._failure is None
    assert unit.output_queue.get_nowait() == PIPELINE_END


@pytest.mark.asyncio
async def test_notifying_queue_put_wakes_bound_signal():
    signal = OutputSignal()
    signal.bind_loop(asyncio.get_running_loop())
    queue: NotifyingQueue = NotifyingQueue(signal=signal)
    wait_task = asyncio.create_task(signal.wait())
    await asyncio.sleep(0)

    def producer() -> None:
        time.sleep(0.02)
        queue.put("item")

    threading.Thread(target=producer, daemon=True).start()
    await asyncio.wait_for(wait_task, timeout=0.5)
    assert queue.get_nowait() == "item"


@pytest.mark.asyncio
async def test_wait_for_pipeline_output_returns_on_queue_put_without_full_timeout():
    unit = make_pipeline_unit()
    unit.output_signal.bind_loop(asyncio.get_running_loop())
    unit.output_signal.clear()

    wait_task = asyncio.create_task(_wait_for_pipeline_output(unit))
    await asyncio.sleep(0)
    started = time.perf_counter()

    def producer() -> None:
        time.sleep(0.02)
        unit.output_queue.put(object())

    threading.Thread(target=producer, daemon=True).start()
    await asyncio.wait_for(wait_task, timeout=0.5)
    assert time.perf_counter() - started < SEND_LOOP_WAIT_TIMEOUT_S / 2


def test_send_loop_wait_timeout_is_long_when_idle_and_short_when_held():
    unit = make_pipeline_unit()
    assert _send_loop_wait_timeout(unit) == SEND_LOOP_WAIT_TIMEOUT_S

    unit.session = SessionState(session_id="sess", pending_output_item=object())
    assert _send_loop_wait_timeout(unit) == SEND_LOOP_HOLD_RETRY_S

    unit.session = SessionState(session_id="sess", pending_text_output_items=[object()])
    assert _send_loop_wait_timeout(unit) == SEND_LOOP_HOLD_RETRY_S


@pytest.mark.asyncio
async def test_held_audio_releases_after_response_created_without_another_queue_write(monkeypatch):
    """Buffered TTS waiting on response.created is re-checked on the hold retry (#692)."""
    unit = make_pipeline_unit()
    transport = RecordingTransport()
    session_id = unit.service.register()
    unit.session = SessionState(session_id=session_id, transport=transport)
    st = unit.service._state(session_id)
    st.in_response = True
    st.current_response_id = "resp_1"
    st.current_response_key = "rk_1"
    st.response_created_pending_key = "rk_1"
    unit.output_queue.put(AudioOutput(audio=_pcm_bytes(256), response_key="rk_1"))
    # The idle safety net must not be what releases held output.
    monkeypatch.setattr(router_module, "SEND_LOOP_WAIT_TIMEOUT_S", 5.0)
    app = create_app(pool=[unit], stop_event=ThreadingEvent())
    async with app.router.lifespan_context(app):

        async def wait_until_parked():
            while unit.session.pending_output_item is None:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_until_parked(), timeout=0.5)
        # Let any callbacks from the initial queue put drain before releasing.
        for _ in range(10):
            await asyncio.sleep(0)
        unit.service.response.mark_response_created_sent(session_id, "rk_1")
        assert await asyncio.wait_for(transport.audio.get(), timeout=0.5) == _pcm_bytes(256)
        assert st.response_created_pending_key is None


def test_unclaimed_send_loop_waits_instead_of_busy_polling(monkeypatch):
    """An idle/unclaimed unit should block in _wait_for_pipeline_output (#692)."""
    unit = make_pipeline_unit()
    stop_event = ThreadingEvent()
    wait_calls: list[float] = []

    async def tracking_wait(tracked_unit):
        wait_calls.append(time.perf_counter())
        if len(wait_calls) >= 3:
            stop_event.set()
            return
        await asyncio.sleep(0.05)

    monkeypatch.setattr(
        "speech_to_speech.api.openai_realtime.websocket_router._wait_for_pipeline_output",
        tracking_wait,
    )
    app = create_app(pool=[unit], stop_event=stop_event)
    with TestClient(app):
        deadline = time.monotonic() + 2.0
        while not stop_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)

    assert len(wait_calls) >= 3
    gaps = [later - earlier for earlier, later in zip(wait_calls, wait_calls[1:])]
    assert all(gap >= 0.04 for gap in gaps)


def test_producer_put_during_idle_wait_delivers_promptly():
    unit = make_pipeline_unit()
    stop_event = ThreadingEvent()
    app = create_app(pool=[unit], stop_event=stop_event)

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()
            # Let the send loop enter its idle wait, then race a put.
            time.sleep(0.05)
            started = time.perf_counter()
            unit.output_queue.put(_pcm_bytes(256))
            event = ws.receive_json()
            assert time.perf_counter() - started < 0.4
            assert event["type"] == "response.created"

    stop_event.set()


def test_session_release_drains_while_send_loop_was_waiting():
    unit = make_pipeline_unit()
    stop_event = ThreadingEvent()
    app = create_app(pool=[unit], stop_event=stop_event)

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()
            time.sleep(0.05)
        # Route release enqueues SESSION_END; forward it like the handler chain.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                item = unit.input_queue.get(timeout=0.05)
            except Empty:
                continue
            if is_control_message(item, SESSION_END.kind):
                unit.output_queue.put(item)
                break
        else:
            raise AssertionError("SESSION_END did not appear on input_queue")

        deadline = time.monotonic() + 2.0
        while unit.session is not None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert unit.session is None

    stop_event.set()


def test_flush_queue_reinsert_notifies_notifying_queue():
    signal = OutputSignal()
    queue: NotifyingQueue = NotifyingQueue(signal=signal)
    queue.put("keep")
    queue.put("drop")
    # Puts before bind_loop set the missed flag; clear it so only re-insert counts.
    signal._missed = False
    _flush_queue(queue, preserve=lambda item: item == "keep")
    assert signal._missed is True
    assert queue.get_nowait() == "keep"


@pytest.mark.parametrize("method", ["put", "put_nowait"])
def test_queue_insert_notifies_once(method):
    signal = Mock(spec=OutputSignal)
    queue = NotifyingQueue(signal=signal)
    getattr(queue, method)("item")
    signal.notify_threadsafe.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_size", [6400, 6398])
async def test_send_loop_drains_audio_backlog_without_new_notifications(chunk_size):
    unit = make_pipeline_unit()
    transport = RecordingTransport()
    unit.session = SessionState(session_id=unit.service.register(), transport=transport)
    # Smaller chunks force batching to park the following chunk as well.
    chunks = [bytes([index]) * chunk_size for index in range(3)]
    # All puts happen before the loop binds, coalescing into one notification.
    for chunk in chunks:
        unit.output_queue.put(chunk)
    app = create_app(pool=[unit], stop_event=ThreadingEvent())
    async with app.router.lifespan_context(app):

        async def receive_all():
            return [await transport.audio.get() for _ in chunks]

        assert await asyncio.wait_for(receive_all(), timeout=0.5) == chunks


@pytest.mark.asyncio
async def test_send_loop_drains_text_backlog_without_new_notifications():
    unit = make_pipeline_unit()
    transport = RecordingTransport()
    unit.session = SessionState(session_id=unit.service.register(), transport=transport)
    for start_ms in (100, 200, 300):
        unit.text_output_queue.put(SpeechStartedEvent(audio_start_ms=start_ms, interrupt_response=False))
    app = create_app(pool=[unit], stop_event=ThreadingEvent())
    async with app.router.lifespan_context(app):

        async def receive_starts():
            starts = []
            while len(starts) < 3:
                event = await transport.events.get()
                if event.type == "input_audio_buffer.speech_started":
                    starts.append(event.audio_start_ms)
            return starts

        assert await asyncio.wait_for(receive_starts(), timeout=0.5) == [100, 200, 300]


@pytest.mark.asyncio
async def test_held_output_waits_despite_queued_audio(monkeypatch):
    unit = make_pipeline_unit()
    session_id = unit.service.register()
    held = AudioOutput(audio=_pcm_bytes(256), response_key="held")
    unit.session = SessionState(session_id=session_id, pending_output_item=held)
    unit.service._state(session_id).response_created_pending_key = "held"
    unit.output_queue.put(_pcm_bytes(256))
    monkeypatch.setattr(router_module, "SEND_LOOP_HOLD_RETRY_S", 5.0)
    unit.output_signal.bind_loop(asyncio.get_running_loop())
    unit.output_signal.clear()
    waiter = asyncio.create_task(_wait_for_pipeline_output(unit))
    try:
        await asyncio.sleep(0.02)
        assert not waiter.done(), "Queued audio must not bypass the held output barrier"
        # Speech must still get through to cancel a held response.
        unit.text_output_queue.put(SpeechStartedEvent())
        await asyncio.wait_for(waiter, timeout=0.5)
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [SESSION_END, AUDIO_RESPONSE_DONE])
async def test_cancellation_preserves_parked_lifecycle_terminal(terminal):
    unit = make_pipeline_unit()
    transport = RecordingTransport()
    session_id = unit.service.register()
    unit.session = SessionState(session_id=session_id, transport=transport, pending_output_item=terminal)
    st = unit.service._state(session_id)
    st.in_response = True
    st.current_response_id = "resp_1"
    await _dispatch_client_event(unit, session_id, {"type": "response.cancel"}, transport)
    app = create_app(pool=[unit], stop_event=ThreadingEvent())
    async with app.router.lifespan_context(app):

        async def wait_for_terminal():
            while unit.session.pending_output_item is not None:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_terminal(), timeout=0.5)
        if terminal is SESSION_END:
            assert unit.session.drained.is_set()
        else:
            assert not unit.cancel_scope.discarding
