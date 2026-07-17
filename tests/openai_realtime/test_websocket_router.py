"""Integration tests for api.openai_realtime.websocket_router.

Uses Starlette's synchronous TestClient with WebSocket support to exercise
the full FastAPI app produced by ``create_app``. Each test gets a fresh
PipelineUnit pool (size 1, matching the single-session semantics of the
old tests) so there is no cross-test state.
"""

import asyncio
import base64
import time
from queue import Empty, Queue
from threading import Event as ThreadingEvent

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketState

import speech_to_speech.api.openai_realtime.websocket_router as router_module
from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
from speech_to_speech.api.openai_realtime.service import CHUNK_SIZE_BYTES, RealtimeService
from speech_to_speech.api.openai_realtime.transports import WebSocketTransport
from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.control import SESSION_END, PipelineControlMessage, is_control_message
from speech_to_speech.pipeline.events import AssistantTextEvent, SpeechStartedEvent, TokenUsageEvent
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END, AudioOutput

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def short_drain_timeout(monkeypatch):
    """Shorten the SESSION_END drain warning threshold so tests don't wait 10s.

    The constant only controls when the release task logs a warning about a
    slow-draining unit. The quarantine timeout
    (SESSION_END_QUARANTINE_TIMEOUT_S) is left at its real value so units
    stay unavailable until SESSION_END actually drains; tests that exercise the
    quarantine shorten it themselves.
    """
    monkeypatch.setattr(router_module, "SESSION_END_DRAIN_TIMEOUT_S", 0.1)


@pytest.fixture
def setup():
    """Return (app, service, input_queue, output_queue, text_output_queue,
    should_listen, stop_event, response_playing, cancel_scope) for a pool of one.

    There is no real handler chain in this fixture, so SESSION_END enqueued by
    the route handler on disconnect never reaches output_queue. Tests that need
    the release task to complete (verifying unit.session is cleared and the
    service unregistered) must drain SESSION_END themselves — see
    `_simulate_session_end_drain` below.
    """
    text_prompt_queue: Queue = Queue()
    should_listen = ThreadingEvent()
    should_listen.set()
    service = RealtimeService(
        text_prompt_queue=text_prompt_queue,
        should_listen=should_listen,
    )
    input_queue: Queue = Queue()
    output_queue: Queue = Queue()
    text_output_queue: Queue = Queue()
    stop_event = ThreadingEvent()
    response_playing = ThreadingEvent()
    cancel_scope = CancelScope()
    unit = PipelineUnit(
        index=0,
        service=service,
        cancel_scope=cancel_scope,
        should_listen=should_listen,
        response_playing=response_playing,
        input_queue=input_queue,
        output_queue=output_queue,
        text_output_queue=text_output_queue,
        text_prompt_queue=text_prompt_queue,
        handlers=[],
    )
    app = create_app(pool=[unit], stop_event=stop_event)
    return (
        app,
        service,
        input_queue,
        output_queue,
        text_output_queue,
        should_listen,
        stop_event,
        response_playing,
        cancel_scope,
    )


def _simulate_session_end_drain(input_queue: Queue, output_queue: Queue, timeout: float = 1.0) -> None:
    """Wait for SESSION_END to land in input_queue (from the route handler's
    release path) and forward it to output_queue — simulating the handler chain.
    The send loop will then observe SESSION_END and set `session.drained`,
    letting the release task complete (unregister + clear `unit.session`).
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            item = input_queue.get(timeout=0.05)
        except Empty:
            continue
        if isinstance(item, PipelineControlMessage) and is_control_message(item, SESSION_END.kind):
            output_queue.put(item)
            return
    raise AssertionError("SESSION_END did not appear on input_queue within timeout")


def _pcm_bytes(n_samples: int) -> bytes:
    return b"\x00" * (n_samples * 2)


class _FakeWebSocket:
    application_state = WebSocketState.CONNECTED

    def __init__(self):
        self.sent: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


# ===================================================================
# Connection
# ===================================================================


class TestConnection:
    def test_connect_receives_session_created(self, setup):
        app, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                msg = ws.receive_json()
                assert msg["type"] == "session.created"
                assert msg["event_id"].startswith("event_")
                assert "session" in msg

    def test_second_connection_rejected(self, setup):
        app, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws1:
                ws1.receive_json()  # session.created
                with client.websocket_connect("/v1/realtime") as ws2:
                    msg = ws2.receive_json()
                    assert msg["type"] == "error"
                    # Rejection uses the stateless build_error_event helper —
                    # the error type identifies pool exhaustion specifically.
                    assert msg["error"]["type"] == "session_limit_reached"


# ===================================================================
# Client event dispatch
# ===================================================================


class TestClientEventDispatch:
    def test_audio_append_forwarded_to_input_queue(self, setup):
        app, _, input_queue, *_ = setup
        audio_b64 = base64.b64encode(_pcm_bytes(512)).decode("ascii")
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                ws.send_json(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64,
                    }
                )
                time.sleep(0.1)
                item = input_queue.get(timeout=1)
                assert isinstance(item, tuple) and len(item) == 2
                chunk, rt_cfg = item
                assert isinstance(chunk, bytes)
                assert len(chunk) == CHUNK_SIZE_BYTES

    def test_session_update_applied(self, setup):
        app, service, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                ws.send_json(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "audio": {"output": {"voice": "coral"}},
                        },
                    }
                )
                time.sleep(0.1)
                cid = service.connection_ids[0]
                assert service._state(cid).runtime_config.session.audio.output.voice == "coral"

    def test_conversation_item_create_returns_events(self, setup):
        app, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                ws.send_json(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "ping"}],
                        },
                    }
                )
                msg = ws.receive_json()
                assert msg["type"] == "conversation.item.created"
                assert msg["item"]["content"][0]["text"] == "ping"

    def test_response_create_error_when_active(self, setup):
        app, service, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                conn_id = list(service._conns.keys())[0]
                service.response._ensure_response(conn_id)
                ws.send_json({"type": "response.create"})
                msg = ws.receive_json()
                assert msg["type"] == "error"
                assert "another response is in progress" in msg["error"]["message"].lower()

    def test_response_cancel_returns_events(self, setup):
        app, service, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                conn_id = list(service._conns.keys())[0]
                service.response._ensure_response(conn_id)
                ws.send_json({"type": "response.cancel"})
                msg1 = ws.receive_json()
                msg2 = ws.receive_json()
                types = {msg1["type"], msg2["type"]}
                assert "response.output_audio.done" in types
                assert "response.done" in types

    def test_response_cancel_flushes_queues(self, setup):
        app, service, _, output_queue, text_output_queue, _, _, response_playing, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                conn_id = list(service._conns.keys())[0]
                service.response._ensure_response(conn_id)
                response_playing.set()
                output_queue.put(_pcm_bytes(256))
                output_queue.put(_pcm_bytes(256))
                text_output_queue.put(AssistantTextEvent(text="stale"))
                ws.send_json({"type": "response.cancel"})
                ws.receive_json()  # response.output_audio.done
                ws.receive_json()  # response.done
                time.sleep(0.1)
                assert output_queue.empty()
                assert text_output_queue.empty()
                assert not response_playing.is_set()
                assert cancel_scope.discarding

    def test_response_cancel_spurious_does_not_set_discarding(self, setup):
        """response.cancel when no response is active must NOT enable discarding,
        otherwise it would stick True forever (no __RESPONSE_DONE__ to clear it)."""
        app, service, _, _, _, _, _, _, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                assert not service._state(list(service._conns.keys())[0]).in_response
                ws.send_json({"type": "response.cancel"})
                time.sleep(0.1)
                assert not cancel_scope.discarding

    def test_response_cancel_late_audio_is_discarded(self, setup):
        """Audio arriving after response.cancel is silently dropped (discard guard)."""
        app, service, _, output_queue, _, _, _, response_playing, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                conn_id = list(service._conns.keys())[0]
                service.response._ensure_response(conn_id)
                response_playing.set()
                ws.send_json({"type": "response.cancel"})
                ws.receive_json()  # response.output_audio.done
                ws.receive_json()  # response.done
                time.sleep(0.1)
                assert cancel_scope.discarding
                output_queue.put(_pcm_bytes(256))
                time.sleep(0.15)
                # No response.created or audio delta should appear; only
                # __RESPONSE_DONE__ will eventually clear the guard.
                output_queue.put(AUDIO_RESPONSE_DONE)
                time.sleep(0.15)
                assert not cancel_scope.discarding

    def test_unknown_event_returns_error(self, setup):
        app, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                ws.send_json({"type": "bogus.event"})
                msg = ws.receive_json()
                assert msg["type"] == "error"


# ===================================================================
# Send loop (pipeline -> client)
# ===================================================================


class TestSendLoop:
    def test_audio_output_ignores_session_end_control_message(self, setup):
        app, _, _, output_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                output_queue.put(SESSION_END)
                output_queue.put(_pcm_bytes(256))

                msg1 = ws.receive_json()
                assert msg1["type"] == "response.created"
                msg2 = ws.receive_json()
                assert msg2["type"] == "response.output_audio.delta"

    def test_audio_output_sends_response_created_and_delta(self, setup):
        app, _, _, output_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                output_queue.put(_pcm_bytes(256))
                msg1 = ws.receive_json()
                assert msg1["type"] == "response.created"
                assert msg1["response"]["status"] == "in_progress"
                msg2 = ws.receive_json()
                assert msg2["type"] == "response.output_audio.delta"
                assert "delta" in msg2

    def test_audio_output_batches_immediately_available_chunks(self, setup):
        app, _, _, output_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                output_queue.put(_pcm_bytes(256))
                output_queue.put(_pcm_bytes(256))
                output_queue.put(PIPELINE_END)

                msg1 = ws.receive_json()
                assert msg1["type"] == "response.created"

                msg2 = ws.receive_json()
                assert msg2["type"] == "response.output_audio.delta"
                decoded = base64.b64decode(msg2["delta"])
                assert len(decoded) == len(_pcm_bytes(512))

                msg3 = ws.receive_json()
                msg4 = ws.receive_json()
                types = {msg3["type"], msg4["type"]}
                assert "response.output_audio.done" in types
                assert "response.done" in types

    def test_end_marker_sends_finish_events(self, setup):
        app, _, _, output_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                output_queue.put(_pcm_bytes(256))
                ws.receive_json()  # response.created
                ws.receive_json()  # audio delta
                output_queue.put(PIPELINE_END)
                msg1 = ws.receive_json()
                msg2 = ws.receive_json()
                types = {msg1["type"], msg2["type"]}
                assert "response.output_audio.done" in types
                assert "response.done" in types

    def test_text_output_sends_pipeline_events(self, setup):
        app, _, _, _, text_output_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                text_output_queue.put(SpeechStartedEvent())
                msg = ws.receive_json()
                assert msg["type"] == "input_audio_buffer.speech_started"
                assert msg["audio_start_ms"] == 0

    def test_barge_in_discard_clears_after_response_done(self, setup):
        """After barge-in sets discarding=True, __RESPONSE_DONE__ must clear it back to False."""
        app, service, _, output_queue, text_output_queue, _, _, response_playing, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                conn_id = list(service._conns.keys())[0]
                service.response._ensure_response(conn_id)
                response_playing.set()
                # Trigger barge-in
                text_output_queue.put(SpeechStartedEvent())
                ws.receive_json()  # input_audio_buffer.speech_started
                ws.receive_json()  # response.output_audio.done
                ws.receive_json()  # response.done
                time.sleep(0.1)
                assert cancel_scope.discarding
                output_queue.put(AUDIO_RESPONSE_DONE)
                time.sleep(0.15)
                assert not cancel_scope.discarding

    def test_speech_started_cancels_pending_implicit_response(self, setup):
        app, service, _, output_queue, text_output_queue, _, _, response_playing, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                conn_id = list(service._conns.keys())[0]
                stale_generation = cancel_scope.generation
                service._state(conn_id).response_pending = True

                text_output_queue.put(SpeechStartedEvent())
                msg = ws.receive_json()

                assert msg["type"] == "input_audio_buffer.speech_started"
                time.sleep(0.15)
                assert cancel_scope.discarding
                assert cancel_scope.generation == stale_generation + 1
                assert service._state(conn_id).response_pending is False
                assert service._state(conn_id).in_response is False
                assert not response_playing.is_set()

                output_queue.put(AudioOutput(audio=AUDIO_RESPONSE_DONE, cancel_generation=stale_generation))
                time.sleep(0.15)
                assert not cancel_scope.discarding

    def test_speech_started_does_not_cancel_pending_when_internal_non_interrupt(self, setup):
        app, service, _, _, text_output_queue, _, _, _, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                conn_id = list(service._conns.keys())[0]
                service._state(conn_id).response_pending = True

                text_output_queue.put(SpeechStartedEvent(interrupt_response=False))
                msg = ws.receive_json()

                assert msg["type"] == "input_audio_buffer.speech_started"
                time.sleep(0.15)
                assert not cancel_scope.discarding
                assert service._state(conn_id).response_pending is True

    def test_stale_tagged_audio_is_dropped_after_interruption(self, setup):
        app, _, _, output_queue, _, _, _, _, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                stale_generation = cancel_scope.generation
                cancel_scope.cancel()
                current_generation = cancel_scope.generation
                output_queue.put(AudioOutput(audio=_pcm_bytes(64), cancel_generation=stale_generation))
                output_queue.put(AudioOutput(audio=_pcm_bytes(512), cancel_generation=current_generation))

                assert ws.receive_json()["type"] == "response.created"
                delta = ws.receive_json()

                assert delta["type"] == "response.output_audio.delta"
                assert len(base64.b64decode(delta["delta"])) == len(_pcm_bytes(512))

    def test_current_generation_text_survives_stuck_discarding(self, setup):
        """Regression: a fresh response's transcript must survive a stuck discard guard.

        A superseded speculative turn can leave ``cancel_scope.discarding`` stuck True
        (its TTS dropped the stale ``EndOfResponse`` without emitting AUDIO_RESPONSE_DONE,
        so ``response_done()`` never cleared the flag). The next response's audio is tagged
        with the current generation and streams fine, but the assistant text used to be
        blanket-dropped while discarding — leaving audio + ``response.done`` with no
        ``response.output_audio_transcript.done``. The text is now discarded by the same
        generation-aware rule as audio, so a current-generation transcript is kept.
        """
        app, _, _, output_queue, text_output_queue, _, _, _, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                cancel_scope.cancel()  # discarding=True, generation bumped; sentinel never arrived
                current_generation = cancel_scope.generation
                assert cancel_scope.discarding

                text_output_queue.put(AssistantTextEvent(text="hello there", cancel_generation=current_generation))
                output_queue.put(AudioOutput(audio=_pcm_bytes(256), cancel_generation=current_generation))
                output_queue.put(AudioOutput(audio=AUDIO_RESPONSE_DONE, cancel_generation=current_generation))

                types: list[str] = []
                transcript = None
                for _ in range(8):
                    msg = ws.receive_json()
                    types.append(msg["type"])
                    if msg["type"] == "response.output_audio_transcript.done":
                        transcript = msg["transcript"]
                    if msg["type"] == "response.done":
                        break
                assert "response.output_audio_transcript.done" in types
                assert transcript == "hello there"

    def test_stale_tagged_response_done_does_not_finish_current_response(self, setup):
        app, service, _, output_queue, _, _, _, _, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                conn_id = list(service._conns.keys())[0]
                stale_generation = cancel_scope.generation
                service.response._ensure_response(conn_id)
                service.finish_response(conn_id, status="cancelled")
                cancel_scope.cancel()
                current_response_id, _ = service.response._ensure_response(conn_id)

                output_queue.put(AudioOutput(audio=AUDIO_RESPONSE_DONE, cancel_generation=stale_generation))
                time.sleep(0.15)

                state = service._state(conn_id)
                assert state.in_response
                assert state.current_response_id == current_response_id

    def test_response_done_drains_pending_token_usage_before_finish(self, setup):
        app, service, _, output_queue, text_output_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                conn_id = list(service._conns.keys())[0]

                text_output_queue.put(
                    AssistantTextEvent(
                        text="",
                        tools=[{"type": "function_call", "call_id": "c1", "name": "f1", "arguments": "{}"}],
                    )
                )
                text_output_queue.put(TokenUsageEvent(input_tokens=10, output_tokens=5))
                output_queue.put(AUDIO_RESPONSE_DONE)

                assert ws.receive_json()["type"] == "response.function_call_arguments.done"
                msg1 = ws.receive_json()
                msg2 = ws.receive_json()
                assert {msg1["type"], msg2["type"]} == {"response.output_audio.done", "response.done"}

                assert service.total_usage.input_tokens == 10
                assert service.total_usage.output_tokens == 5
                assert service._state(conn_id).response_usage.input_tokens == 0
                assert service._state(conn_id).response_usage.output_tokens == 0

    def test_response_completion_drain_sends_pending_tool_before_done(self, setup):
        _, service, input_queue, output_queue, text_output_queue, should_listen, _, response_playing, cancel_scope = (
            setup
        )
        unit = PipelineUnit(
            index=0,
            service=service,
            cancel_scope=cancel_scope,
            should_listen=should_listen,
            response_playing=response_playing,
            input_queue=input_queue,
            output_queue=output_queue,
            text_output_queue=text_output_queue,
            text_prompt_queue=Queue(),
            handlers=[],
        )
        conn_id = service.register()
        response_id, _ = service.response._ensure_response(conn_id)
        text_output_queue.put(
            AssistantTextEvent(
                text="",
                tools=[
                    {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "play_emotion",
                        "arguments": '{"emotion":"loving"}',
                    }
                ],
            )
        )
        text_output_queue.put(TokenUsageEvent(input_tokens=10, output_tokens=5))
        ws = _FakeWebSocket()

        asyncio.run(router_module._drain_pending_response_events(WebSocketTransport(ws), unit, conn_id))
        done_events = service.finish_response(conn_id)

        assert [payload["type"] for payload in ws.sent] == ["response.function_call_arguments.done"]
        assert [event.type for event in done_events] == ["response.output_audio.done", "response.done"]
        assert ws.sent[0]["response_id"] == response_id
        assert done_events[1].response.id == response_id
        assert done_events[1].response.usage.input_tokens == 10
        assert done_events[1].response.usage.output_tokens == 5
        assert text_output_queue.empty()

    def test_response_completion_drain_preserves_usage_across_non_response_boundary(self, setup):
        _, service, input_queue, output_queue, text_output_queue, should_listen, _, response_playing, cancel_scope = (
            setup
        )
        unit = PipelineUnit(
            index=0,
            service=service,
            cancel_scope=cancel_scope,
            should_listen=should_listen,
            response_playing=response_playing,
            input_queue=input_queue,
            output_queue=output_queue,
            text_output_queue=text_output_queue,
            text_prompt_queue=Queue(),
            handlers=[],
        )
        conn_id = service.register()
        response_id, _ = service.response._ensure_response(conn_id)
        text_output_queue.put(
            AssistantTextEvent(
                text="",
                tools=[{"type": "function_call", "call_id": "c1", "name": "play_emotion", "arguments": "{}"}],
            )
        )
        text_output_queue.put(SpeechStartedEvent())
        text_output_queue.put(TokenUsageEvent(input_tokens=10, output_tokens=5))
        text_output_queue.put(AssistantTextEvent(text="queued after boundary"))
        ws = _FakeWebSocket()

        asyncio.run(router_module._drain_pending_response_events(WebSocketTransport(ws), unit, conn_id))
        done_events = service.finish_response(conn_id)

        assert [payload["type"] for payload in ws.sent] == ["response.function_call_arguments.done"]
        assert ws.sent[0]["response_id"] == response_id
        assert done_events[1].response.usage.input_tokens == 10
        assert done_events[1].response.usage.output_tokens == 5

        boundary = text_output_queue.get_nowait()
        queued_assistant = text_output_queue.get_nowait()
        assert isinstance(boundary, SpeechStartedEvent)
        assert isinstance(queued_assistant, AssistantTextEvent)
        assert queued_assistant.text == "queued after boundary"
        assert text_output_queue.empty()

    def test_speech_started_does_not_cancel_when_interrupt_disabled(self, setup):
        """With interrupt_response=False, speech during playback should NOT cancel or flush."""
        from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

        app, service, _, output_queue, text_output_queue, _, _, response_playing, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                conn_id = list(service._conns.keys())[0]
                service._state(conn_id).runtime_config.session.audio.input.turn_detection = ServerVad(
                    type="server_vad",
                    interrupt_response=False,
                )
                _, response_item_id = service.response._ensure_response(conn_id)
                response_playing.set()
                text_output_queue.put(SpeechStartedEvent())
                msg = ws.receive_json()
                assert msg["type"] == "input_audio_buffer.speech_started"
                time.sleep(0.15)
                assert response_playing.is_set(), "response_playing should remain set"
                assert not cancel_scope.discarding, "cancel_scope should not be discarding"
                assert service._state(conn_id).in_response, "response should still be active"
                assert service._state(conn_id).current_item_id == response_item_id


# ===================================================================
# Cleanup
# ===================================================================


class TestCleanup:
    def test_new_connection_resets_discard_after_invalidating_generation(self, setup):
        """connect-time _clean_unit cancels+resets: stale work is invalidated, discarding cleared."""
        app, _, *_rest, cancel_scope = setup
        cancel_scope.cancel()
        assert cancel_scope.discarding
        assert cancel_scope.generation == 1
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                assert not cancel_scope.discarding
                assert cancel_scope.generation == 2

    def test_disconnect_bumps_cancel_scope_generation(self, setup):
        """_clean_unit() on disconnect calls cancel() so in-flight generations go stale."""
        app, _, _, _, _, _, _, _, cancel_scope = setup
        assert cancel_scope.generation == 0
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                assert cancel_scope.generation == 1
            # disconnect triggers _clean_unit again + drain (short timeout in tests)
            time.sleep(0.3)
        assert cancel_scope.generation == 2

    def test_disconnect_unregisters(self, setup):
        app, service, input_queue, output_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                assert len(service._conns) == 1
            # Simulate the handler chain consuming SESSION_END so the release
            # task can complete and unregister the session.
            _simulate_session_end_drain(input_queue, output_queue)
            time.sleep(0.3)
            assert len(service._conns) == 0

    def test_last_disconnect_cancels_and_clears_response_state(self, setup):
        app, service, input_queue, output_queue, text_output_queue, _, _, response_playing, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                conn_id = list(service._conns.keys())[0]
                service.response._ensure_response(conn_id)
                response_playing.set()
                output_queue.put(_pcm_bytes(256))
                text_output_queue.put(AssistantTextEvent(text="stale"))
            _simulate_session_end_drain(input_queue, output_queue)
            time.sleep(0.3)

        assert not cancel_scope.discarding
        assert cancel_scope.generation == 2
        assert not response_playing.is_set()
        assert text_output_queue.empty()


# ===================================================================
# Drain / release robustness
# ===================================================================


class TestDrainRelease:
    def test_barge_in_flush_preserves_session_end(self):
        """The output_queue flush on barge-in must not swallow an in-flight
        SESSION_END — losing it would leave the release task waiting forever."""
        q: Queue = Queue()
        q.put(_pcm_bytes(10))
        q.put(PipelineControlMessage(SESSION_END.kind, session_id="sess_a"))
        q.put(_pcm_bytes(10))
        router_module._flush_queue(q, preserve=router_module._keep_audio_sentinel)
        assert is_control_message(q.get_nowait(), SESSION_END.kind)
        assert q.empty()

    def test_quarantine_keeps_unit_unclaimable_when_session_end_never_drains(self, setup, monkeypatch):
        """With no handler chain, SESSION_END never reaches output_queue; past
        the quarantine timeout the session is unregistered (no more chat
        mutation or billing) but the unit must NOT become claimable — its
        handlers could still emit the old session's output."""
        monkeypatch.setattr(router_module, "SESSION_END_QUARANTINE_TIMEOUT_S", 0.2)
        app, service, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
            time.sleep(0.8)
            assert len(service._conns) == 0
            pool = client.get("/v1/pool").json()
            assert pool["in_use"] == 1
            assert pool["units"][0]["state"] == "stuck"
            assert pool["units"][0]["stuck_for_s"] >= 0
            with client.websocket_connect("/v1/realtime") as ws2:
                msg = ws2.receive_json()
                assert msg["type"] == "error"
                assert msg["error"]["type"] == "session_limit_reached"

    def test_quarantined_unit_returns_to_pool_after_late_drain(self, setup, monkeypatch):
        """If SESSION_END eventually drains after the quarantine kicked in, the
        chain has proven itself clean and the unit becomes claimable again."""
        monkeypatch.setattr(router_module, "SESSION_END_QUARANTINE_TIMEOUT_S", 0.2)
        app, service, input_queue, output_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
            time.sleep(0.8)
            assert client.get("/v1/pool").json()["units"][0]["state"] == "stuck"
            # Late drain: the wedged "handler chain" finally forwards SESSION_END.
            _simulate_session_end_drain(input_queue, output_queue)
            time.sleep(0.3)
            assert client.get("/v1/pool").json()["in_use"] == 0
            with client.websocket_connect("/v1/realtime") as ws2:
                assert ws2.receive_json()["type"] == "session.created"

    def test_stale_session_end_does_not_satisfy_next_sessions_drain(self):
        """A SESSION_END tagged with a force-released session's id must not set
        `drained` for the session that claimed the unit afterwards."""
        unit = _make_unit(0)
        app = create_app(pool=[unit], stop_event=ThreadingEvent())
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                assert unit.session is not None
                unit.output_queue.put(PipelineControlMessage(SESSION_END.kind, session_id="sess_stale"))
                time.sleep(0.3)
                assert not unit.session.drained.is_set()
                unit.output_queue.put(PipelineControlMessage(SESSION_END.kind, session_id=unit.session.session_id))
                time.sleep(0.3)
                assert unit.session.drained.is_set()

    def test_register_failure_still_releases_unit(self, setup, monkeypatch):
        """An exception during session setup (after the claim) must not leak the
        slot: the finally still enqueues SESSION_END and spawns the release task."""
        app, service, input_queue, output_queue, *_ = setup

        def _boom():
            raise RuntimeError("register failed")

        monkeypatch.setattr(service, "register", _boom)
        with TestClient(app) as client:
            try:
                with client.websocket_connect("/v1/realtime"):
                    pass
            except Exception:
                pass
            _simulate_session_end_drain(input_queue, output_queue)
            time.sleep(0.3)
            assert client.get("/v1/pool").json()["in_use"] == 0


# ===================================================================
# Pool semantics (new in pool refactor)
# ===================================================================


def _make_unit(index: int) -> PipelineUnit:
    text_prompt_queue: Queue = Queue()
    should_listen = ThreadingEvent()
    should_listen.set()
    return PipelineUnit(
        index=index,
        service=RealtimeService(text_prompt_queue=text_prompt_queue, should_listen=should_listen),
        cancel_scope=CancelScope(),
        should_listen=should_listen,
        response_playing=ThreadingEvent(),
        input_queue=Queue(),
        output_queue=Queue(),
        text_output_queue=Queue(),
        text_prompt_queue=text_prompt_queue,
        handlers=[],
    )


class TestPool:
    def test_pool_endpoint_reports_idle_state(self):
        pool = [_make_unit(0), _make_unit(1)]
        app = create_app(pool=pool, stop_event=ThreadingEvent())
        with TestClient(app) as client:
            r = client.get("/v1/pool")
            assert r.status_code == 200
            data = r.json()
            assert data["size"] == 2
            assert data["in_use"] == 0
            assert [u["session_id"] for u in data["units"]] == [None, None]

    def test_two_clients_claim_two_slots_third_rejected(self):
        pool = [_make_unit(0), _make_unit(1)]
        app = create_app(pool=pool, stop_event=ThreadingEvent())
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws1:
                ws1.receive_json()  # session.created
                with client.websocket_connect("/v1/realtime") as ws2:
                    ws2.receive_json()  # session.created (different unit)
                    with client.websocket_connect("/v1/realtime") as ws3:
                        msg = ws3.receive_json()
                        assert msg["type"] == "error"
                        assert msg["error"]["type"] == "session_limit_reached"
                    # Pool now reports 2 in_use
                    r = client.get("/v1/pool")
                    assert r.json()["in_use"] == 2

    def test_usage_aggregates_errors_by_type_across_units(self):
        pool = [_make_unit(0), _make_unit(1)]
        pool[0].service.total_usage.record_error("foo")
        pool[0].service.total_usage.record_error("foo")
        pool[1].service.total_usage.record_error("bar")
        app = create_app(pool=pool, stop_event=ThreadingEvent())
        with TestClient(app) as client:
            data = client.get("/v1/usage").json()
            assert data["errors_by_type"] == {"foo": 2, "bar": 1}
            assert data["total_errors"] == 3
