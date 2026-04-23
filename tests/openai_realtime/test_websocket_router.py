"""Integration tests for api.openai_realtime.websocket_router.

Uses Starlette's synchronous TestClient with WebSocket support to exercise
the full FastAPI app produced by ``create_app``.  Each test gets a fresh
app, service, and set of queues so there is no cross-test state.
"""

import base64
import time
from queue import Queue
from threading import Event as ThreadingEvent

import pytest
from starlette.testclient import TestClient

from speech_to_speech.api.openai_realtime.service import CHUNK_SIZE_BYTES, RealtimeService
from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.control import SESSION_END, is_control_message
from speech_to_speech.pipeline.events import AssistantTextEvent, SpeechStartedEvent
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def setup():
    """Return (app, service, input_queue, output_queue, text_output_queue, should_listen, stop_event, response_playing, cancel_scope)."""
    text_prompt_queue = Queue()
    should_listen = ThreadingEvent()
    should_listen.set()
    service = RealtimeService(
        text_prompt_queue=text_prompt_queue,
        should_listen=should_listen,
    )
    input_queue = Queue()
    output_queue = Queue()
    text_output_queue = Queue()
    stop_event = ThreadingEvent()
    response_playing = ThreadingEvent()
    cancel_scope = CancelScope()
    app = create_app(
        service, input_queue, output_queue, text_output_queue, should_listen, response_playing, cancel_scope, stop_event
    )
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


def _pcm_bytes(n_samples: int) -> bytes:
    return b"\x00" * (n_samples * 2)


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
                service.response._ensure_response(conn_id)
                response_playing.set()
                text_output_queue.put(SpeechStartedEvent())
                msg = ws.receive_json()
                assert msg["type"] == "input_audio_buffer.speech_started"
                time.sleep(0.15)
                assert response_playing.is_set(), "response_playing should remain set"
                assert not cancel_scope.discarding, "cancel_scope should not be discarding"
                assert service._state(conn_id).in_response, "response should still be active"


# ===================================================================
# Cleanup
# ===================================================================


class TestCleanup:
    def test_new_connection_resets_discard_after_invalidating_generation(self, setup):
        """connect-time clean_session cancels+resets: stale work is invalidated, discarding cleared."""
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
        """clean_session() calls cancel() so in-flight pipeline generations go stale."""
        app, _, _, _, _, _, _, _, cancel_scope = setup
        assert cancel_scope.generation == 0
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                assert cancel_scope.generation == 1
            time.sleep(0.2)
        assert cancel_scope.generation == 2

    def test_disconnect_unregisters(self, setup):
        app, service, input_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                assert len(service._conns) == 1
            time.sleep(0.2)
            assert len(service._conns) == 0
            end = input_queue.get(timeout=1)
            assert is_control_message(end, SESSION_END.kind)

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
            time.sleep(0.2)

        assert not cancel_scope.discarding
        assert cancel_scope.generation == 2
        assert not response_playing.is_set()
        assert output_queue.empty()
        assert text_output_queue.empty()
        end = input_queue.get(timeout=1)
        assert is_control_message(end, SESSION_END.kind)
