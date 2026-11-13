"""History follows the assistant audio the user actually heard (#801)."""

import asyncio
import logging
import time
from threading import Event as ThreadingEvent

from openai.types.realtime import (
    ConversationItemDeleteEvent,
    ConversationItemRetrieveEvent,
    ConversationItemTruncateEvent,
)
from openai.types.realtime.conversation_item import RealtimeConversationItemAssistantMessage
from openai.types.responses import ResponseOutputMessage, ResponseOutputText, ResponseReasoningItem
from starlette.testclient import TestClient
from starlette.websockets import WebSocketState

import speech_to_speech.api.openai_realtime.handlers.response as response_module
import speech_to_speech.api.openai_realtime.websocket_router as router_module
from speech_to_speech.api.openai_realtime.transports import SessionTransport, WebSocketTransport
from speech_to_speech.LLM.chat import (
    Chat,
    CompactionResult,
    ResponsesAssistantMessage,
    make_assistant_message,
    make_user_message,
)
from speech_to_speech.pipeline.events import AssistantOutputEvent, AssistantResponseDoneEvent, SpeechStartedEvent
from speech_to_speech.pipeline.history import ResponseHistory
from tests.openai_realtime.conftest import make_pipeline_unit

RATE = 16_000
FIRST = "Sure, here are five ways to improve it."
SECOND = "First, tighten the thesis so every paragraph supports it."
RESPONSE_KEY = "response_essay"


def _speak(service, conn_id, parts_with_seconds, *, complete=True, response_key=RESPONSE_KEY):
    """Stream assistant transcript parts, each followed by its delivered audio."""
    chat = service._state(conn_id).runtime_config.chat
    full_text = " ".join(text for text, _ in parts_with_seconds)
    history = ResponseHistory.capture(
        chat,
        [make_assistant_message(full_text)],
        after_item_id=chat.history_anchor_id(),
        complete=complete,
    )
    item_id = None
    for text, seconds in parts_with_seconds:
        events = service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(text=text, response_key=response_key, history=history),
        )
        item_id = item_id or next(event.item_id for event in events if hasattr(event, "item_id"))
        for _ in range(int(seconds * 10)):
            service.encode_audio_chunk(conn_id, b"\x00\x00" * (RATE // 10), response_key)
    return item_id


def _history(service, conn_id):
    return [
        (item.role, item.status if item.role == "assistant" else None, item.content[0].text)
        for item in service._state(conn_id).runtime_config.chat.buffer
    ]


def _start_conversation(service, conn_id):
    chat = service._state(conn_id).runtime_config.chat
    chat.add_item(make_user_message("Hi"))
    chat.add_item(make_assistant_message("Hello! What are you working on?"))
    chat.add_item(make_user_message("How can I improve my essay?"))


EARLIER_TURNS = [
    ("user", None, "Hi"),
    ("assistant", None, "Hello! What are you working on?"),
    ("user", None, "How can I improve my essay?"),
]


def _truncate(item_id, audio_end_ms):
    return ConversationItemTruncateEvent(
        type="conversation.item.truncate",
        item_id=item_id,
        content_index=0,
        audio_end_ms=audio_end_ms,
    )


def test_interrupting_after_a_few_words_keeps_only_the_played_portion(service, conn_id):
    _start_conversation(service, conn_id)
    item_id = _speak(service, conn_id, [(FIRST, 2.0), (SECOND, 3.0)])

    service.finish_response(conn_id, status="cancelled", reason="turn_detected")
    service.handle_conversation_item_truncate(conn_id, _truncate(item_id, 800))

    # 0.8 s of the 2 s first part covers 15.6 of its 39 characters.
    assert _history(service, conn_id) == [*EARLIER_TURNS, ("assistant", "incomplete", "Sure, here are")]
    request = service._state(conn_id).runtime_config.chat.to_transformers_chat()
    assert "five ways" not in str(request)
    assert "thesis" not in str(request)
    # Replayed Responses assistant messages need provider-style message ids.
    replayed = service._state(conn_id).runtime_config.chat.to_responses_api_chat()[-1]
    assert replayed["id"].startswith("msg_")
    assert replayed["status"] == "incomplete"


def test_cancelled_reply_keeps_only_speech_already_delivered(service, conn_id):
    """Without a client position, the server keeps the speech it sent, never unsynthesized text."""
    _start_conversation(service, conn_id)
    _speak(service, conn_id, [(FIRST, 2.0), (SECOND, 0.0)])

    service.finish_response(conn_id, status="cancelled", reason="turn_detected")

    assert _history(service, conn_id) == [*EARLIER_TURNS, ("assistant", "incomplete", FIRST)]


def test_fully_generated_reply_with_buffered_audio_is_cut_to_playback(service, conn_id):
    _start_conversation(service, conn_id)
    item_id = _speak(service, conn_id, [(FIRST, 2.0), (SECOND, 3.0)])
    service.dispatch_pipeline_event(conn_id, AssistantResponseDoneEvent(response_key=RESPONSE_KEY))
    service.finish_response(conn_id, response_key=RESPONSE_KEY)
    assert _history(service, conn_id)[-1] == ("assistant", None, f"{FIRST} {SECOND}")

    service.handle_conversation_item_truncate(conn_id, _truncate(item_id, 2600))

    # The first part played in full; 0.6 s of the 3 s second part covers 11.4 characters.
    assert _history(service, conn_id) == [*EARLIER_TURNS, ("assistant", "incomplete", f"{FIRST} First,")]
    request = service._state(conn_id).runtime_config.chat.to_responses_api_chat()
    assert "thesis" not in str(request)
    assert request[-1]["status"] == "incomplete"


def test_reply_with_no_played_audio_is_not_kept_as_delivered(service, conn_id):
    _start_conversation(service, conn_id)
    item_id = _speak(service, conn_id, [(FIRST, 2.0)])
    service.finish_response(conn_id, status="cancelled", reason="turn_detected")

    service.handle_conversation_item_truncate(conn_id, _truncate(item_id, 0))

    assert _history(service, conn_id) == EARLIER_TURNS


def test_cancelled_reply_before_any_audio_leaves_no_assistant_message(service, conn_id):
    _start_conversation(service, conn_id)
    _speak(service, conn_id, [(FIRST, 0.0)])

    service.finish_response(conn_id, status="cancelled", reason="turn_detected")

    assert _history(service, conn_id) == EARLIER_TURNS


def test_uninterrupted_reply_keeps_its_full_history(service, conn_id):
    _start_conversation(service, conn_id)
    item_id = _speak(service, conn_id, [(FIRST, 2.0), (SECOND, 3.0)])
    service.dispatch_pipeline_event(conn_id, AssistantResponseDoneEvent(response_key=RESPONSE_KEY))
    service.finish_response(conn_id, response_key=RESPONSE_KEY)

    # A truncate at or past the end of delivered audio removes nothing.
    service.handle_conversation_item_truncate(conn_id, _truncate(item_id, 5000))

    assert _history(service, conn_id) == [*EARLIER_TURNS, ("assistant", None, f"{FIRST} {SECOND}")]


def test_truncating_an_in_progress_item_waits_for_its_response_to_end(service, conn_id):
    _start_conversation(service, conn_id)
    item_id = _speak(service, conn_id, [(FIRST, 2.0)])

    service.handle_conversation_item_truncate(conn_id, _truncate(item_id, 0))
    service.finish_response(conn_id, status="cancelled", reason="client_cancelled")

    # The ignored truncate leaves the cancellation's own estimate in place.
    assert _history(service, conn_id) == [*EARLIER_TURNS, ("assistant", "incomplete", "Sure, here are five ways")]


def test_cancel_during_synthesis_does_not_keep_unsynthesized_words(service, conn_id):
    """0.9 s of a ~3 s sentence was synthesized when the user barged in."""
    _start_conversation(service, conn_id)
    item_id = _speak(service, conn_id, [(FIRST, 0.9)])

    service.finish_response(conn_id, status="cancelled", reason="turn_detected")

    # Without a measured speaking rate, 12 characters per second gives the
    # 39-character sentence 3.25 s; 0.9 s of it covers 10.8 characters.
    assert _history(service, conn_id)[-1] == ("assistant", "incomplete", "Sure, here")

    service.handle_conversation_item_truncate(conn_id, _truncate(item_id, 800))

    assert _history(service, conn_id)[-1] == ("assistant", "incomplete", "Sure,")


def test_cancel_during_synthesis_uses_the_measured_speaking_rate(service, conn_id):
    _start_conversation(service, conn_id)
    _speak(service, conn_id, [(FIRST, 2.0), (SECOND, 1.0)])

    service.finish_response(conn_id, status="cancelled", reason="turn_detected")

    # The first segment took 2 s for 39 characters, so the 58-character second
    # segment takes about 2.97 s; its 1 s delivered covers 19.5 characters.
    assert _history(service, conn_id)[-1] == ("assistant", "incomplete", f"{FIRST} First, tighten the")


def test_an_estimated_duration_does_not_feed_later_speaking_rates(service, conn_id):
    """A cancel-time estimate is not measured speech."""
    _start_conversation(service, conn_id)
    _speak(service, conn_id, [(FIRST, 0.9)], response_key="response_first")
    service.finish_response(conn_id, status="cancelled", reason="turn_detected")
    service._state(conn_id).runtime_config.chat.add_item(make_user_message("Go on."))

    _speak(service, conn_id, [(FIRST, 2.0), (SECOND, 1.0)], response_key="response_second")
    service.finish_response(conn_id, status="cancelled", reason="turn_detected")

    # Only the fully synthesized 2 s first segment sets the rate, as when it
    # is the only speech (see the measured speaking rate test above).
    assert _history(service, conn_id)[-1] == ("assistant", "incomplete", f"{FIRST} First, tighten the")


def test_server_paced_playback_cuts_the_newest_items_by_unplayed_audio(service, conn_id):
    _start_conversation(service, conn_id)
    _speak(service, conn_id, [(FIRST, 2.0), (SECOND, 3.0)])
    service.dispatch_pipeline_event(conn_id, AssistantResponseDoneEvent(response_key=RESPONSE_KEY))
    service.finish_response(conn_id, response_key=RESPONSE_KEY)

    service.truncate_unplayed_audio(conn_id, 4200.0)

    assert _history(service, conn_id) == [*EARLIER_TURNS, ("assistant", "incomplete", "Sure, here are")]


def test_provider_native_message_is_replaced_by_its_heard_text():
    chat = Chat(size=10)
    chat.add_item(make_user_message("Explain it"))
    chat.add_item(ResponseReasoningItem(id="rs_1", type="reasoning", summary=[]))
    chat.add_item(
        ResponsesAssistantMessage(
            type="message",
            role="assistant",
            id="msg_1",
            content=[{"type": "output_text", "text": "Sure, here are five ways."}],
            response_item=ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="completed",
                content=[ResponseOutputText(type="output_text", text="Sure, here are five ways.", annotations=[])],
            ),
        )
    )
    chat.add_item(make_user_message("Wait"))

    assert chat.truncate_assistant_message("msg_1", "Sure, here")

    assert [type(item) for item in chat.buffer] == [
        type(chat.buffer[0]),
        RealtimeConversationItemAssistantMessage,
        type(chat.buffer[0]),
    ]
    assert chat.buffer[1].content[0].text == "Sure, here"
    request = chat.to_responses_api_chat()
    assert all(item.get("type") != "reasoning" for item in request)
    assert "five ways" not in str(request)
    assert request[1]["id"].startswith("msg_")
    # The original id still addresses the replacement.
    assert chat.truncate_assistant_message("msg_1", "Sure,")
    assert chat.buffer[1].content[0].text == "Sure,"


class _WebSocketTransport(SessionTransport):
    kind = "websocket"

    def __init__(self):
        self.sent = []

    async def send_events(self, events):
        self.sent.extend(events)

    async def send_audio_chunk(self, service, session_id, pcm, response_key=None):
        raise AssertionError("not used")

    def discard_pending_audio(self):
        return 0.0

    async def close(self):
        pass


async def test_websocket_truncate_updates_history_without_an_acknowledgement():
    unit = make_pipeline_unit()
    conn_id = unit.service.register()
    _start_conversation(unit.service, conn_id)
    item_id = _speak(unit.service, conn_id, [(FIRST, 2.0)])
    unit.service.dispatch_pipeline_event(conn_id, AssistantResponseDoneEvent(response_key=RESPONSE_KEY))
    unit.service.finish_response(conn_id, status="cancelled", reason="turn_detected")
    transport = _WebSocketTransport()

    await router_module._dispatch_client_event(
        unit,
        conn_id,
        _truncate(item_id, 800).model_dump(),
        transport,
        transport_kind="websocket",
    )

    # No conversation.item.truncated acknowledgement is sent.
    assert transport.sent == []
    assert _history(unit.service, conn_id)[-1] == ("assistant", "incomplete", "Sure, here are")


# ── Clients that never send conversation.item.truncate ─────────────────────


class _FakeWebSocket:
    application_state = WebSocketState.CONNECTED

    def __init__(self):
        self.sent = []

    async def send_json(self, data):
        self.sent.append(data)


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


async def _speak_over(transport, service, conn_id, parts_with_seconds, *, complete=True):
    """Like ``_speak``, but audio reaches the client through *transport*."""
    chat = service._state(conn_id).runtime_config.chat
    full_text = " ".join(text for text, _ in parts_with_seconds)
    history = ResponseHistory.capture(
        chat,
        [make_assistant_message(full_text)],
        after_item_id=chat.history_anchor_id(),
        complete=complete,
    )
    for text, seconds in parts_with_seconds:
        service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(text=text, response_key=RESPONSE_KEY, history=history),
        )
        for _ in range(int(seconds * 10)):
            await transport.send_audio_chunk(service, conn_id, b"\x00\x00" * (RATE // 10), RESPONSE_KEY)


def _websocket_session():
    unit = make_pipeline_unit()
    conn_id = unit.service.register()
    _start_conversation(unit.service, conn_id)
    clock = _Clock()
    transport = WebSocketTransport(_FakeWebSocket(), clock=clock)
    return unit, conn_id, transport, clock


async def _cancel(unit, conn_id, transport):
    await router_module._dispatch_client_event(
        unit,
        conn_id,
        {"type": "response.cancel"},
        transport,
        transport_kind="websocket",
    )


async def test_websocket_transport_estimates_audio_the_client_has_not_played():
    clock = _Clock()
    transport = WebSocketTransport(_FakeWebSocket(), clock=clock)
    service = make_pipeline_unit().service
    conn_id = service.register()

    await transport.send_audio_chunk(service, conn_id, b"\x00\x00" * RATE * 2, RESPONSE_KEY)
    clock.now += 0.5
    assert transport.discard_pending_audio() == 1500.0
    # The client flushed what was left, so nothing is pending any more.
    assert transport.discard_pending_audio() == 0.0

    # Audio that arrives after playback caught up starts a new playback run.
    clock.now += 10
    await transport.send_audio_chunk(service, conn_id, b"\x00\x00" * RATE, RESPONSE_KEY)
    clock.now += 0.25
    assert transport.discard_pending_audio() == 750.0


async def test_websocket_cancel_without_truncate_cuts_a_completed_reply_to_playback():
    """A client that only cancels and flushes its player (e.g. Reachy Mini)."""
    unit, conn_id, transport, clock = _websocket_session()
    service = unit.service
    await _speak_over(transport, service, conn_id, [(FIRST, 2.0), (SECOND, 3.0)])
    service.dispatch_pipeline_event(conn_id, AssistantResponseDoneEvent(response_key=RESPONSE_KEY))
    service.finish_response(conn_id, response_key=RESPONSE_KEY)
    assert _history(service, conn_id)[-1] == ("assistant", None, f"{FIRST} {SECOND}")

    clock.now += 2.6
    await _cancel(unit, conn_id, transport)

    assert _history(service, conn_id) == [*EARLIER_TURNS, ("assistant", "incomplete", f"{FIRST} First,")]


async def test_websocket_cancel_without_truncate_cuts_an_active_reply_to_playback():
    unit, conn_id, transport, clock = _websocket_session()
    service = unit.service
    await _speak_over(transport, service, conn_id, [(FIRST, 2.0), (SECOND, 3.0)])
    assert service._state(conn_id).in_response

    clock.now += 0.8
    await _cancel(unit, conn_id, transport)

    assert _history(service, conn_id) == [*EARLIER_TURNS, ("assistant", "incomplete", "Sure, here are")]


async def test_websocket_reply_that_finished_playing_is_kept_whole():
    unit, conn_id, transport, clock = _websocket_session()
    service = unit.service
    await _speak_over(transport, service, conn_id, [(FIRST, 2.0), (SECOND, 3.0)])
    service.dispatch_pipeline_event(conn_id, AssistantResponseDoneEvent(response_key=RESPONSE_KEY))
    service.finish_response(conn_id, response_key=RESPONSE_KEY)

    clock.now += 6.0
    await _cancel(unit, conn_id, transport)

    assert _history(service, conn_id)[-1] == ("assistant", None, f"{FIRST} {SECOND}")


async def test_client_truncate_replaces_the_server_estimate():
    unit, conn_id, transport, clock = _websocket_session()
    service = unit.service
    await _speak_over(transport, service, conn_id, [(FIRST, 2.0), (SECOND, 3.0)])
    service.dispatch_pipeline_event(conn_id, AssistantResponseDoneEvent(response_key=RESPONSE_KEY))
    service.finish_response(conn_id, response_key=RESPONSE_KEY)
    item_id = next(iter(service._state(conn_id).assistant_audio_segments))
    clock.now += 2.6
    await _cancel(unit, conn_id, transport)
    assert _history(service, conn_id)[-1] == ("assistant", "incomplete", f"{FIRST} First,")

    # The client heard less than estimated: the cut is measured from the
    # original segments, so 0.2 s of the second part keeps none of it.
    service.handle_conversation_item_truncate(conn_id, _truncate(item_id, 2200))
    assert _history(service, conn_id)[-1] == ("assistant", "incomplete", FIRST)

    # The client heard more than estimated: its position wins.
    service.handle_conversation_item_truncate(conn_id, _truncate(item_id, 4000))
    assert _history(service, conn_id)[-1] == ("assistant", "incomplete", f"{FIRST} First, tighten the thesis so every")


def test_forgotten_audio_items_drop_their_cuts(service, conn_id, monkeypatch):
    monkeypatch.setattr(response_module, "_AUDIO_ITEM_HISTORY", 1)
    st = service._state(conn_id)
    service.response.record_audio_samples(conn_id, "item_old", RATE)
    st.assistant_audio_cuts["item_old"] = RATE // 2

    service.response.record_audio_samples(conn_id, "item_new", RATE)

    assert list(st.assistant_audio_segments) == ["item_new"]
    assert st.assistant_audio_cuts == {}


def test_websocket_server_vad_barge_in_cuts_an_active_reply_to_playback():
    """Server VAD cancels the reply before the unplayed audio is removed."""
    unit = make_pipeline_unit()
    stop_event = ThreadingEvent()
    app = router_module.create_app(pool=[unit], stop_event=stop_event)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()  # session.created
            assert unit.session is not None
            conn_id = unit.session.session_id
            service = unit.service
            _start_conversation(service, conn_id)
            clock = _Clock()
            transport = WebSocketTransport(_FakeWebSocket(), clock=clock)
            unit.session.transport = transport
            asyncio.run(_speak_over(transport, service, conn_id, [(FIRST, 2.0), (SECOND, 3.0)]))
            assert service._state(conn_id).in_response

            clock.now += 0.8
            unit.text_output_queue.put(SpeechStartedEvent())

            chat = service._state(conn_id).runtime_config.chat
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and _history(service, conn_id)[-1][2] != "Sure, here are":
                time.sleep(0.02)
            assert _history(service, conn_id) == [*EARLIER_TURNS, ("assistant", "incomplete", "Sure, here are")]
            assert chat.buffer[-1].role == "assistant"
    stop_event.set()


def test_retrieve_returns_the_heard_transcript_of_a_truncated_reply(service, conn_id):
    _start_conversation(service, conn_id)
    item_id = _speak(service, conn_id, [(FIRST, 2.0), (SECOND, 3.0)])
    service.dispatch_pipeline_event(conn_id, AssistantResponseDoneEvent(response_key=RESPONSE_KEY))
    service.finish_response(conn_id, response_key=RESPONSE_KEY)
    service.handle_conversation_item_truncate(conn_id, _truncate(item_id, 800))

    [event] = service.handle_conversation_item_retrieve(
        conn_id,
        ConversationItemRetrieveEvent(type="conversation.item.retrieve", item_id=item_id),
    )

    assert event.type == "conversation.item.retrieved"
    assert event.item.id == item_id
    assert event.item.status == "incomplete"
    assert [(part.type, part.transcript) for part in event.item.content] == [("output_audio", "Sure, here are")]


def _retrieve(service, conn_id, item_id):
    [event] = service.handle_conversation_item_retrieve(
        conn_id,
        ConversationItemRetrieveEvent(type="conversation.item.retrieve", item_id=item_id),
    )
    return event


def _reply_nobody_heard(service, conn_id):
    _start_conversation(service, conn_id)
    item_id = _speak(service, conn_id, [(FIRST, 2.0)])
    service.finish_response(conn_id, status="cancelled", reason="turn_detected")
    service.handle_conversation_item_truncate(conn_id, _truncate(item_id, 0))
    return item_id


def test_retrieve_of_a_reply_nobody_heard_returns_it_empty(service, conn_id):
    item_id = _reply_nobody_heard(service, conn_id)
    # Later items must not make the removed reply unknown.
    service.conversation.record_acknowledged_item(conn_id, "item_later")

    event = _retrieve(service, conn_id, item_id)

    # Like a truncate to 0 ms on OpenAI: the item stays, with nothing heard.
    assert event.type == "conversation.item.retrieved"
    assert event.item.id == item_id
    assert event.item.status == "incomplete"
    assert [(part.type, part.transcript) for part in event.item.content] == [("output_audio", "")]
    # The model still never sees it, and an expected lookup is not an error.
    assert _history(service, conn_id) == EARLIER_TURNS
    assert service.get_usage()["total_errors"] == 0


def test_deleting_a_reply_nobody_heard_succeeds_and_is_final(service, conn_id):
    item_id = _reply_nobody_heard(service, conn_id)

    [deleted] = service.handle_conversation_item_delete(
        conn_id,
        ConversationItemDeleteEvent(type="conversation.item.delete", item_id=item_id),
    )
    service.handle_conversation_item_truncate(conn_id, _truncate(item_id, 800))

    assert deleted.type == "conversation.item.deleted"
    assert _retrieve(service, conn_id, item_id).error.type == "conversation_item_not_found"
    assert _history(service, conn_id) == EARLIER_TURNS


async def _estimate_removes_a_completed_reply():
    """Cancel a fully sent two-second reply before any of it plays."""
    unit, conn_id, transport, clock = _websocket_session()
    service = unit.service
    await _speak_over(transport, service, conn_id, [(FIRST, 2.0)])
    service.dispatch_pipeline_event(conn_id, AssistantResponseDoneEvent(response_key=RESPONSE_KEY))
    service.finish_response(conn_id, response_key=RESPONSE_KEY)
    item_id = next(iter(service._state(conn_id).assistant_audio_segments))
    await _cancel(unit, conn_id, transport)
    assert _history(service, conn_id) == EARLIER_TURNS
    # The user's next turn arrives before the client reports its position.
    service._state(conn_id).runtime_config.chat.add_item(make_user_message("Wait"))
    return unit, conn_id, item_id


async def test_client_truncate_restores_a_reply_the_estimate_removed():
    unit, conn_id, item_id = await _estimate_removes_a_completed_reply()

    unit.service.handle_conversation_item_truncate(conn_id, _truncate(item_id, 800))

    assert _history(unit.service, conn_id) == [
        *EARLIER_TURNS,
        ("assistant", "incomplete", "Sure, here are"),
        ("user", None, "Wait"),
    ]
    request = unit.service._state(conn_id).runtime_config.chat.to_responses_api_chat()
    assert request[-2]["id"].startswith("msg_")
    # The restored reply is addressed by its Realtime id again.
    unit.service.handle_conversation_item_truncate(conn_id, _truncate(item_id, 400))
    assert _history(unit.service, conn_id)[-2] == ("assistant", "incomplete", "Sure,")


async def test_client_truncate_does_not_restore_an_explicitly_deleted_reply():
    unit, conn_id, item_id = await _estimate_removes_a_completed_reply()
    unit.service.handle_conversation_item_delete(
        conn_id,
        ConversationItemDeleteEvent(type="conversation.item.delete", item_id=item_id),
    )

    unit.service.handle_conversation_item_truncate(conn_id, _truncate(item_id, 800))

    assert _history(unit.service, conn_id) == [*EARLIER_TURNS, ("user", None, "Wait")]


def _chat_with_unheard_reply(size=10):
    chat = Chat(size=size)
    chat.add_item(make_user_message("Hi"))
    chat.add_item(make_assistant_message("Hello there, how can I help?"))
    reply_id = chat.buffer[-1].id
    assert chat.truncate_assistant_message(reply_id, "")
    return chat, reply_id


def test_unheard_reply_is_not_restored_once_its_turn_is_gone():
    chat, reply_id = _chat_with_unheard_reply(size=1)
    for text in ("Second", "Third", "Fourth"):
        chat.add_item(make_user_message(text))
    assert "Hi" not in str(chat.buffer)

    assert not chat.truncate_assistant_message(reply_id, "Hello there")

    assert all(item.role == "user" for item in chat.buffer)


def test_unheard_reply_is_not_restored_over_a_new_item_with_its_id():
    chat, reply_id = _chat_with_unheard_reply()
    reused = make_user_message("Wait")
    reused.id = reply_id
    chat.add_item(reused)

    assert not chat.truncate_assistant_message(reply_id, "Hello there")

    assert [item.role for item in chat.buffer] == ["user", "user"]
    assert chat.get_item(reply_id).role == "user"


def test_restored_reply_can_be_removed_and_restored_again():
    chat, reply_id = _chat_with_unheard_reply()
    chat.add_item(make_user_message("Wait"))

    assert chat.truncate_assistant_message(reply_id, "Hello there,")
    assert chat.truncate_assistant_message(reply_id, "")
    assert chat.truncate_assistant_message(reply_id, "Hello")

    assert [(item.role, item.content[0].text) for item in chat.buffer] == [
        ("user", "Hi"),
        ("assistant", "Hello"),
        ("user", "Wait"),
    ]


def test_unheard_reply_is_not_restored_over_a_new_assistant_message_with_its_id():
    chat, reply_id = _chat_with_unheard_reply()
    reused = make_assistant_message("A new unrelated assistant message")
    reused.id = reply_id
    chat.add_item(reused)

    # Late playback reports for the removed reply's audio leave it untouched.
    assert not chat.truncate_assistant_message(reply_id, "Hello there")
    assert not chat.truncate_assistant_message(reply_id, "")

    assert [(item.role, item.status, item.content[0].text) for item in chat.buffer] == [
        ("user", None, "Hi"),
        ("assistant", None, "A new unrelated assistant message"),
    ]


def _chat_with_unheard_greeting():
    """An unheard greeting that opened the conversation, before any user turn."""
    chat = Chat(size=1)
    chat.add_item(make_assistant_message("Welcome!"))
    greeting_id = chat.buffer[0].id
    assert chat.truncate_assistant_message(greeting_id, "")
    return chat, greeting_id


def test_unheard_first_reply_is_not_restored_once_evicted():
    chat, greeting_id = _chat_with_unheard_greeting()
    for text in ("First", "Second", "Third", "Fourth"):
        chat.add_item(make_user_message(text))

    assert not chat.truncate_assistant_message(greeting_id, "Welcome")

    assert all(item.role == "user" for item in chat.buffer)


def test_unheard_reply_is_not_restored_once_compacted():
    chat, greeting_id = _chat_with_unheard_greeting()
    chat.add_item(make_user_message("First"))
    first_id = chat.buffer[0].id

    chat._apply_compaction(
        CompactionResult(user_summary="Earlier", assistant_summary="Summary"),
        {first_id},
        chat._gen_counter,
    )

    assert not chat.truncate_assistant_message(greeting_id, "Welcome")
    assert [item.content[0].text for item in chat.buffer] == ["Earlier", "Summary"]


def test_cut_replies_are_logged_at_debug(service, conn_id, caplog):
    _start_conversation(service, conn_id)
    item_id = _speak(service, conn_id, [(FIRST, 2.0)])

    with caplog.at_level(logging.DEBUG, logger="speech_to_speech.api.openai_realtime.handlers.response"):
        service.finish_response(conn_id, status="cancelled", reason="turn_detected")
        service.handle_conversation_item_truncate(conn_id, _truncate(item_id, 800))

    lines = [record.getMessage() for record in caplog.records if "kept in history" in record.getMessage()]
    assert lines == [
        f"Interrupted reply {item_id} kept in history (audio sent before cancel): 'Sure, here are five ways', "
        "dropped 15 of 39 chars",
        f"Interrupted reply {item_id} kept in history (client playback position): 'Sure,', dropped 34 of 39 chars",
    ]
    assert all(record.levelno == logging.DEBUG for record in caplog.records if "kept in history" in record.getMessage())


class _GatedWebSocket(_FakeWebSocket):
    """Holds audio sends open while blocked, like a slow socket."""

    def __init__(self):
        super().__init__()
        self.block = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def send_json(self, data):
        if self.block and data["type"] == "response.output_audio.delta":
            self.entered.set()
            await self.release.wait()
        self.sent.append(data)


async def test_cancel_during_an_audio_send_counts_that_audio_as_unplayed():
    unit = make_pipeline_unit()
    service = unit.service
    conn_id = service.register()
    _start_conversation(service, conn_id)
    clock = _Clock()
    socket = _GatedWebSocket()
    transport = WebSocketTransport(socket, clock=clock)
    await _speak_over(transport, service, conn_id, [(FIRST, 1.0)])
    clock.now += 0.8
    socket.block = True
    sending = asyncio.create_task(transport.send_audio_chunk(service, conn_id, bytes(6400), RESPONSE_KEY))
    await socket.entered.wait()

    cancelling = asyncio.create_task(_cancel(unit, conn_id, transport))
    await asyncio.sleep(0)
    socket.release.set()
    await asyncio.gather(sending, cancelling)

    # 0.8 s of 1.2 s sent played, the same as when the send had completed.
    assert _history(service, conn_id)[-1] == ("assistant", "incomplete", "Sure,")
    # The send that finished after the cancel adds nothing back.
    assert transport.discard_pending_audio() == 0.0


async def test_late_playback_report_does_not_edit_a_new_item_reusing_a_deleted_id():
    unit = make_pipeline_unit()
    service = unit.service
    conn_id = service.register()
    _start_conversation(service, conn_id)
    item_id = _speak(service, conn_id, [(FIRST, 2.0)])
    service.dispatch_pipeline_event(conn_id, AssistantResponseDoneEvent(response_key=RESPONSE_KEY))
    service.finish_response(conn_id, response_key=RESPONSE_KEY)
    transport = _WebSocketTransport()

    async def send(event):
        await router_module._dispatch_client_event(unit, conn_id, event, transport, transport_kind="websocket")

    await send(_truncate(item_id, 800).model_dump())
    await send({"type": "conversation.item.delete", "item_id": item_id})
    replacement = "A completely different assistant message"
    await send(
        {
            "type": "conversation.item.create",
            "item": {
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": replacement}],
            },
        }
    )
    await send(_truncate(item_id, 400).model_dump())

    assert _history(service, conn_id)[-1] == ("assistant", None, replacement)
    assert all(event.type != "error" for event in transport.sent)


def test_unheard_native_reply_is_not_restored_once_its_own_user_turn_is_gone():
    """Its reasoning item leaves with it, so it cannot anchor the reply's turn."""
    chat = Chat(size=10)
    chat.add_item(make_user_message("Older question"))
    chat.add_item(make_assistant_message("Older answer."))
    chat.add_item(make_user_message("Target user turn"))
    target_turn_id = chat.buffer[-1].id
    chat.add_item(ResponseReasoningItem(id="rs_target", type="reasoning", summary=[]))
    chat.add_item(
        ResponsesAssistantMessage(
            type="message",
            role="assistant",
            id="msg_target",
            content=[{"type": "output_text", "text": "Heard target reply."}],
            response_item=ResponseOutputMessage(
                id="msg_target",
                type="message",
                role="assistant",
                status="completed",
                content=[ResponseOutputText(type="output_text", text="Heard target reply.", annotations=[])],
            ),
        )
    )
    assert chat.truncate_assistant_message("msg_target", "")

    chat.delete_item(target_turn_id)

    assert not chat.truncate_assistant_message("msg_target", "Heard")
    assert [item.content[0].text for item in chat.buffer] == ["Older question", "Older answer."]


def test_truncate_at_the_end_of_audio_with_a_fractional_millisecond_keeps_the_reply(service, conn_id):
    """audio_end_ms is whole milliseconds, so the end of 2000.4375 ms is reported as 2000."""
    _start_conversation(service, conn_id)
    chat = service._state(conn_id).runtime_config.chat
    history = ResponseHistory.capture(
        chat, [make_assistant_message(FIRST)], after_item_id=chat.history_anchor_id(), complete=True
    )
    events = service.dispatch_pipeline_event(
        conn_id, AssistantOutputEvent(text=FIRST, response_key=RESPONSE_KEY, history=history)
    )
    item_id = next(event.item_id for event in events if hasattr(event, "item_id"))
    service.encode_audio_chunk(conn_id, b"\x00\x00" * 32_007, RESPONSE_KEY)
    service.dispatch_pipeline_event(conn_id, AssistantResponseDoneEvent(response_key=RESPONSE_KEY))
    service.finish_response(conn_id, response_key=RESPONSE_KEY)

    service.handle_conversation_item_truncate(conn_id, _truncate(item_id, 2000))

    assert _history(service, conn_id) == [*EARLIER_TURNS, ("assistant", None, FIRST)]
