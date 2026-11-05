"""History follows the assistant audio the user actually heard (#801)."""

from openai.types.realtime import ConversationItemTruncateEvent
from openai.types.realtime.conversation_item import RealtimeConversationItemAssistantMessage
from openai.types.responses import ResponseOutputMessage, ResponseOutputText, ResponseReasoningItem

import speech_to_speech.api.openai_realtime.websocket_router as router_module
from speech_to_speech.api.openai_realtime.transports import SessionTransport
from speech_to_speech.LLM.chat import Chat, ResponsesAssistantMessage, make_assistant_message, make_user_message
from speech_to_speech.pipeline.events import AssistantOutputEvent, AssistantResponseDoneEvent
from speech_to_speech.pipeline.history import ResponseHistory
from tests.openai_realtime.conftest import make_pipeline_unit

RATE = 16_000
FIRST = "Sure, here are five ways to improve it."
SECOND = "First, tighten the thesis so every paragraph supports it."
RESPONSE_KEY = "response_essay"


def _speak(service, conn_id, parts_with_seconds, *, complete=True):
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
            AssistantOutputEvent(text=text, response_key=RESPONSE_KEY, history=history),
        )
        item_id = item_id or next(event.item_id for event in events if hasattr(event, "item_id"))
        for _ in range(int(seconds * 10)):
            service.encode_audio_chunk(conn_id, b"\x00\x00" * (RATE // 10), RESPONSE_KEY)
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
    assert chat.buffer[1].id == "msg_1"
    assert chat.buffer[1].content[0].text == "Sure, here"
    request = chat.to_responses_api_chat()
    assert all(item.get("type") != "reasoning" for item in request)
    assert "five ways" not in str(request)


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

    # The stock Agents SDK answers conversation.item.truncated with an
    # unsupported conversation.item.retrieve, so the server stays silent.
    assert transport.sent == []
    assert _history(unit.service, conn_id)[-1] == ("assistant", "incomplete", "Sure, here are")
