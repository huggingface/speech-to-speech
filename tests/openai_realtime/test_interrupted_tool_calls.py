"""A cancelled response keeps the function calls its client is already running."""

from openai.types.realtime import ConversationItemCreatedEvent, ConversationItemCreateEvent, RealtimeErrorEvent
from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.LLM.chat import make_user_message
from speech_to_speech.LLM.language_model import BaseLanguageModelHandler
from speech_to_speech.pipeline.events import AssistantOutputEvent
from speech_to_speech.pipeline.history import ResponseHistory
from speech_to_speech.pipeline.messages import AssistantTextPart, AssistantToolCallPart

RATE = 16_000
SPEECH = "Sure, watch this."
RESPONSE_KEY = "response_dance"


def _call_tool_while_speaking(service, conn_id, *, status=None, text_after_call=None):
    """Deliver speech and a function call, then leave the response in progress."""
    chat = service._state(conn_id).runtime_config.chat
    chat.add_item(make_user_message("Dance for me"))
    parts = [
        AssistantTextPart(text=SPEECH),
        AssistantToolCallPart(
            tool=ResponseFunctionToolCall(
                type="function_call", id="fc_dance", call_id="call_dance", name="dance", arguments="{}", status=status
            )
        ),
    ]
    if text_after_call is not None:
        parts.append(AssistantTextPart(text=text_after_call))
    history = ResponseHistory.capture(
        chat,
        BaseLanguageModelHandler._ordered_output_items(parts, wants_audio=True),
        after_item_id=chat.history_anchor_id(),
    )
    for part in parts:
        service.dispatch_pipeline_event(
            conn_id, AssistantOutputEvent(parts=[part], response_key=RESPONSE_KEY, history=history)
        )
        if isinstance(part, AssistantTextPart):
            for _ in range(20):
                service.encode_audio_chunk(conn_id, b"\x00\x00" * (RATE // 10), RESPONSE_KEY)
    return chat


def _send_output(service, conn_id):
    return service.handle_conversation_item_create(
        conn_id,
        ConversationItemCreateEvent(
            type="conversation.item.create",
            item={"type": "function_call_output", "call_id": "call_dance", "output": '{"status": "ok"}'},
        ),
    )


def _model_input(chat):
    return [(item["type"], item.get("role")) for item in chat.to_responses_api_chat()]


def test_barge_in_keeps_a_delivered_call_and_accepts_its_output(service, conn_id):
    chat = _call_tool_while_speaking(service, conn_id)

    service.finish_response(conn_id, status="cancelled", reason="turn_detected")

    # The heard speech stays ahead of the call it preceded.
    assert [item.type for item in chat.buffer] == ["message", "message", "function_call"]
    assert (chat.buffer[1].status, chat.buffer[1].content[0].text) == ("incomplete", SPEECH)
    # The interrupting turn must get a reply even though the tool is still running.
    assert not chat.has_pending_tool_calls()
    assert not chat.copy(deep=True).has_pending_tool_calls()
    assert _model_input(chat) == [("message", "user"), ("message", "assistant")]

    events = _send_output(service, conn_id)

    assert [type(event) for event in events] == [ConversationItemCreatedEvent]
    assert _model_input(chat) == [
        ("message", "user"),
        ("message", "assistant"),
        ("function_call", None),
        ("function_call_output", None),
    ]


def test_output_sent_before_the_barge_in_is_kept_once(service, conn_id):
    chat = _call_tool_while_speaking(service, conn_id)
    assert _send_output(service, conn_id) == []  # Deferred behind the active response.

    events = service.finish_response(conn_id, status="cancelled", reason="turn_detected")

    assert not any(isinstance(event, RealtimeErrorEvent) for event in events)
    assert [event.item.type for event in events if isinstance(event, ConversationItemCreatedEvent)] == [
        "function_call_output"
    ]
    assert [item.type for item in chat.buffer] == ["message", "message", "function_call", "function_call_output"]


def test_speech_after_a_kept_call_stays_after_it(service, conn_id):
    chat = _call_tool_while_speaking(service, conn_id, text_after_call="Here I go.")

    service.finish_response(conn_id, status="cancelled", reason="turn_detected")

    assert [item.type for item in chat.buffer] == ["message", "message", "function_call", "message"]
    assert [chat.buffer[index].content[0].text for index in (1, 3)] == [SPEECH, "Here I go."]


def test_call_the_client_never_received_complete_is_still_rolled_back(service, conn_id):
    chat = _call_tool_while_speaking(service, conn_id, status="in_progress")

    service.finish_response(conn_id, status="cancelled", reason="turn_detected")

    assert [item.type for item in chat.buffer] == ["message", "message"]
    assert [type(event) for event in _send_output(service, conn_id)] == [RealtimeErrorEvent]


def test_failed_response_still_rolls_back_its_call(service, conn_id):
    chat = _call_tool_while_speaking(service, conn_id)

    service.finish_response(conn_id, status="failed")

    assert [item.type for item in chat.buffer] == ["message"]
