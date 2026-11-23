"""A cancelled response keeps the function calls its client is already running."""

from openai.types.realtime import (
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    RealtimeErrorEvent,
    ResponseDoneEvent,
    ResponseOutputItemDoneEvent,
)
from openai.types.realtime.conversation_item import RealtimeConversationItemFunctionCall
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.LLM.chat import LATE_TOOL_RESULT, PENDING_TOOL_OUTPUT, make_user_message
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


def _send_output(service, conn_id, call_id="call_dance", output="danced"):
    return service.handle_conversation_item_create(
        conn_id,
        ConversationItemCreateEvent(
            type="conversation.item.create",
            item={"type": "function_call_output", "call_id": call_id, "output": output},
        ),
    )


def _model_input(chat):
    """What the model is shown: each message's role and its text, or the tool it calls.

    Spoken text and the call it led into share one assistant message; the text is
    listed first.
    """
    shown = []
    for message in chat.to_transformers_chat():
        if "tool_calls" in message:
            if message["content"]:
                shown.append((message["role"], message["content"]))
            shown.append((message["role"], message["tool_calls"][0]["function"]["name"]))
        else:
            shown.append((message["role"], message["content"].split("\n[Tool status")[0]))
    return shown


def test_barge_in_keeps_a_delivered_call_and_accepts_its_output(service, conn_id):
    chat = _call_tool_while_speaking(service, conn_id)

    service.finish_response(conn_id, status="cancelled", reason="turn_detected")

    # The heard speech stays ahead of the call it preceded.
    assert [item.type for item in chat.buffer] == ["message", "message", "function_call"]
    assert (chat.buffer[1].status, chat.buffer[1].content[0].text) == ("incomplete", SPEECH)
    heard = [("user", "Dance for me"), ("assistant", SPEECH), ("assistant", "dance")]
    assert _model_input(chat) == [*heard, ("tool", PENDING_TOOL_OUTPUT)]

    events = _send_output(service, conn_id)

    assert [type(event) for event in events] == [ConversationItemCreatedEvent]
    assert _model_input(chat) == [*heard, ("tool", "danced")]


def test_barge_in_while_an_earlier_call_is_still_running(service, conn_id):
    """The cancelled reply was said past the earlier call, but not past its own."""
    chat = service._state(conn_id).runtime_config.chat
    chat.add_item(make_user_message("Make a report"))
    report = RealtimeConversationItemFunctionCall(
        type="function_call", id="fc_report", call_id="call_report", name="report", arguments="{}"
    )
    chat.add_provisional_generation_items("response_report", [report])
    service.dispatch_pipeline_event(
        conn_id,
        AssistantOutputEvent(response_key="response_report", parts=[AssistantToolCallPart(tool=report.model_dump())]),
    )
    service.finish_response(conn_id, response_key="response_report")
    _call_tool_while_speaking(service, conn_id, text_after_call="Here I go.")

    service.finish_response(conn_id, status="cancelled", reason="turn_detected")
    assert [type(event) for event in _send_output(service, conn_id)] == [ConversationItemCreatedEvent]
    assert [type(event) for event in _send_output(service, conn_id, "call_report", "the report")] == [
        ConversationItemCreatedEvent
    ]

    assert _model_input(chat) == [
        ("user", "Make a report"),
        ("assistant", "report"),
        ("tool", PENDING_TOOL_OUTPUT),
        ("user", "Dance for me"),
        ("assistant", SPEECH),
        ("assistant", "dance"),
        ("tool", "danced"),
        ("assistant", "Here I go."),
        ("user", LATE_TOOL_RESULT.format(name="report", arguments="{}", output="the report")),
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

    # Speech after its own call does not make that call's result a late one.
    _send_output(service, conn_id)
    assert _model_input(chat)[2:] == [("assistant", "dance"), ("tool", "danced"), ("assistant", "Here I go.")]


def test_call_from_a_local_model_is_kept_and_listed_as_completed(service, conn_id):
    """Local models' calls stay in progress until the response ends, but the client already runs them."""
    chat = _call_tool_while_speaking(service, conn_id, status="in_progress")

    events = service.finish_response(conn_id, status="cancelled", reason="turn_detected")

    done = next(event for event in events if isinstance(event, ResponseDoneEvent))
    closed = next(event for event in events if isinstance(event, ResponseOutputItemDoneEvent))
    assert [item.status for item in done.response.output if item.type == "function_call"] == ["completed"]
    assert (closed.item.call_id, closed.item.status) == ("call_dance", "completed")
    assert [item.type for item in chat.buffer] == ["message", "message", "function_call"]
    assert [type(event) for event in _send_output(service, conn_id)] == [ConversationItemCreatedEvent]


def test_call_of_a_cancelled_out_of_band_response_is_listed_as_incomplete(service, conn_id):
    """Nothing of an out-of-band response is in the conversation, so no output is expected."""
    service._state(conn_id).current_response_params = RealtimeResponseCreateParams(conversation="none")
    service.dispatch_pipeline_event(
        conn_id,
        AssistantOutputEvent(
            response_key=RESPONSE_KEY,
            parts=[
                AssistantToolCallPart(
                    tool=ResponseFunctionToolCall(
                        type="function_call",
                        id="fc_dance",
                        call_id="call_dance",
                        name="dance",
                        arguments="{}",
                        status="in_progress",
                    )
                )
            ],
        ),
    )

    events = service.finish_response(conn_id, status="cancelled", reason="client_cancelled")

    done = next(event for event in events if isinstance(event, ResponseDoneEvent))
    assert [item.status for item in done.response.output] == ["incomplete"]


def test_failed_response_still_rolls_back_its_call(service, conn_id):
    chat = _call_tool_while_speaking(service, conn_id)

    service.finish_response(conn_id, status="failed")

    assert [item.type for item in chat.buffer] == ["message"]
