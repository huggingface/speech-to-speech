"""Realtime clients can keep the conversation going while a tool call is running (#760)."""

import pytest
from openai.types.realtime import (
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    RealtimeErrorEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
)
from openai.types.realtime.conversation_item import RealtimeConversationItemFunctionCall

from speech_to_speech.LLM.chat import LATE_TOOL_RESULT, PENDING_TOOL_OUTPUT, make_assistant_message
from speech_to_speech.pipeline.events import AssistantOutputEvent, TranscriptionCompletedEvent
from speech_to_speech.pipeline.messages import AssistantToolCallPart


def _create_item(service, conn_id, item):
    return service.handle_conversation_item_create(
        conn_id, ConversationItemCreateEvent.model_validate({"type": "conversation.item.create", "item": item})
    )


def _user_text(text):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _tool_output(call_id, output):
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _respond(service, conn_id, prompts):
    """Open a response the way a client does and return the request the LM receives."""
    created = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
    assert isinstance(created, ResponseCreatedEvent), created
    return prompts.get_nowait()


def _finish_with_calls(service, conn_id, request, *calls):
    chat = request.runtime_config.chat
    chat.add_provisional_generation_items(request.response_key, list(calls))
    service.dispatch_pipeline_event(
        conn_id,
        AssistantOutputEvent(
            response_key=request.response_key,
            parts=[AssistantToolCallPart(tool=call.model_dump()) for call in calls],
        ),
    )
    service.finish_response(conn_id, response_key=request.response_key)


def _finish_with_text(service, conn_id, request, text):
    request.runtime_config.chat.add_provisional_generation_items(request.response_key, [make_assistant_message(text)])
    service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text=text, response_key=request.response_key))
    service.finish_response(conn_id, response_key=request.response_key)


def _call(call_id, name="research_report", arguments='{"topic": "batteries"}'):
    return RealtimeConversationItemFunctionCall(
        type="function_call", id=f"fc_{call_id}", call_id=call_id, name=name, arguments=arguments
    )


def _roles(chat):
    return [message["role"] for message in chat.to_transformers_chat()]


def test_text_client_continues_while_a_tool_runs_and_gets_the_late_result(service, conn_id, text_prompt_queue):
    chat = service._state(conn_id).runtime_config.chat
    assert isinstance(
        _create_item(service, conn_id, _user_text("Make a report on batteries."))[0], ConversationItemCreatedEvent
    )
    _finish_with_calls(service, conn_id, _respond(service, conn_id, text_prompt_queue), _call("call_report"))
    assert chat.has_pending_tool_calls()

    # The tool has not returned. An unrelated question is answered anyway.
    _create_item(service, conn_id, _user_text("What is the capital of Australia?"))
    question = _respond(service, conn_id, text_prompt_queue)
    assert _roles(chat) == ["user", "assistant", "tool", "user"]
    assert chat.to_transformers_chat()[2]["content"] == PENDING_TOOL_OUTPUT
    _finish_with_text(service, conn_id, question, "Canberra.")

    # The result is accepted once, whenever it arrives, and starts nothing by itself.
    accepted = _create_item(service, conn_id, _tool_output("call_report", "the report"))
    assert isinstance(accepted[0], ConversationItemCreatedEvent)
    assert not chat.has_pending_tool_calls()
    assert text_prompt_queue.empty()

    # The client's follow-up sees the result as the newest thing in the conversation.
    _respond(service, conn_id, text_prompt_queue)
    assert _roles(chat) == ["user", "assistant", "tool", "user", "assistant", "user"]
    assert chat.to_transformers_chat()[-1]["content"] == LATE_TOOL_RESULT.format(
        name="research_report", arguments='{"topic": "batteries"}', output="the report"
    )


def test_spoken_turn_is_generated_while_a_tool_runs(service, conn_id, text_prompt_queue):
    chat = service._state(conn_id).runtime_config.chat
    _create_item(service, conn_id, _user_text("Make a report on batteries."))
    _finish_with_calls(service, conn_id, _respond(service, conn_id, text_prompt_queue), _call("call_report"))

    service.dispatch_pipeline_event(
        conn_id,
        TranscriptionCompletedEvent(transcript="What is the capital of Australia?", turn_id="t", turn_revision=0),
    )

    request = text_prompt_queue.get_nowait()
    assert request.runtime_config.chat is chat
    assert _roles(chat) == ["user", "assistant", "tool", "user"]
    assert chat.has_pending_tool_calls()


def test_follow_up_for_part_of_a_parallel_batch_waits_for_the_rest(service, conn_id, text_prompt_queue):
    """Clients that request a follow-up per tool result still get one reply for the batch."""
    chat = service._state(conn_id).runtime_config.chat
    _create_item(service, conn_id, _user_text("Weather in Paris and Rome?"))
    _finish_with_calls(
        service,
        conn_id,
        _respond(service, conn_id, text_prompt_queue),
        _call("call_paris", "get_weather", '{"city": "Paris"}'),
        _call("call_rome", "get_weather", '{"city": "Rome"}'),
    )

    _create_item(service, conn_id, _tool_output("call_rome", "sunny"))
    early = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
    assert isinstance(early, RealtimeErrorEvent)
    assert early.error.type == "function_call_output_pending"
    assert text_prompt_queue.empty()

    _create_item(service, conn_id, _tool_output("call_paris", "rain"))
    _respond(service, conn_id, text_prompt_queue)
    assert _roles(chat) == ["user", "assistant", "tool", "assistant", "tool"]
    assert PENDING_TOOL_OUTPUT not in [message["content"] for message in chat.to_transformers_chat()]


def test_follow_up_is_not_held_back_by_a_call_the_user_has_spoken_after(service, conn_id, text_prompt_queue):
    chat = service._state(conn_id).runtime_config.chat
    _create_item(service, conn_id, _user_text("Make a report on batteries."))
    _finish_with_calls(service, conn_id, _respond(service, conn_id, text_prompt_queue), _call("call_report"))

    # While the report runs, the user asks for something a quick tool answers.
    _create_item(service, conn_id, _user_text("And the weather in Paris?"))
    _finish_with_calls(
        service,
        conn_id,
        _respond(service, conn_id, text_prompt_queue),
        _call("call_paris", "get_weather", '{"city": "Paris"}'),
    )
    _create_item(service, conn_id, _tool_output("call_paris", "rain"))

    _respond(service, conn_id, text_prompt_queue)
    assert chat.has_pending_tool_calls()
    shown = chat.to_transformers_chat()
    assert [message["content"] for message in shown if message["role"] == "tool"] == [PENDING_TOOL_OUTPUT, "rain"]


@pytest.mark.parametrize("status", ["cancelled", "failed", "incomplete", "completed"])
def test_old_tool_result_is_not_duplicated_when_the_reply_it_arrived_during_ends(
    service, conn_id, text_prompt_queue, status
):
    from speech_to_speech.pipeline.events import ResponseGenerationDoneEvent

    chat = service._state(conn_id).runtime_config.chat
    _create_item(service, conn_id, _user_text("Make a report."))
    _finish_with_calls(service, conn_id, _respond(service, conn_id, text_prompt_queue), _call("call_report"))
    _create_item(service, conn_id, _user_text("What is the capital of Australia?"))
    request = _respond(service, conn_id, text_prompt_queue)
    chat.add_provisional_generation_items(request.response_key, [make_assistant_message("Canberra.")])
    _create_item(service, conn_id, _tool_output("call_report", "REPORT FINISHED"))
    service.dispatch_pipeline_event(
        conn_id, ResponseGenerationDoneEvent(response_key=request.response_key, call_ids=[])
    )
    assert len([item for item in chat.buffer if item.type == "function_call_output"]) == 1

    service.finish_response(conn_id, status=status, response_key=request.response_key)

    after = [item for item in chat.buffer if item.type == "function_call_output"]
    assert len(after) == 1, [(item.id, item.call_id) for item in after]


def test_late_result_stays_after_a_separate_progress_reply(service, conn_id, text_prompt_queue):
    import json

    chat = service._state(conn_id).runtime_config.chat
    _create_item(service, conn_id, _user_text("Make a report."))
    _finish_with_calls(service, conn_id, _respond(service, conn_id, text_prompt_queue), _call("call_report"))
    progress = _respond(service, conn_id, text_prompt_queue)
    _finish_with_text(service, conn_id, progress, "The report is still running.")
    _create_item(service, conn_id, _tool_output("call_report", "REPORT FINISHED"))
    _respond(service, conn_id, text_prompt_queue)

    for view in (chat.to_transformers_chat(), chat.to_responses_api_chat()):
        rendered = json.dumps(view)
        assert rendered.index("The report is still running.") < rendered.index("REPORT FINISHED"), view
