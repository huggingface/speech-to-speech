"""Realtime conversation continues while client function calls are pending (#760)."""

from types import SimpleNamespace

import pytest
from openai.types.realtime import (
    ConversationItemCreateEvent,
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
    ResponseCreateEvent,
)
from starlette.testclient import TestClient

from speech_to_speech.LLM.chat import make_assistant_message, make_user_message
from speech_to_speech.pipeline.events import (
    ResponseFailedEvent,
    ResponseGenerationDoneEvent,
    SpeechStartedEvent,
)
from speech_to_speech.pipeline.messages import EndOfResponse
from tests.llm_history import drive_llm
from tests.test_response_overrides import _RecordingLocalHandler

from .test_websocket_router import setup as setup


def _call(call_id: str) -> RealtimeConversationItemFunctionCall:
    return RealtimeConversationItemFunctionCall(type="function_call", call_id=call_id, name="slow", arguments="{}")


def _output_event(call_id: str, output: str = "done") -> ConversationItemCreateEvent:
    return ConversationItemCreateEvent(
        type="conversation.item.create",
        item={"type": "function_call_output", "call_id": call_id, "output": output},
    )


def _output(call_id: str, output: str) -> RealtimeConversationItemFunctionCallOutput:
    return RealtimeConversationItemFunctionCallOutput(type="function_call_output", call_id=call_id, output=output)


def _user_input(text: str) -> dict:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _start_unrelated_response(service, conn_id, text_prompt_queue):
    event = ResponseCreateEvent(type="response.create", response={"input": [_user_input("unrelated question")]})
    assert service.handle_response_create(conn_id, event).type == "response.created"
    return text_prompt_queue.get_nowait()


def test_response_create_proceeds_while_a_call_is_pending(service, conn_id, text_prompt_queue):
    chat = service._state(conn_id).runtime_config.chat
    chat.add_ordered_function_call(_call("call_slow"))

    _start_unrelated_response(service, conn_id, text_prompt_queue)

    assert chat.has_pending_tool_calls()


def test_multiple_pending_calls_resolve_out_of_order(service, conn_id):
    chat = service._state(conn_id).runtime_config.chat
    chat.add_ordered_function_call(_call("call_a"))
    chat.add_ordered_function_call(_call("call_b"))

    service.handle_conversation_item_create(conn_id, _output_event("call_b"))
    assert chat.has_pending_tool_calls()

    service.handle_conversation_item_create(conn_id, _output_event("call_a"))
    assert not chat.has_pending_tool_calls()
    assert sorted(chat.tool_output_call_ids()) == ["call_a", "call_b"]


def test_late_result_is_paired_after_an_unrelated_turn(service, conn_id):
    chat = service._state(conn_id).runtime_config.chat
    chat.add_item(make_user_message("run the slow tool"))
    chat.add_ordered_function_call(_call("call_slow"))
    chat.add_item(make_user_message("unrelated question"))

    service.handle_conversation_item_create(conn_id, _output_event("call_slow", "late result"))

    assert not chat.has_pending_tool_calls()
    text = str(chat.to_transformers_chat())
    assert "late result" in text
    assert "call_slow" in text


def test_duplicate_result_for_one_call_is_rejected(service, conn_id):
    chat = service._state(conn_id).runtime_config.chat
    chat.add_ordered_function_call(_call("call_slow"))
    service.handle_conversation_item_create(conn_id, _output_event("call_slow", "first"))

    events = service.handle_conversation_item_create(conn_id, _output_event("call_slow", "second"))

    assert [e.type for e in events] == ["error"]
    assert chat.tool_output_call_ids() == ["call_slow"]


def test_result_arriving_mid_response_is_deferred_then_paired_once(service, conn_id, text_prompt_queue):
    st = service._state(conn_id)
    chat = st.runtime_config.chat
    chat.add_ordered_function_call(_call("call_slow"))
    request = _start_unrelated_response(service, conn_id, text_prompt_queue)
    st.in_response = True
    st.current_response_key = request.response_key

    assert service.handle_conversation_item_create(conn_id, _output_event("call_slow", "late")) == []
    assert chat.has_pending_tool_calls()
    assert len(st.deferred_items) == 1

    service.finish_response(conn_id, response_key=request.response_key)

    assert not chat.has_pending_tool_calls()
    assert chat.tool_output_call_ids() == ["call_slow"]
    # The late result does not start a follow-up by itself.
    assert text_prompt_queue.empty()


def test_unrelated_turn_finishing_does_not_start_a_followup(service, conn_id, text_prompt_queue):
    st = service._state(conn_id)
    st.runtime_config.chat.add_ordered_function_call(_call("call_slow"))
    request = _start_unrelated_response(service, conn_id, text_prompt_queue)

    service.dispatch_pipeline_event(
        conn_id, ResponseGenerationDoneEvent(response_key=request.response_key, call_ids=[])
    )
    service.finish_response(conn_id, response_key=request.response_key)

    assert text_prompt_queue.empty()
    assert st.runtime_config.chat.has_pending_tool_calls()


def _cancel(service, conn_id, st, request):
    service.handle_response_cancel(conn_id)


def _fail(service, conn_id, st, request):
    service.dispatch_pipeline_event(conn_id, ResponseFailedEvent(message="boom", response_key=request.response_key))
    service.finish_response(conn_id, response_key=request.response_key)


def _barge_in(service, conn_id, st, request):
    st.in_response = True
    st.current_response_key = request.response_key
    service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(turn_id="turn_2", turn_revision=0))
    assert not st.in_response


@pytest.mark.parametrize("interrupt", [_cancel, _fail, _barge_in], ids=["cancel", "failure", "barge_in"])
def test_interrupted_unrelated_response_keeps_the_pending_call(service, conn_id, text_prompt_queue, interrupt):
    st = service._state(conn_id)
    chat = st.runtime_config.chat
    chat.add_ordered_function_call(_call("call_slow"))
    request = _start_unrelated_response(service, conn_id, text_prompt_queue)

    interrupt(service, conn_id, st, request)

    assert chat.has_pending_tool_calls()
    service.handle_conversation_item_create(conn_id, _output_event("call_slow", "late"))
    assert not chat.has_pending_tool_calls()
    assert text_prompt_queue.empty()


def test_issue_760_sequence_over_the_websocket(setup):
    """Pending tool, unrelated turn on the default conversation, then the late result."""
    app, service, *_ = setup
    with TestClient(app) as client, client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        (conn_id,) = service._conns
        service._state(conn_id).runtime_config.chat.add_ordered_function_call(_call("call_slow"))

        ws.send_json({"type": "conversation.item.create", "item": _user_input("unrelated question")})
        assert ws.receive_json()["type"] == "conversation.item.created"
        ws.send_json({"type": "response.create", "response": {"output_modalities": ["text"]}})
        assert ws.receive_json()["type"] == "response.created"

        request = service.text_prompt_queue.get(timeout=1)
        handler = object.__new__(_RecordingLocalHandler)
        handler.cancel_scope = handler.speculative_turns = handler.compactor = None
        handler.enable_lang_prompt = False
        handler.tokenizer = SimpleNamespace(encode=lambda _text: [])
        handler.emit_text = True
        ends = [chunk for chunk in drive_llm(handler, request) if isinstance(chunk, EndOfResponse)]
        assert ends and ends[0].error is None
        assert "unrelated question" in str(handler.seen_chat.to_transformers_chat())
        assert "call_slow" not in str(handler.seen_chat.to_transformers_chat())

        ws.send_json({"type": "response.cancel"})
        assert ws.receive_json()["type"] == "response.done"

        result = {"type": "function_call_output", "call_id": "call_slow", "output": "late"}
        ws.send_json({"type": "conversation.item.create", "item": result})
        assert ws.receive_json()["type"] == "conversation.item.created"
        ws.send_json({"type": "conversation.item.create", "item": {**result, "output": "again"}})
        assert ws.receive_json()["type"] == "error"
        assert not service._state(conn_id).runtime_config.chat.has_pending_tool_calls()


def _assert_every_call_is_paired(chat) -> None:
    """The invariant strict providers enforce: every tool output follows its call, and no call is left bare."""
    messages = chat.to_transformers_chat()
    calls = [c["id"] for m in messages for c in m.get("tool_calls", [])]
    assert sorted(calls) == sorted(m["tool_call_id"] for m in messages if m["role"] == "tool")
    seen_calls: set[str] = set()
    outputs = []
    for item in chat.to_responses_api_chat():
        if item.get("type") == "function_call":
            seen_calls.add(item["call_id"])
        elif item.get("type") == "function_call_output":
            assert item["call_id"] in seen_calls
            outputs.append(item["call_id"])
    assert sorted(outputs) == sorted(seen_calls)


def test_provider_requests_stay_paired_through_every_pending_state(service, conn_id):
    chat = service._state(conn_id).runtime_config.chat
    chat.add_item(make_user_message("start two tools"))
    chat.add_ordered_function_call(_call("call_a"))
    chat.add_ordered_function_call(_call("call_b"))
    _assert_every_call_is_paired(chat)
    chat.add_item(make_user_message("unrelated question"))
    _assert_every_call_is_paired(chat)
    assert "unrelated question" in str(chat.to_transformers_chat())
    chat.add_item(make_assistant_message("an answer"))
    chat.add_item(_output("call_b", "b done"))
    _assert_every_call_is_paired(chat)
    assert chat.has_pending_tool_calls()
    chat.add_item(_output("call_a", "a done"))
    _assert_every_call_is_paired(chat)
    assert chat.pending_tool_calls_notice() is None


def test_failed_tool_output_resolves_only_its_own_call(service, conn_id):
    chat = service._state(conn_id).runtime_config.chat
    chat.add_ordered_function_call(_call("call_a"))
    chat.add_ordered_function_call(_call("call_b"))

    service.handle_conversation_item_create(conn_id, _output_event("call_a", '{"error": "RuntimeError: boom"}'))

    assert chat.has_pending_tool_calls()
    assert "call_a" not in chat.pending_tool_calls_notice()
    assert "boom" in str(chat.to_transformers_chat())
    _assert_every_call_is_paired(chat)


def test_followup_prefetch_waits_for_the_last_pending_call(service, conn_id, text_prompt_queue):
    st = service._state(conn_id)
    response_key = "response_origin"
    calls = [_call("call_a"), _call("call_b")]
    assert st.runtime_config.chat.add_provisional_generation_items(response_key, calls) is not None
    st.in_response = True
    st.current_response_id = "resp_origin"
    st.current_response_key = response_key
    st.pending_function_calls = dict(enumerate(calls))
    service.dispatch_pipeline_event(
        conn_id, ResponseGenerationDoneEvent(response_key=response_key, call_ids=["call_a", "call_b"])
    )

    service.handle_conversation_item_create(conn_id, _output_event("call_a"))
    assert text_prompt_queue.empty()
    assert st.tool_followup_prefetch_request is None

    service.handle_conversation_item_create(conn_id, _output_event("call_b"))
    prefetch = text_prompt_queue.get_nowait()
    assert st.tool_followup_prefetch_request is prefetch
    assert not st.runtime_config.chat.has_pending_tool_calls()
