"""Realtime conversation continues while client function calls await their outputs."""

from types import SimpleNamespace

import pytest
from openai.types.realtime import (
    ConversationItemCreateEvent,
    RealtimeConversationItemFunctionCall,
    ResponseCreateEvent,
)
from starlette.testclient import TestClient

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


def _user_input(text: str) -> dict:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _start_unrelated_response(service, conn_id, text_prompt_queue):
    event = ResponseCreateEvent(type="response.create", response={"input": [_user_input("unrelated question")]})
    assert service.handle_response_create(conn_id, event).type == "response.created"
    return text_prompt_queue.get_nowait()


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
        assert "awaiting_output" in str(handler.seen_chat.to_transformers_chat())
        assert not any("tool_calls" in item for item in handler.seen_chat.to_transformers_chat())

        ws.send_json({"type": "response.cancel"})
        assert ws.receive_json()["type"] == "response.done"

        result = {"type": "function_call_output", "call_id": "call_slow", "output": "late"}
        ws.send_json({"type": "conversation.item.create", "item": result})
        assert ws.receive_json()["type"] == "conversation.item.created"
        ws.send_json({"type": "conversation.item.create", "item": {**result, "output": "again"}})
        assert ws.receive_json()["type"] == "error"
        assert not service._state(conn_id).runtime_config.chat.has_pending_tool_calls()
