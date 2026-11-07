"""Realtime conversation continues while client function calls await their outputs."""

import asyncio
from types import SimpleNamespace

import pytest
from openai.types.realtime import (
    ConversationItemCreateEvent,
    RealtimeConversationItemFunctionCall,
    ResponseCreateEvent,
)
from starlette.testclient import TestClient

from speech_to_speech.api.openai_realtime.audio_client import RealtimeAudioClientConfig, _ToolCallCoordinator
from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    ResponseFailedEvent,
    ResponseGenerationDoneEvent,
    SpeechStartedEvent,
)
from speech_to_speech.pipeline.messages import EndOfResponse
from tests.llm_history import drive_llm
from tests.openai_realtime.test_audio_client import TOOL_DEFINITION, response_done, wait_until
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


@pytest.mark.parametrize("status", ["completed", "cancelled", "failed", "incomplete"])
async def test_accepted_result_gets_client_followup_after_unrelated_response_ends(
    service, conn_id, text_prompt_queue, status
):
    st = service._state(conn_id)
    chat = st.runtime_config.chat
    call = _call("call_slow")
    chat.add_provisional_generation_items("origin", [call])
    chat.finalize_provisional_generation("origin")
    release = asyncio.Event()

    async def execute(*_):
        await release.wait()
        return "real late result"

    class Connection:
        def __init__(self):
            self.sent = []

        async def send(self, event):
            self.sent.append(event)
            if event["type"] == "conversation.item.create":
                for reply in service.handle_conversation_item_create(
                    conn_id, ConversationItemCreateEvent.model_validate(event)
                ):
                    coordinator.handle_event(reply)

    connection = Connection()
    coordinator = _ToolCallCoordinator(
        connection, RealtimeAudioClientConfig(tools=[TOOL_DEFINITION], tool_executor=execute)
    )
    try:
        coordinator.handle_event(response_done("origin", output=[call]))
        created = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
        coordinator.handle_event(created)
        request = text_prompt_queue.get_nowait()
        for event in service.dispatch_pipeline_event(
            conn_id, AssistantOutputEvent(text="An unrelated answer.", response_key=request.response_key)
        ):
            coordinator.handle_event(event)
        release.set()
        await wait_until(lambda: bool(st.deferred_items))
        for event in service.dispatch_pipeline_event(
            conn_id, ResponseGenerationDoneEvent(response_key=request.response_key, call_ids=[])
        ):
            coordinator.handle_event(event)
        assert chat.tool_output_call_ids() == [call.call_id]
        assert len(st.pending_item_acks) == 1
        assert [event["type"] for event in connection.sent] == ["conversation.item.create"]

        events = service.finish_response(conn_id, status=status, response_key=request.response_key)
        assert not any(event.type == "error" for event in events)
        for event in events:
            coordinator.handle_event(event)
        await wait_until(lambda: any(event["type"] == "response.create" for event in connection.sent))
        assert len([item for item in chat.buffer if item.type == "function_call_output"]) == 1
        assert not coordinator._pending_outputs
        assert not coordinator._failure.done()
        assert [event["type"] for event in connection.sent] == ["conversation.item.create", "response.create"]
    finally:
        await coordinator.close()


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

        result = {"type": "function_call_output", "call_id": "call_slow", "output": "late"}
        ws.send_json({"type": "conversation.item.create", "event_id": "result_first", "item": result})
        ws.send_json(
            {"type": "conversation.item.create", "event_id": "result_duplicate", "item": {**result, "output": "again"}}
        )
        ws.send_json({"type": "response.cancel", "event_id": "cancel_unrelated"})
        assert ws.receive_json()["type"] == "response.done"
        assert ws.receive_json()["type"] == "conversation.item.created"
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"]["event_id"] == "result_duplicate"
        assert not service._state(conn_id).runtime_config.chat.has_pending_tool_calls()
