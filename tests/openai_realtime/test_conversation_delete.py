"""Realtime history deletion through Chat, the service, and the wire route."""

from types import SimpleNamespace

import pytest
from openai.types.realtime import (
    ConversationItemCreateEvent,
    ConversationItemDeleteEvent,
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
    ResponseCreateEvent,
)
from starlette.testclient import TestClient

from speech_to_speech.LLM.chat import (
    Chat,
    CompactionResult,
    make_assistant_message,
    make_system_message,
    make_user_message,
)
from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.messages import AssistantTextPart, EndOfResponse
from tests.test_response_overrides import _RecordingLocalHandler

from .test_websocket_router import setup as setup


def _call():
    return RealtimeConversationItemFunctionCall(
        type="function_call", id="client_call", call_id="call_test", name="lookup", arguments="{}"
    )


def _output():
    return RealtimeConversationItemFunctionCallOutput(
        type="function_call_output", id="client_output", call_id="call_test", output="result"
    )


@pytest.mark.parametrize("kind", ["system", "user", "assistant", "function_call", "function_call_output"])
def test_delete_supported_item(service, conn_id, kind):
    chat = service._state(conn_id).runtime_config.chat
    if kind == "function_call_output":
        chat.add_item(_call())
        item = _output()
    elif kind == "function_call":
        item = _call()
    else:
        item = {"system": make_system_message, "user": make_user_message, "assistant": make_assistant_message}[kind](
            "delete me"
        )
    chat.add_item(item)
    item_id = item.id
    events = service.handle_conversation_item_delete(
        conn_id, ConversationItemDeleteEvent(type="conversation.item.delete", item_id=item_id, event_id="client_delete")
    )
    assert [(e.type, e.item_id) for e in events] == [("conversation.item.deleted", item_id)]
    assert events[0].event_id.startswith("event_")
    assert chat.delete_item(item_id) is None
    assert "delete me" not in str(chat.to_transformers_chat())
    missing = service.handle_conversation_item_delete(
        conn_id,
        ConversationItemDeleteEvent(type="conversation.item.delete", item_id=item_id, event_id="missing_delete"),
    )
    assert missing[0].type == "error"
    assert missing[0].error.event_id == "missing_delete"


def test_delete_call_preserves_output_but_omits_orphan_from_model():
    chat = Chat(10)
    chat.add_item(_call())
    output = chat.add_item(_output())
    assert chat.delete_item("client_call").type == "function_call"
    assert chat.buffer == [output]
    assert not chat.has_pending_tool_calls()
    assert chat.to_transformers_chat() == []
    assert chat.to_responses_api_chat() == []
    assert chat.copy(deep=True).to_transformers_chat() == []
    assert chat.copy_without_provisional_generation("unused").to_responses_api_chat() == []
    assert chat.delete_item(output.id) is output


def test_delete_output_requires_replacement_before_response(service, conn_id):
    chat = service._state(conn_id).runtime_config.chat
    chat.add_item(_call())
    chat.add_item(_output())
    assert chat.delete_item("client_output").type == "function_call_output"
    assert chat.has_pending_tool_calls()
    result = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
    assert result.type == "error"
    assert result.error.type == "function_call_output_pending"
    chat.add_item(_output())
    assert not chat.has_pending_tool_calls()
    assert (
        service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create")).type == "response.created"
    )


def test_delete_invalidates_compaction_snapshot():
    chat = Chat(10)
    first = chat.add_item(make_user_message("private detail"))
    second = chat.add_item(make_user_message("keep this"))
    generation = chat._gen_counter
    chat.delete_item(first.id)
    chat._apply_compaction(
        CompactionResult(user_summary="private detail", assistant_summary="old summary"), {first.id}, generation
    )
    assert chat.buffer == [second]
    assert chat._user_turn_count == 1


def test_active_create_delete_replace_preserves_order_and_error_correlation(service, conn_id):
    st = service._state(conn_id)
    service.response._ensure_response(conn_id, "active")
    for item_id, text in [("context_1", "old"), ("context_2", "new")]:
        item = make_system_message(text)
        item.id = item_id
        assert (
            service.handle_conversation_item_create(
                conn_id, ConversationItemCreateEvent(type="conversation.item.create", item=item)
            )
            == []
        )
        if item_id == "context_1":
            assert (
                service.handle_conversation_item_delete(
                    conn_id, ConversationItemDeleteEvent(type="conversation.item.delete", item_id=item_id)
                )
                == []
            )
            assert (
                service.handle_conversation_item_delete(
                    conn_id,
                    ConversationItemDeleteEvent(type="conversation.item.delete", item_id=item_id, event_id="repeated"),
                )
                == []
            )
    events = service.handle_response_cancel(conn_id)
    history_events = [e for e in events if e.type.startswith("conversation.item.") or e.type == "error"]
    assert [e.type for e in history_events] == [
        "conversation.item.created",
        "conversation.item.deleted",
        "error",
        "conversation.item.created",
    ]
    assert history_events[2].error.event_id == "repeated"
    assert history_events[-1].previous_item_id is None
    assert st.runtime_config.chat.to_transformers_chat() == [{"role": "system", "content": "new"}]


def test_generated_assistant_is_deletable_by_wire_id(service, conn_id):
    st = service._state(conn_id)
    created = service.handle_response_create(
        conn_id, ResponseCreateEvent(type="response.create", response={"output_modalities": ["text"]})
    )
    request = service.text_prompt_queue.get_nowait()
    handler = object.__new__(_RecordingLocalHandler)
    handler.cancel_scope = handler.speculative_turns = handler.compactor = None
    handler.enable_lang_prompt = False
    handler.tokenizer = SimpleNamespace(encode=lambda _text: [])
    handler.emit_text = True
    list(handler.process(request))
    events = service.dispatch_pipeline_event(
        conn_id, AssistantOutputEvent(parts=[AssistantTextPart(text="ok")], response_key=request.response_key)
    )
    wire_item_id = next(e.item.id for e in events if e.type == "response.output_item.added")
    service.finish_response(conn_id)
    assert created.type == "response.created"
    assert st.runtime_config.chat.buffer[-1].id == wire_item_id
    assert (
        service.handle_conversation_item_delete(
            conn_id, ConversationItemDeleteEvent(type="conversation.item.delete", item_id=wire_item_id)
        )[0].type
        == "conversation.item.deleted"
    )
    assert not st.runtime_config.chat.buffer


def test_spoken_user_is_deletable_by_wire_id(service, conn_id):
    events = service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(turn_id="spoken", turn_revision=0))
    wire_item_id = next(e.item_id for e in events if e.type == "input_audio_buffer.speech_started")
    events = service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent(turn_id="spoken", turn_revision=0))
    events += service.dispatch_pipeline_event(
        conn_id, TranscriptionCompletedEvent(transcript="delete this speech", turn_id="spoken", turn_revision=0)
    )
    assert any(e.type == "conversation.item.created" and e.item.id == wire_item_id for e in events)
    assert service._state(conn_id).runtime_config.chat.buffer[-1].id == wire_item_id
    # A queued response finishes/cancels before the deletion is applied.
    assert (
        service.handle_conversation_item_delete(
            conn_id, ConversationItemDeleteEvent(type="conversation.item.delete", item_id=wire_item_id)
        )
        == []
    )
    assert any(e.type == "conversation.item.deleted" for e in service.handle_response_cancel(conn_id))
    assert not service._state(conn_id).runtime_config.chat.buffer
    # A late transcript must not restore the deleted turn or queue new work.
    queued = service.text_prompt_queue.qsize()
    assert (
        service.dispatch_pipeline_event(
            conn_id, TranscriptionCompletedEvent(transcript="late speech", turn_id="spoken", turn_revision=0)
        )
        == []
    )
    assert not service._state(conn_id).runtime_config.chat.buffer
    assert service.text_prompt_queue.qsize() == queued


def test_delete_tail_restores_staged_call_predecessor(service, conn_id):
    first = _call()
    second = _call().model_copy(update={"id": "second_call", "call_id": "call_second"})
    service.handle_conversation_item_create(
        conn_id, ConversationItemCreateEvent(type="conversation.item.create", item=first)
    )
    service.handle_conversation_item_create(
        conn_id, ConversationItemCreateEvent(type="conversation.item.create", item=second)
    )
    service.handle_conversation_item_delete(
        conn_id, ConversationItemDeleteEvent(type="conversation.item.delete", item_id=second.id)
    )
    event = service.handle_conversation_item_create(
        conn_id, ConversationItemCreateEvent(type="conversation.item.create", item=make_user_message("next"))
    )[0]
    assert event.previous_item_id == first.id


def test_openwebui_refresh_reaches_model_and_socket_stays_open(setup):
    app, service, *_ = setup
    with TestClient(app) as client, client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json({"type": "session.update", "session": {"type": "realtime", "instructions": "Be concise."}})
        assert ws.receive_json()["type"] == "session.updated"
        for revision in (1, 2):
            if revision == 2:
                ws.send_json({"type": "conversation.item.delete", "item_id": "chat_context_1"})
                deleted = ws.receive_json()
                assert deleted["type"] == "conversation.item.deleted"
                assert deleted["item_id"] == "chat_context_1"
            ws.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "id": f"chat_context_{revision}",
                        "type": "message",
                        "role": "system",
                        "content": [{"type": "input_text", "text": f"Snapshot {revision}"}],
                    },
                }
            )
            created = ws.receive_json()
            assert created["type"] == "conversation.item.created"
            assert created["previous_item_id"] is None
        ws.send_json({"type": "response.create", "response": {"output_modalities": ["text"]}})
        assert ws.receive_json()["type"] == "response.created"
        request = service.text_prompt_queue.get(timeout=1)
        handler = object.__new__(_RecordingLocalHandler)
        handler.cancel_scope = handler.speculative_turns = handler.compactor = None
        handler.enable_lang_prompt = False
        handler.tokenizer = SimpleNamespace(encode=lambda _text: [])
        handler.emit_text = True
        assert any(isinstance(chunk, EndOfResponse) for chunk in handler.process(request))
        assert "Snapshot 2" in str(handler.seen_chat.to_transformers_chat())
        assert "Be concise." in str(handler.seen_chat.to_transformers_chat())
        assert "Snapshot 1" not in str(handler.seen_chat.to_transformers_chat())
        ws.send_json({"type": "response.cancel"})
        assert ws.receive_json()["type"] == "response.done"
        ws.send_json({"type": "conversation.item.delete", "item_id": "missing", "event_id": "delete_error"})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"]["event_id"] == "delete_error"
        ws.send_json({"type": "session.update", "session": {"type": "realtime"}})
        assert ws.receive_json()["type"] == "session.updated"


def test_direct_audio_request_preserves_wire_identity(service, conn_id):
    import numpy as np

    from speech_to_speech.pipeline.events import AudioInputCompletedEvent

    started = service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(turn_id="audio", turn_revision=0))
    item_id = next(e.item_id for e in started if e.type == "input_audio_buffer.speech_started")
    service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent(turn_id="audio", turn_revision=0))
    service.dispatch_pipeline_event(
        conn_id,
        AudioInputCompletedEvent(
            audio=np.zeros(10, dtype=np.float32),
            audio_sample_rate=16000,
            audio_duration_s=0.001,
            turn_id="audio",
            turn_revision=0,
        ),
    )
    assert service.text_prompt_queue.get_nowait().input_item_id == item_id


def test_delete_discards_hidden_prefetch_and_late_write(service, conn_id):
    from speech_to_speech.pipeline.messages import GenerateResponseRequest, ResponsePrefetchTransaction

    st = service._state(conn_id)
    item = st.runtime_config.chat.add_item(make_system_message("old context"))
    request = GenerateResponseRequest(
        runtime_config=st.runtime_config, prefetch_transaction=ResponsePrefetchTransaction()
    )
    st.tool_followup_prefetch_request = request
    st.mark_response_pending(request.response_key)
    service.text_prompt_queue.put(request)
    events = service.handle_conversation_item_delete(
        conn_id, ConversationItemDeleteEvent(type="conversation.item.delete", item_id=item.id)
    )
    assert events[0].type == "conversation.item.deleted"
    assert st.tool_followup_prefetch_request is None
    assert not st.response_pending
    assert service.text_prompt_queue.empty()
    assert (
        st.runtime_config.chat.add_provisional_generation_items(request.response_key, [make_assistant_message("stale")])
        is None
    )
    assert st.runtime_config.chat.to_transformers_chat() == []


def test_generated_message_binding_preserves_pending_tool_context():
    chat = Chat(10)
    first = make_assistant_message("before")
    last = make_assistant_message("after")
    chat.add_provisional_generation_items("response", [first, _call(), last])
    chat.bind_assistant_item_ids("response", ["wire_first", "wire_last"])
    assert [item.id for item in chat.buffer] == ["wire_first", "client_call", "wire_last"]
    assert chat._ordered_pending_calls["call_test"] == {"wire_first", "client_call", "wire_last"}
    chat.rollback_provisional_generation("response")
    assert not chat.buffer
    assert not chat.has_pending_tool_calls()


def test_one_wire_message_deletes_all_retained_provider_fragments():
    chat = Chat(10)
    before = [make_assistant_message("first"), make_assistant_message("second")]
    after = make_assistant_message("after call")
    chat.add_provisional_generation_items("response", [*before, _call(), after])
    chat.bind_assistant_item_ids("response", ["wire_before", "wire_after"])
    chat.finalize_provisional_generation("response")
    assert "wire_before" in chat.item_ids()
    assert "wire_before" in chat.copy(deep=True).item_ids()
    assert chat.delete_item("wire_before").role == "assistant"
    assert chat.buffer == [chat._pending_tool_calls["call_test"], after]
    assert chat.delete_item("wire_before") is None
    assert chat.delete_item("wire_after") is after


@pytest.mark.parametrize("cancel_active", [False, True])
def test_delete_waits_for_overlapping_queued_response(service, conn_id, cancel_active):
    st = service._state(conn_id)
    original = st.runtime_config.chat.add_item(make_system_message("old"))
    service.response._ensure_response(conn_id, "active")
    st.mark_response_pending("queued")
    assert (
        service.handle_conversation_item_delete(
            conn_id, ConversationItemDeleteEvent(type="conversation.item.delete", item_id=original.id)
        )
        == []
    )
    if not cancel_active:
        assert not any(e.type == "conversation.item.deleted" for e in service.finish_response(conn_id))
        assert st.runtime_config.chat.init_chat_message is original
    events = service.handle_response_cancel(conn_id)
    assert any(e.type == "conversation.item.deleted" for e in events)
    assert not st.response_pending
    assert st.runtime_config.chat.init_chat_message is None
    assert st.runtime_config.chat.add_provisional_generation_items("queued", [make_assistant_message("late")]) is None


def test_deleting_provisional_call_does_not_remove_replacement_on_rollback():
    chat = Chat(10)
    chat.add_provisional_generation_items("response", [_call()])
    chat.delete_item("client_call")
    replacement = chat.add_item(_call().model_copy(update={"id": "replacement"}))
    chat.rollback_provisional_generation("response")
    assert chat._pending_tool_calls == {"call_test": replacement}


def test_evicted_message_does_not_steal_later_wire_identity():
    chat = Chat(1)
    chat.add_item(make_user_message("first turn"))
    before = make_assistant_message("before call")
    chat.add_provisional_generation_items("response", [before, _call()])
    chat.add_item(_output())
    chat.add_item(make_user_message("second turn"))
    chat.add_item(make_user_message("third turn"))
    assert before not in chat.buffer
    after = make_assistant_message("after call")
    chat.add_provisional_generation_items("response", [after])
    chat.bind_assistant_item_ids("response", ["wire_before", "wire_after"])
    chat.finalize_provisional_generation("response")
    assert after.id == "wire_after"
    assert chat.delete_item("wire_before") is None
    assert chat.delete_item("wire_after") is after
