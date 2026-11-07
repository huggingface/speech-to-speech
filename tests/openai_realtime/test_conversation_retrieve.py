"""Realtime ``conversation.item.retrieve`` through Chat, the service, and the wire route."""

from types import SimpleNamespace

import pytest
from openai.types.realtime import (
    ConversationItemCreateEvent,
    ConversationItemDeleteEvent,
    ConversationItemRetrieveEvent,
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
    ResponseCreateEvent,
)
from starlette.testclient import TestClient

from speech_to_speech.LLM.chat import Chat, make_assistant_message, make_user_message
from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.messages import AssistantTextPart
from speech_to_speech.pipeline.speaker_metadata import SpeakerAttribution, SpeakerInterval
from tests.llm_history import drive_llm
from tests.test_response_overrides import _RecordingLocalHandler

from .test_websocket_router import setup as setup


def _retrieve(service, conn_id, item_id):
    return service.handle_conversation_item_retrieve(
        conn_id, ConversationItemRetrieveEvent(type="conversation.item.retrieve", item_id=item_id)
    )


def _call():
    return RealtimeConversationItemFunctionCall(
        type="function_call", id="client_call", call_id="call_test", name="lookup", arguments='{"q": 1}'
    )


def _output():
    return RealtimeConversationItemFunctionCallOutput(
        type="function_call_output", id="client_output", call_id="call_test", output="result"
    )


def _speak(service, conn_id, turn_id):
    started = service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(turn_id=turn_id, turn_revision=0))
    item_id = next(e.item_id for e in started if e.type == "input_audio_buffer.speech_started")
    stopped = service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent(turn_id=turn_id, turn_revision=0))
    assert any(e.type == "conversation.item.created" and e.item.id == item_id for e in stopped)
    return item_id


@pytest.mark.parametrize("kind", ["user", "function_call", "function_call_output"])
def test_retrieve_created_item(service, conn_id, kind):
    items = {
        "user": [make_user_message("hello")],
        "function_call": [_call()],
        "function_call_output": [_call(), _output()],
    }
    for item in items[kind]:
        created = service.handle_conversation_item_create(
            conn_id, ConversationItemCreateEvent(type="conversation.item.create", item=item)
        )
        assert created[0].type == "conversation.item.created"
    item = items[kind][-1]

    events = _retrieve(service, conn_id, item.id)

    assert [e.type for e in events] == ["conversation.item.retrieved"]
    assert events[0].event_id.startswith("event_")
    retrieved = events[0].item
    assert retrieved.object == "realtime.item"
    assert retrieved.status == "completed"
    assert retrieved.model_dump(exclude={"object", "status"}) == item.model_dump(exclude={"object", "status"})


def test_retrieve_transcribed_speech_returns_audio_item_with_transcript(service, conn_id):
    item_id = _speak(service, conn_id, "spoken")

    pending = _retrieve(service, conn_id, item_id)[0].item
    assert pending.role == "user"
    assert [(part.type, part.transcript) for part in pending.content] == [("input_audio", None)]

    completed = service.dispatch_pipeline_event(
        conn_id, TranscriptionCompletedEvent(transcript="what time is it", turn_id="spoken", turn_revision=0)
    )
    assert any(e.type == "conversation.item.input_audio_transcription.completed" for e in completed)

    retrieved = _retrieve(service, conn_id, item_id)[0].item
    assert retrieved.id == item_id
    assert retrieved.status == "completed"
    assert [(part.type, part.transcript) for part in retrieved.content] == [("input_audio", "what time is it")]
    # The chat keeps the text form the LLM consumes.
    assert service._state(conn_id).runtime_config.chat.get_item(item_id).content[0].type == "input_text"


def test_retrieve_speech_returns_published_transcript_not_speaker_annotation(service, conn_id):
    item_id = _speak(service, conn_id, "diarized")
    completed = service.dispatch_pipeline_event(
        conn_id,
        TranscriptionCompletedEvent(
            transcript="hello there",
            turn_id="diarized",
            turn_revision=0,
            speaker_attribution=SpeakerAttribution(intervals=[SpeakerInterval(speaker=0, start=0.0, end=1.0)]),
        ),
    )
    sent = next(e for e in completed if e.type == "conversation.item.input_audio_transcription.completed")
    assert sent.transcript == "hello there"
    held = service._state(conn_id).runtime_config.chat.get_item(item_id)
    assert held.content[0].text == "[speaker_0] hello there"

    retrieved = _retrieve(service, conn_id, item_id)[0].item
    assert [(part.type, part.transcript) for part in retrieved.content] == [("input_audio", "hello there")]


def test_retrieve_empty_transcript_item(service, conn_id):
    item_id = _speak(service, conn_id, "noise")
    completed = service.dispatch_pipeline_event(
        conn_id, TranscriptionCompletedEvent(transcript="", turn_id="noise", turn_revision=0)
    )
    assert any(
        e.type == "conversation.item.input_audio_transcription.completed" and e.item_id == item_id for e in completed
    )
    assert service._state(conn_id).runtime_config.chat.get_item(item_id) is None

    retrieved = _retrieve(service, conn_id, item_id)[0].item
    assert [(part.type, part.transcript) for part in retrieved.content] == [("input_audio", "")]

    deleted = service.handle_conversation_item_delete(
        conn_id, ConversationItemDeleteEvent(type="conversation.item.delete", item_id=item_id)
    )
    assert [e.type for e in deleted] == ["conversation.item.deleted"]
    assert _retrieve(service, conn_id, item_id)[0].type == "error"


def test_retrieve_generated_assistant_by_wire_id(service, conn_id):
    service.handle_response_create(
        conn_id, ResponseCreateEvent(type="response.create", response={"output_modalities": ["text"]})
    )
    request = service.text_prompt_queue.get_nowait()
    handler = object.__new__(_RecordingLocalHandler)
    handler.cancel_scope = handler.speculative_turns = handler.compactor = None
    handler.enable_lang_prompt = False
    handler.tokenizer = SimpleNamespace(encode=lambda _text: [])
    handler.emit_text = True
    list(drive_llm(handler, request, service=service, conn_id=conn_id))
    events = service.dispatch_pipeline_event(
        conn_id, AssistantOutputEvent(parts=[AssistantTextPart(text="ok")], response_key=request.response_key)
    )
    wire_item_id = next(e.item.id for e in events if e.type == "response.output_item.added")

    live = _retrieve(service, conn_id, wire_item_id)[0].item
    assert live.role == "assistant"
    assert live.status == "in_progress"
    assert [(part.type, part.text) for part in live.content] == [("output_text", "ok")]

    service.finish_response(conn_id)
    held = _retrieve(service, conn_id, wire_item_id)[0].item
    assert held.id == wire_item_id
    assert held.status == "completed"
    # After response.done the chat entry bound to the wire ID is authoritative.
    assert held.content == service._state(conn_id).runtime_config.chat.get_item(wire_item_id).content
    assert held.content[0].type == "output_text"
    assert "response_item" not in held.model_dump()


def test_retrieve_completed_audio_reply_keeps_output_audio(service, conn_id):
    service.handle_response_create(
        conn_id, ResponseCreateEvent(type="response.create", response={"output_modalities": ["audio"]})
    )
    request = service.text_prompt_queue.get_nowait()
    handler = object.__new__(_RecordingLocalHandler)
    handler.cancel_scope = handler.speculative_turns = handler.compactor = None
    handler.enable_lang_prompt = False
    handler.tokenizer = SimpleNamespace(encode=lambda _text: [])
    handler.emit_text = True
    list(drive_llm(handler, request, service=service, conn_id=conn_id))
    events = service.dispatch_pipeline_event(
        conn_id, AssistantOutputEvent(parts=[AssistantTextPart(text="ok")], response_key=request.response_key)
    )
    wire_item_id = next(e.item.id for e in events if e.type == "response.output_item.added")
    live = _retrieve(service, conn_id, wire_item_id)[0].item
    assert [part.type for part in live.content] == ["output_audio"]

    done = next(e for e in service.finish_response(conn_id) if e.type == "response.done")
    assert [part.type for part in done.response.output[0].content] == ["output_audio"]
    held_text = service._state(conn_id).runtime_config.chat.get_item(wire_item_id).content[0].text

    retrieved = _retrieve(service, conn_id, wire_item_id)[0].item
    assert retrieved.status == "completed"
    assert [(part.type, part.transcript, part.text) for part in retrieved.content] == [
        ("output_audio", held_text.strip(), None)
    ]


def test_chat_get_item_merges_aliased_provider_fragments():
    chat = Chat(10)
    before = [make_assistant_message("first"), make_assistant_message("second")]
    chat.add_provisional_generation_items("response", [*before, _call()])
    chat.bind_assistant_item_ids(
        "response", [make_assistant_message("wire").model_copy(update={"id": "wire_before"}), _call()]
    )
    chat.finalize_provisional_generation("response")

    merged = chat.get_item("wire_before")

    assert merged.id == "wire_before"
    assert [part.text for part in merged.content] == ["first", "second"]
    merged.content[0].text = "mutated"
    assert chat.buffer[0].content[0].text == "first"
    assert chat.get_item("client_call").call_id == "call_test"
    assert chat.get_item("missing") is None


def test_retrieve_unknown_or_unseen_item_is_an_error(service, conn_id):
    started = service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(turn_id="speaking", turn_revision=0))
    uncommitted_id = started[0].item_id
    for item_id in ("missing", uncommitted_id):
        events = _retrieve(service, conn_id, item_id)
        assert [e.type for e in events] == ["error"]
        assert events[0].error.type == "conversation_item_not_found"
        assert item_id in events[0].error.message


def test_retrieve_over_websocket(setup):
    app, *_ = setup
    with TestClient(app) as client, client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "conversation.item.create",
                "item": {
                    "id": "typed",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
            }
        )
        assert ws.receive_json()["type"] == "conversation.item.created"

        ws.send_json({"type": "conversation.item.retrieve", "item_id": "typed", "event_id": "get_typed"})
        retrieved = ws.receive_json()
        assert retrieved["type"] == "conversation.item.retrieved"
        assert retrieved["item"]["id"] == "typed"
        assert retrieved["item"]["content"][0]["text"] == "hi"

        ws.send_json({"type": "conversation.item.retrieve", "item_id": "missing", "event_id": "get_missing"})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"]["type"] == "conversation_item_not_found"
        assert error["error"]["event_id"] == "get_missing"


def test_response_usage_reports_cached_token_details_object(service, conn_id):
    service.response._ensure_response(conn_id)
    done = service.finish_response(conn_id)[-1]
    assert done.type == "response.done"
    details = done.model_dump()["response"]["usage"]["input_token_details"]["cached_tokens_details"]
    assert details == {"audio_tokens": 0, "image_tokens": 0, "text_tokens": 0}
