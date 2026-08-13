"""Unit tests for api.openai_realtime.service.RealtimeService.

Every public method is exercised and the emitted OpenAI Realtime events are
validated for correct type, attributes, and state transitions.
"""

import base64
import json
from queue import Queue
from threading import Event, Thread
from time import sleep

import numpy as np
import pytest
from openai.types.realtime import (
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    ConversationItemTruncateEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    RealtimeErrorEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCancelEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    SessionUpdateEvent,
)
from openai.types.realtime.conversation_item import (
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
    RealtimeConversationItemUserMessage,
)

from speech_to_speech.api.openai_realtime.service import (
    CHUNK_SIZE_BYTES,
    RealtimeService,
)
from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    AssistantResponseDoneEvent,
    AssistantToolCallReadyEvent,
    AudioInputCompletedEvent,
    PartialTranscriptionEvent,
    ResponseFailedEvent,
    ResponseGenerationDoneEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TokenUsageEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.messages import (
    AssistantTextPart,
    AssistantToolCallPart,
    GenerateResponseRequest,
    ResponsePrefetchTransaction,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pcm_bytes(n_samples: int) -> bytes:
    """Return n_samples * 2 zero bytes (valid PCM16 silence)."""
    return b"\x00" * (n_samples * 2)


def _b64_pcm(n_samples: int) -> str:
    return base64.b64encode(_pcm_bytes(n_samples)).decode("ascii")


def _make_audio_append(audio_b64: str) -> InputAudioBufferAppendEvent:
    return InputAudioBufferAppendEvent(type="input_audio_buffer.append", audio=audio_b64)


# ===================================================================
# Connection lifecycle
# ===================================================================


class TestConnectionLifecycle:
    def test_register_creates_session_id(self, service):
        sid = service.register()
        assert sid.startswith("session_")
        st = service._state(sid)
        assert st.conversation_id.startswith("conv_")
        assert st.in_response is False
        assert st.last_item_id is None
        service.unregister(sid)

    def test_register_applies_server_default_instructions(self, text_prompt_queue, should_listen):
        service = RealtimeService(
            text_prompt_queue=text_prompt_queue,
            should_listen=should_listen,
            default_instructions="Use the configured persona.",
        )

        sid = service.register()

        assert service._state(sid).runtime_config.session.instructions == "Use the configured persona."
        service.unregister(sid)

    def test_unregister_removes_state(self, service):
        sid = service.register()
        service.unregister(sid)
        with pytest.raises(KeyError):
            service._state(sid)

    def test_build_session_created(self, service, conn_id, runtime_config):
        service.handle_session_update(
            conn_id,
            SessionUpdateEvent(
                type="session.update",
                session={
                    "type": "realtime",
                    "instructions": "Be helpful",
                    "tools": [{"type": "function", "name": "get_weather"}],
                    "tool_choice": "auto",
                    "audio": {
                        "input": {"turn_detection": {"type": "server_vad"}},
                        "output": {"voice": "echo"},
                    },
                },
            ),
        )

        evt = service.build_session_created(conn_id)
        assert isinstance(evt, SessionCreatedEvent)
        assert evt.event_id.startswith("event_")
        assert evt.session is not None
        assert evt.session.instructions == "Be helpful"
        assert evt.session.tools is not None
        assert evt.session.tool_choice == "auto"
        assert evt.session.audio.output.voice == "echo"
        assert evt.session.audio.input.turn_detection.type == "server_vad"

    def test_build_session_updated(self, service, conn_id, runtime_config):
        service.handle_session_update(
            conn_id,
            SessionUpdateEvent(
                type="session.update",
                session={"type": "realtime", "instructions": "Be concise"},
            ),
        )

        evt = service.build_session_updated(conn_id)
        assert isinstance(evt, SessionUpdatedEvent)
        assert evt.event_id.startswith("event_")
        assert evt.session is not None
        assert evt.session.instructions == "Be concise"


# ===================================================================
# Client event parsing
# ===================================================================


class TestParseClientEvent:
    def test_parse_valid_audio_append(self, service):
        raw = {"type": "input_audio_buffer.append", "audio": "AAAA"}
        evt = service.parse_client_event(raw)
        assert isinstance(evt, InputAudioBufferAppendEvent)

    def test_parse_valid_session_update(self, service):
        raw = {"type": "session.update", "session": {"type": "realtime"}, "voice": "alloy"}
        evt = service.parse_client_event(raw)
        assert isinstance(evt, SessionUpdateEvent)
        assert evt.voice == "alloy"

    def test_parse_valid_conversation_item_create(self, service):
        raw = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            },
        }
        evt = service.parse_client_event(raw)
        assert isinstance(evt, ConversationItemCreateEvent)

    def test_parse_valid_response_create(self, service):
        raw = {"type": "response.create"}
        evt = service.parse_client_event(raw)
        assert isinstance(evt, ResponseCreateEvent)

    def test_parse_valid_response_cancel(self, service):
        raw = {"type": "response.cancel"}
        evt = service.parse_client_event(raw)
        assert isinstance(evt, ResponseCancelEvent)

    def test_parse_valid_conversation_item_truncate(self, service):
        """The stock Agents SDK sends this after cancelling audible WS output."""
        raw = {
            "type": "conversation.item.truncate",
            "item_id": "item_assistant",
            "content_index": 0,
            "audio_end_ms": 120,
        }
        evt = service.parse_client_event(raw)
        assert isinstance(evt, ConversationItemTruncateEvent)

    def test_parse_unknown_event_type(self, service):
        assert service.parse_client_event({"type": "bogus.event"}) is None

    def test_parse_invalid_payload(self, service):
        raw = {"type": "input_audio_buffer.append"}  # missing required 'audio'
        assert service.parse_client_event(raw) is None


# ===================================================================
# Audio append
# ===================================================================


class TestHandleAudioAppend:
    def test_audio_append_decodes_and_chunks(self, service, conn_id):
        audio_b64 = _b64_pcm(512 * 3)  # exactly 3 chunks
        evt = _make_audio_append(audio_b64)
        chunks = service.handle_audio_append(conn_id, evt)
        assert len(chunks) == 3
        assert all(len(c) == CHUNK_SIZE_BYTES for c in chunks)
        assert service._state(conn_id).audio_buffer_has_data is True

    def test_audio_append_invalid_base64(self, service, conn_id):
        evt = InputAudioBufferAppendEvent(type="input_audio_buffer.append", audio="!!!invalid!!!")
        chunks = service.handle_audio_append(conn_id, evt)
        assert chunks == []

    def test_audio_append_undersized_tail(self, service, conn_id):
        audio_b64 = _b64_pcm(512 + 100)  # 1 full chunk + 100 samples remainder
        evt = _make_audio_append(audio_b64)
        chunks = service.handle_audio_append(conn_id, evt)
        assert len(chunks) == 1


# ===================================================================
# Session update
# ===================================================================


class TestHandleSessionUpdate:
    def _make_update(self, **session_fields) -> SessionUpdateEvent:
        session_fields.setdefault("type", "realtime")
        return SessionUpdateEvent(type="session.update", session=session_fields)  # type: ignore[arg-type]

    def test_session_update_voice(self, service, conn_id, runtime_config):
        evt = self._make_update(
            audio={"output": {"voice": "shimmer"}},
        )
        service.handle_session_update(conn_id, evt)
        assert runtime_config.session.audio.output.voice == "shimmer"

    def test_session_update_instructions(self, service, conn_id, runtime_config):
        service.handle_session_update(conn_id, self._make_update(instructions="Be concise"))
        assert runtime_config.session.instructions == "Be concise"

    def test_session_update_tools_and_tool_choice(self, service, conn_id, runtime_config):
        tools = [{"type": "function", "name": "f1"}]
        service.handle_session_update(conn_id, self._make_update(tools=tools, tool_choice="required"))
        assert runtime_config.session.tools is not None
        assert runtime_config.session.tool_choice == "required"

    def test_session_update_rejects_transcription_session(self, service, conn_id, runtime_config):
        raw = {
            "type": "session.update",
            "session": {"type": "transcription"},
        }
        evt = SessionUpdateEvent.model_validate(raw)
        err = service.handle_session_update(conn_id, evt)
        assert isinstance(err, RealtimeErrorEvent)
        assert err.error.type == "invalid_session_type"

    def test_session_update_nested_audio_format(self, service, conn_id, runtime_config):
        raw = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "audio": {
                    "input": {"turn_detection": {"type": "server_vad", "threshold": 0.5}},
                    "output": {"voice": "nova"},
                },
            },
        }
        evt = SessionUpdateEvent.model_validate(raw)
        service.handle_session_update(conn_id, evt)
        assert runtime_config.session.audio.output.voice == "nova"
        assert runtime_config.session.audio.input.turn_detection.type == "server_vad"

    def test_session_update_merges_partial_updates(self, service, conn_id, runtime_config):
        """Partial updates preserve previously-set fields."""
        service.handle_session_update(
            conn_id,
            self._make_update(
                audio={"output": {"voice": "echo"}},
                instructions="Be helpful",
            ),
        )
        assert runtime_config.session.audio.output.voice == "echo"
        assert runtime_config.session.instructions == "Be helpful"

        service.handle_session_update(conn_id, self._make_update(instructions="Be concise"))
        assert runtime_config.session.instructions == "Be concise"
        assert runtime_config.session.audio.output.voice == "echo"  # preserved from first update


# ===================================================================
# Conversation item create
# ===================================================================


class TestHandleConversationItemCreate:
    def _text_event(self, text: str = "hello", item_id: str = "msg_abc") -> ConversationItemCreateEvent:
        return ConversationItemCreateEvent(
            type="conversation.item.create",
            item={  # type: ignore[arg-type]
                "id": item_id,
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        )

    def test_text_input_emits_conversation_item_created(
        self,
        service,
        conn_id,
        text_prompt_queue,
    ):
        events = service.handle_conversation_item_create(conn_id, self._text_event("hi"))
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, ConversationItemCreatedEvent)
        assert evt.previous_item_id is None  # first item
        assert evt.item.role == "user"
        assert evt.item.content[0].type == "input_text"
        assert evt.item.content[0].text == "hi"
        last = service._state(conn_id).runtime_config.chat.buffer[-1]
        assert last.role == "user"
        assert last.content[0].type == "input_text"
        assert last.content[0].text == "hi"

    def test_text_input_previous_item_id_chain(self, service, conn_id):
        e1 = service.handle_conversation_item_create(conn_id, self._text_event("a", "msg_1"))
        e2 = service.handle_conversation_item_create(conn_id, self._text_event("b", "msg_2"))
        assert e1[0].previous_item_id is None
        assert e2[0].previous_item_id == e1[0].item.id

    def test_function_call_output_forwarded(self, service, conn_id, text_prompt_queue):
        from openai.types.realtime.realtime_conversation_item_function_call import (
            RealtimeConversationItemFunctionCall,
        )

        service._state(conn_id).runtime_config.chat.add_item(
            RealtimeConversationItemFunctionCall(
                type="function_call", call_id="call_1", name="get_weather", arguments="{}"
            )
        )
        evt = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={"type": "function_call_output", "output": '{"result": 42}', "call_id": "call_1"},
        )
        events = service.handle_conversation_item_create(conn_id, evt)
        assert len(events) == 1
        assert isinstance(events[0], ConversationItemCreatedEvent)
        last = service._state(conn_id).runtime_config.chat.buffer[-1]
        assert last.type == "function_call_output"
        assert last.call_id == "call_1"
        assert last.output == '{"result": 42}'

    def test_function_call_output_rejected_for_unknown_call_id(self, service, conn_id, text_prompt_queue):
        evt = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={"type": "function_call_output", "output": '{"result": 42}', "call_id": "call_unknown"},
        )
        events = service.handle_conversation_item_create(conn_id, evt)
        assert len(events) == 1
        assert isinstance(events[0], RealtimeErrorEvent)
        assert "call_unknown" in events[0].error.message
        assert not any(
            getattr(e, "type", None) == "function_call_output"
            for e in service._state(conn_id).runtime_config.chat.buffer
        )

    def test_input_image_forwarded(self, service, conn_id, text_prompt_queue):
        evt = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "image_url": "https://example.com/img.png"}],
            },
        )
        events = service.handle_conversation_item_create(conn_id, evt)
        assert len(events) == 1
        assert isinstance(events[0], ConversationItemCreatedEvent)
        last = service._state(conn_id).runtime_config.chat.buffer[-1]
        assert last.role == "user"
        assert last.content[0].type == "input_image"
        assert last.content[0].image_url == "https://example.com/img.png"

    def test_mixed_text_and_image_forwarded(self, service, conn_id, text_prompt_queue):
        evt = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is this?"},
                    {"type": "input_image", "image_url": "data:image/png;base64,abc123"},
                ],
            },
        )
        events = service.handle_conversation_item_create(conn_id, evt)
        assert len(events) == 1
        last = service._state(conn_id).runtime_config.chat.buffer[-1]
        assert last.role == "user"
        assert len(last.content) == 2
        assert last.content[0].type == "input_text"
        assert last.content[0].text == "What is this?"
        assert last.content[1].type == "input_image"
        assert last.content[1].image_url == "data:image/png;base64,abc123"


class TestDeferConversationItemsDuringResponse:
    """conversation.item.create is buffered while a response is generating and
    flushed, in order, once it completes — so a client item never races the LLM
    handler's chat write-back (which runs on the pipeline thread)."""

    def _text_event(self, text: str, item_id: str = "msg_x") -> ConversationItemCreateEvent:
        return ConversationItemCreateEvent(
            type="conversation.item.create",
            item={  # type: ignore[arg-type]
                "id": item_id,
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        )

    def _user_texts(self, chat) -> list[str]:
        return [i.content[0].text for i in chat.buffer if getattr(i, "role", None) == "user"]

    def test_applied_immediately_when_no_active_response(self, service, conn_id):
        st = service._state(conn_id)
        assert st.in_response is False
        events = service.handle_conversation_item_create(conn_id, self._text_event("hi"))
        assert len(events) == 1
        assert isinstance(events[0], ConversationItemCreatedEvent)
        assert self._user_texts(st.runtime_config.chat) == ["hi"]
        assert st.deferred_items == []

    def test_item_deferred_while_in_response(self, service, conn_id):
        st = service._state(conn_id)
        st.in_response = True
        events = service.handle_conversation_item_create(conn_id, self._text_event("hi"))
        assert events == []  # ack deferred too
        assert len(st.deferred_items) == 1
        assert self._user_texts(st.runtime_config.chat) == []  # not yet in chat

    def test_deferred_items_flushed_in_order_on_finish(self, service, conn_id):
        st = service._state(conn_id)
        st.in_response = True
        service.handle_conversation_item_create(conn_id, self._text_event("a", "msg_1"))
        service.handle_conversation_item_create(conn_id, self._text_event("b", "msg_2"))
        assert self._user_texts(st.runtime_config.chat) == []

        events = service.finish_response(conn_id)

        assert st.in_response is False
        assert st.deferred_items == []
        assert self._user_texts(st.runtime_config.chat) == ["a", "b"]  # arrival order preserved
        created = [e for e in events if isinstance(e, ConversationItemCreatedEvent)]
        assert len(created) == 2

    def test_function_call_output_flushes_at_logical_generation_done(self, service, conn_id):
        from openai.types.realtime.realtime_conversation_item_function_call import (
            RealtimeConversationItemFunctionCall,
        )

        st = service._state(conn_id)
        chat = st.runtime_config.chat
        # The function_call the generation produced (held in _pending_tool_calls).
        chat.add_item(
            RealtimeConversationItemFunctionCall(
                type="function_call", call_id="call_1", name="camera_snapshot", arguments="{}"
            )
        )
        st.in_response = True
        st.current_response_key = "response_1"
        evt = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={"type": "function_call_output", "output": "ok", "call_id": "call_1"},
        )
        # A fast result waits for the origin LM's trailing items, not for TTS.
        assert service.handle_conversation_item_create(conn_id, evt) == []
        assert len(st.deferred_items) == 1

        created = service.dispatch_pipeline_event(
            conn_id,
            ResponseGenerationDoneEvent(response_key="response_1", call_ids=["call_1"]),
        )

        assert created == []
        assert st.deferred_items == []
        assert len(st.pending_item_acks) == 1
        assert chat._has_call_id_in_buffer("call_1")
        assert chat.buffer[-1].type == "function_call_output"

        finish_events = service.finish_response(conn_id)

        assert not any(isinstance(e, RealtimeErrorEvent) for e in finish_events)
        assert isinstance(finish_events[-1], ConversationItemCreatedEvent)
        assert st.pending_item_acks == []

    def test_fast_tool_output_does_not_overtake_trailing_origin_items(self, service, conn_id):
        st = service._state(conn_id)
        chat = st.runtime_config.chat
        response_key = "response_origin"
        first_call = RealtimeConversationItemFunctionCall(
            type="function_call",
            id="fc_1",
            call_id="call_1",
            name="first",
            arguments="{}",
        )
        assert chat.add_provisional_generation_items(response_key, [first_call]) is not None
        st.in_response = True
        st.current_response_key = response_key

        assert (
            service.handle_conversation_item_create(
                conn_id,
                ConversationItemCreateEvent(
                    type="conversation.item.create",
                    item={"type": "function_call_output", "call_id": "call_1", "output": "first result"},
                ),
            )
            == []
        )

        trailing_message = RealtimeConversationItemAssistantMessage(
            type="message",
            role="assistant",
            content=[{"type": "output_text", "text": "I also need one more thing."}],
        )
        second_call = RealtimeConversationItemFunctionCall(
            type="function_call",
            id="fc_2",
            call_id="call_2",
            name="second",
            arguments="{}",
        )
        assert chat.add_provisional_generation_items(response_key, [trailing_message, second_call]) is not None

        service.dispatch_pipeline_event(
            conn_id,
            ResponseGenerationDoneEvent(response_key=response_key, call_ids=["call_1", "call_2"]),
        )

        assert [item.type for item in chat.buffer] == [
            "function_call",
            "message",
            "function_call",
            "function_call_output",
        ]
        assert chat.has_pending_tool_calls()

    def test_early_tool_output_ack_follows_trailing_origin_output(
        self,
        service,
        conn_id,
        text_prompt_queue,
    ):
        st = service._state(conn_id)
        chat = st.runtime_config.chat
        response_key = "response_origin"
        call = RealtimeConversationItemFunctionCall(
            type="function_call",
            id="fc_1",
            call_id="call_1",
            name="lookup",
            arguments="{}",
        )
        trailing_message = RealtimeConversationItemAssistantMessage(
            type="message",
            role="assistant",
            content=[{"type": "output_text", "text": "One more detail."}],
        )
        assert chat.add_provisional_generation_items(response_key, [call, trailing_message]) is not None
        st.in_response = True
        st.current_response_key = response_key

        tool_event = next(
            event
            for event in service.dispatch_pipeline_event(
                conn_id,
                AssistantOutputEvent(
                    response_key=response_key,
                    tools=[
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "lookup",
                            "arguments": "{}",
                        }
                    ],
                ),
            )
            if isinstance(event, ResponseFunctionCallArgumentsDoneEvent)
        )
        assert st.last_item_id == tool_event.item_id
        assert (
            service.handle_conversation_item_create(
                conn_id,
                ConversationItemCreateEvent(
                    type="conversation.item.create",
                    item={"type": "function_call_output", "call_id": "call_1", "output": "result"},
                ),
            )
            == []
        )

        assert (
            service.dispatch_pipeline_event(
                conn_id,
                ResponseGenerationDoneEvent(response_key=response_key, call_ids=["call_1"]),
            )
            == []
        )
        assert chat.buffer[-1].type == "function_call_output"
        assert len(st.pending_item_acks) == 1
        assert isinstance(text_prompt_queue.get_nowait(), GenerateResponseRequest)

        trailing_event = next(
            event
            for event in service.dispatch_pipeline_event(
                conn_id,
                AssistantOutputEvent(response_key=response_key, text="One more detail."),
            )
            if isinstance(event, ResponseAudioTranscriptDeltaEvent)
        )
        terminal_events = service.finish_response(conn_id, response_key=response_key)
        done_index = next(index for index, event in enumerate(terminal_events) if isinstance(event, ResponseDoneEvent))
        created_index = next(
            index for index, event in enumerate(terminal_events) if isinstance(event, ConversationItemCreatedEvent)
        )
        created = terminal_events[created_index]

        assert created_index > done_index
        assert created.previous_item_id == trailing_event.item_id
        assert st.last_item_id == created.item.id
        assert [item.type for item in chat.buffer] == ["function_call", "message", "function_call_output"]

    def test_cancel_rolls_back_provisional_call_before_flushing_deferred_output(self, service, conn_id):
        st = service._state(conn_id)
        chat = st.runtime_config.chat
        user = chat.add_item(
            RealtimeConversationItemUserMessage(
                type="message",
                role="user",
                content=[{"type": "input_text", "text": "use a tool"}],
            )
        )
        response_key = "response_cancelled"
        recorded_items = chat.add_provisional_generation_items(
            response_key,
            [
                RealtimeConversationItemFunctionCall(
                    type="function_call",
                    id="fc_cancelled",
                    call_id="call_cancelled",
                    name="camera_snapshot",
                    arguments="{}",
                )
            ],
        )
        assert recorded_items is not None
        call = recorded_items[0]
        assert isinstance(call, RealtimeConversationItemFunctionCall)
        st.in_response = True
        st.current_response_id = "resp_cancelled"
        st.current_response_key = response_key
        st.pending_function_calls[0] = call
        result_event = ConversationItemCreateEvent(
            type="conversation.item.create",
            item=RealtimeConversationItemFunctionCallOutput(
                type="function_call_output",
                call_id=call.call_id,
                output="result",
            ),
        )

        assert service.handle_conversation_item_create(conn_id, result_event) == []

        events = service.finish_response(conn_id, status="cancelled", reason="client_cancelled")

        assert any(isinstance(event, ResponseDoneEvent) for event in events)
        assert any(isinstance(event, RealtimeErrorEvent) for event in events), events
        assert not any(isinstance(event, ConversationItemCreatedEvent) for event in events)
        assert chat.buffer == [user]
        assert not chat.has_pending_tool_calls()
        assert chat._provisional_generations == {}

        # response.done is now a safe boundary: an immediate follow-up does not
        # see the cancelled call or get rejected as waiting for its output.
        next_response = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
        assert isinstance(next_response, ResponseCreatedEvent)

    def test_cancel_preserves_image_applied_for_prefetch(
        self,
        service,
        conn_id,
        text_prompt_queue,
    ):
        st = service._state(conn_id)
        chat = st.runtime_config.chat
        user = chat.add_item(
            RealtimeConversationItemUserMessage(
                type="message",
                role="user",
                content=[{"type": "input_text", "text": "use a tool"}],
            )
        )
        response_key = "response_cancelled_after_generation"
        call = RealtimeConversationItemFunctionCall(
            type="function_call",
            id="fc_cancelled",
            call_id="call_cancelled",
            name="camera_snapshot",
            arguments="{}",
        )
        assert chat.add_provisional_generation_items(response_key, [call]) is not None
        st.in_response = True
        st.current_response_id = "resp_cancelled"
        st.current_response_key = response_key
        st.pending_function_calls[0] = call

        image_event = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={
                "id": "msg_client_camera_frame_cancelled",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "image_url": "data:image/jpeg;base64,abc"}],
            },
        )
        assert service.handle_conversation_item_create(conn_id, image_event) == []
        assert (
            service.handle_conversation_item_create(
                conn_id,
                ConversationItemCreateEvent(
                    type="conversation.item.create",
                    previous_item_id="msg_client_camera_frame_cancelled",
                    item={"type": "function_call_output", "call_id": call.call_id, "output": "result"},
                ),
            )
            == []
        )
        assert (
            service.dispatch_pipeline_event(
                conn_id,
                ResponseGenerationDoneEvent(response_key=response_key, call_ids=[call.call_id]),
            )
            == []
        )
        prefetch = text_prompt_queue.get_nowait()
        assert isinstance(prefetch, GenerateResponseRequest)
        assert chat.buffer[-1].type == "function_call_output"
        assert len(st.pending_item_acks) == 2

        events = service.finish_response(
            conn_id,
            status="cancelled",
            reason="client_cancelled",
            response_key=response_key,
        )

        assert any(isinstance(event, RealtimeErrorEvent) for event in events)
        created = [event for event in events if isinstance(event, ConversationItemCreatedEvent)]
        assert len(created) == 1
        assert created[0].item.id == "msg_client_camera_frame_cancelled"
        assert created[0].item.content[0].image_url == "data:image/jpeg;base64,abc"
        assert chat.buffer == [user, created[0].item]
        assert st.pending_item_acks == []
        assert st.tool_followup_prefetch_request is None
        assert prefetch.response_key in st.closed_response_keys

    def test_cancel_preserves_standalone_user_image(self, service, conn_id):
        st = service._state(conn_id)
        chat = st.runtime_config.chat
        user = chat.add_item(
            RealtimeConversationItemUserMessage(
                type="message",
                role="user",
                content=[{"type": "input_text", "text": "use a tool"}],
            )
        )
        response_key = "response_cancelled_with_user_image"
        call = RealtimeConversationItemFunctionCall(
            type="function_call",
            id="fc_cancelled",
            call_id="call_cancelled",
            name="camera_snapshot",
            arguments="{}",
        )
        assert chat.add_provisional_generation_items(response_key, [call]) is not None
        st.in_response = True
        st.current_response_id = "resp_cancelled"
        st.current_response_key = response_key
        st.pending_function_calls[0] = call
        service.dispatch_pipeline_event(
            conn_id,
            ResponseGenerationDoneEvent(response_key=response_key, call_ids=[call.call_id]),
        )

        assert (
            service.handle_conversation_item_create(
                conn_id,
                ConversationItemCreateEvent(
                    type="conversation.item.create",
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_image", "image_url": "data:image/jpeg;base64,user"}],
                    },
                ),
            )
            == []
        )
        assert len(st.deferred_items) == 1
        assert chat.buffer == [user, call]

        events = service.finish_response(
            conn_id,
            status="cancelled",
            reason="client_cancelled",
            response_key=response_key,
        )

        created = [event for event in events if isinstance(event, ConversationItemCreatedEvent)]
        assert len(created) == 1
        assert created[0].item.content[0].image_url == "data:image/jpeg;base64,user"
        assert chat.buffer == [user, created[0].item]

    def test_cancel_preserves_ordered_user_image_between_tool_outputs(
        self,
        service,
        conn_id,
        text_prompt_queue,
    ):
        st = service._state(conn_id)
        chat = st.runtime_config.chat
        user = chat.add_item(
            RealtimeConversationItemUserMessage(
                type="message",
                role="user",
                content=[{"type": "input_text", "text": "use two tools"}],
            )
        )
        response_key = "response_cancelled_with_interleaved_user_image"
        calls = [
            RealtimeConversationItemFunctionCall(
                type="function_call",
                id=f"fc_{index}",
                call_id=f"call_{index}",
                name="camera_snapshot",
                arguments="{}",
            )
            for index in (1, 2)
        ]
        assert chat.add_provisional_generation_items(response_key, calls) is not None
        st.in_response = True
        st.current_response_id = "resp_cancelled"
        st.current_response_key = response_key
        st.pending_function_calls = {index: call for index, call in enumerate(calls)}
        service.dispatch_pipeline_event(
            conn_id,
            ResponseGenerationDoneEvent(
                response_key=response_key,
                call_ids=[call.call_id for call in calls],
            ),
        )

        assert (
            service.handle_conversation_item_create(
                conn_id,
                ConversationItemCreateEvent(
                    type="conversation.item.create",
                    item={"type": "function_call_output", "call_id": "call_1", "output": "first"},
                ),
            )
            == []
        )
        assert (
            service.handle_conversation_item_create(
                conn_id,
                ConversationItemCreateEvent(
                    type="conversation.item.create",
                    item={
                        "id": "msg_ordinary_user_image",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_image", "image_url": "data:image/jpeg;base64,user"}],
                    },
                ),
            )
            == []
        )
        assert (
            service.handle_conversation_item_create(
                conn_id,
                ConversationItemCreateEvent(
                    type="conversation.item.create",
                    previous_item_id="msg_ordinary_user_image",
                    item={"type": "function_call_output", "call_id": "call_2", "output": "second"},
                ),
            )
            == []
        )
        prefetch = text_prompt_queue.get_nowait()
        assert isinstance(prefetch, GenerateResponseRequest)
        assert st.tool_followup_prefetch_request is prefetch
        assert st.deferred_items == []
        assert len(st.pending_item_acks) == 3
        assert prefetch.prefetch_transaction is not None
        prefetch.prefetch_transaction.complete(lambda: chat.strip_images({"msg_ordinary_user_image"}))

        events = service.finish_response(
            conn_id,
            status="cancelled",
            reason="client_cancelled",
            response_key=response_key,
        )

        created = [event for event in events if isinstance(event, ConversationItemCreatedEvent)]
        errors = [event for event in events if isinstance(event, RealtimeErrorEvent)]
        assert len(created) == 1
        assert len(errors) == 2
        assert created[0].item.id == "msg_ordinary_user_image"
        assert created[0].item.content[0].image_url == "data:image/jpeg;base64,user"
        assert chat.buffer == [user, created[0].item]

    def test_cancel_rolls_back_generated_call_not_yet_delivered_after_tts(self, service, conn_id):
        st = service._state(conn_id)
        chat = st.runtime_config.chat
        user = chat.add_item(
            RealtimeConversationItemUserMessage(
                type="message",
                role="user",
                content=[{"type": "input_text", "text": "use a tool"}],
            )
        )
        response_key = "response_waiting_for_tts"
        recorded = chat.add_provisional_generation_items(
            response_key,
            [
                RealtimeConversationItemFunctionCall(
                    type="function_call",
                    id="fc_waiting",
                    call_id="call_waiting",
                    name="camera_snapshot",
                    arguments="{}",
                )
            ],
        )
        assert recorded is not None
        st.in_response = True
        st.current_response_id = "resp_waiting"
        st.current_response_key = response_key

        # Model generation is complete, but the function-call event is still
        # queued behind earlier TTS and has not populated pending_function_calls.
        assert st.pending_function_calls == {}
        service.finish_response(conn_id, status="cancelled", reason="client_cancelled")

        assert chat.buffer == [user]
        assert not chat.has_pending_tool_calls()
        next_response = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
        assert isinstance(next_response, ResponseCreatedEvent)


# ===================================================================
# Audio commit
# ===================================================================


class TestHandleAudioCommit:
    def test_commit_after_audio(self, service, conn_id):
        service._state(conn_id).audio_buffer_has_data = True
        err = service.handle_audio_commit(conn_id)
        assert err is None
        assert service._state(conn_id).audio_buffer_has_data is False

    def test_commit_empty_buffer(self, service, conn_id):
        err = service.handle_audio_commit(conn_id)
        assert isinstance(err, RealtimeErrorEvent)
        assert err.error.type == "input_audio_buffer_commit_empty"


# ===================================================================
# Response create
# ===================================================================


class TestHandleResponseCreate:
    def test_response_create_ok(self, service, conn_id):
        evt = ResponseCreateEvent(type="response.create")
        result = service.handle_response_create(conn_id, evt)
        assert isinstance(result, ResponseCreatedEvent)
        assert result.response.status == "in_progress"
        st = service._state(conn_id)
        assert st.in_response is True
        assert st.current_response_id is not None
        assert st.current_item_id is not None

    def test_response_create_while_active(self, service, conn_id):
        service._state(conn_id).in_response = True
        evt = ResponseCreateEvent(type="response.create")
        err = service.handle_response_create(conn_id, evt)
        assert isinstance(err, RealtimeErrorEvent)
        assert err.error.type == "conversation_already_has_active_response"

    def test_standard_tool_followup_claims_internal_prefetch_after_response_done(
        self,
        service,
        conn_id,
        text_prompt_queue,
    ):
        st = service._state(conn_id)
        response_key = "response_origin"
        call = RealtimeConversationItemFunctionCall(
            type="function_call",
            id="fc_lookup",
            call_id="call_lookup",
            name="lookup",
            arguments="{}",
        )
        assert st.runtime_config.chat.add_provisional_generation_items(response_key, [call]) is not None
        st.in_response = True
        st.current_response_id = "resp_origin"
        st.current_response_key = response_key
        st.pending_function_calls = {0: call}

        service.dispatch_pipeline_event(
            conn_id,
            ResponseGenerationDoneEvent(response_key=response_key, call_ids=[call.call_id]),
        )
        assert text_prompt_queue.empty()

        created_items = service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={"type": "function_call_output", "call_id": call.call_id, "output": "result"},
            ),
        )
        assert created_items == []
        assert len(st.pending_item_acks) == 1
        prefetch = text_prompt_queue.get_nowait()
        assert isinstance(prefetch, GenerateResponseRequest)
        assert prefetch.response is None
        assert st.in_response is True
        assert st.current_response_id == "resp_origin"
        assert st.response_pending is True
        assert st.tool_followup_prefetch_request is prefetch

        # An early in-band create retains ordinary Realtime collision behavior.
        collision = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
        assert isinstance(collision, RealtimeErrorEvent)
        assert collision.error.type == "conversation_already_has_active_response"

        terminal_events = service.finish_response(conn_id, response_key=response_key)
        assert isinstance(terminal_events[-1], ConversationItemCreatedEvent)
        claimed = service.handle_response_create(
            conn_id,
            ResponseCreateEvent(
                type="response.create",
                response={"metadata": {"s2s_demo_create_id": "create_followup"}},
            ),
        )

        assert isinstance(claimed, ResponseCreatedEvent)
        assert claimed.response.metadata == {"s2s_demo_create_id": "create_followup"}
        assert st.current_response_key == prefetch.response_key
        assert st.tool_followup_prefetch_request is None
        assert text_prompt_queue.empty()

    def test_image_tool_followup_prefetches_with_standard_conversation_items(
        self,
        service,
        conn_id,
        text_prompt_queue,
    ):
        st = service._state(conn_id)
        response_key = "response_origin"
        call = RealtimeConversationItemFunctionCall(
            type="function_call",
            id="fc_camera",
            call_id="call_camera",
            name="camera_snapshot",
            arguments="{}",
        )
        assert st.runtime_config.chat.add_provisional_generation_items(response_key, [call]) is not None
        st.in_response = True
        st.current_response_id = "resp_origin"
        st.current_response_key = response_key
        st.pending_function_calls = {0: call}
        st.last_item_id = call.id

        service.dispatch_pipeline_event(
            conn_id,
            ResponseGenerationDoneEvent(response_key=response_key, call_ids=[call.call_id]),
        )
        image_events = service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={
                    "id": "msg_client_camera_frame_42",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "data:image/jpeg;base64,abc"}],
                },
            ),
        )
        output_events = service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                previous_item_id="msg_client_camera_frame_42",
                item={"type": "function_call_output", "call_id": call.call_id, "output": "snapshot ready"},
            ),
        )

        assert image_events == []
        assert output_events == []
        prefetch = text_prompt_queue.get_nowait()
        assert isinstance(prefetch, GenerateResponseRequest)
        assert st.tool_followup_prefetch_request is prefetch
        assert [item.type for item in st.runtime_config.chat.buffer] == [
            "function_call",
            "message",
            "function_call_output",
        ]
        assert st.runtime_config.chat.buffer[1].content[0].type == "input_image"
        assert len(st.pending_item_acks) == 2
        image_item_id = st.runtime_config.chat.buffer[1].id
        assert image_item_id is not None

        # The prefetched LM finishes before the origin response's ordered TTS
        # path reaches response.done, but irreversible cleanup remains parked.
        assert prefetch.prefetch_transaction is not None
        prefetch.prefetch_transaction.complete(lambda: st.runtime_config.chat.strip_images({image_item_id}))
        assert st.runtime_config.chat.buffer[1].content[0].type == "input_image"

        terminal_events = service.finish_response(conn_id, response_key=response_key)
        created = [event for event in terminal_events if isinstance(event, ConversationItemCreatedEvent)]
        assert len(created) == 2
        assert created[0].previous_item_id == call.id
        assert created[0].item.id == image_item_id
        assert created[0].item.content[0].type == "input_image"
        assert created[0].item.content[0].image_url == "data:image/jpeg;base64,abc"
        assert created[1].previous_item_id == created[0].item.id
        assert isinstance(
            service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create")),
            ResponseCreatedEvent,
        )
        assert st.current_response_key == prefetch.response_key
        assert st.runtime_config.chat.buffer[1].content == []
        assert created[0].item.content[0].type == "input_image"

    def test_discarded_image_prefetch_does_not_consume_replacement_input(
        self,
        service,
        conn_id,
        text_prompt_queue,
    ):
        st = service._state(conn_id)
        image = RealtimeConversationItemUserMessage(
            type="message",
            role="user",
            content=[{"type": "input_image", "image_url": "data:image/jpeg;base64,abc"}],
        )
        st.runtime_config.chat.add_item(image)
        assert image.id is not None
        transaction = ResponsePrefetchTransaction()
        prefetch = GenerateResponseRequest(
            runtime_config=st.runtime_config,
            prefetch_transaction=transaction,
        )
        st.tool_followup_prefetch_request = prefetch
        st.tool_followup_prefetch_origin_response_key = "response_origin"
        st.mark_response_pending(prefetch.response_key)
        transaction.complete(lambda: st.runtime_config.chat.strip_images({image.id}))

        created = service.handle_response_create(
            conn_id,
            ResponseCreateEvent(type="response.create", response={"output_modalities": ["text"]}),
        )

        assert isinstance(created, ResponseCreatedEvent)
        replacement = text_prompt_queue.get_nowait()
        assert isinstance(replacement, GenerateResponseRequest)
        assert replacement.response_key != prefetch.response_key
        live_image = next(item for item in st.runtime_config.chat.buffer if item.id == image.id)
        assert live_image.content[0].type == "input_image"

    def test_prefetch_abort_failure_still_clears_response_state(self, service, conn_id):
        st = service._state(conn_id)
        transaction = ResponsePrefetchTransaction()
        prefetch = GenerateResponseRequest(
            runtime_config=st.runtime_config,
            prefetch_transaction=transaction,
        )
        st.tool_followup_prefetch_request = prefetch
        st.tool_followup_prefetch_origin_response_key = "response_origin"
        st.mark_response_pending(prefetch.response_key)

        def fail_to_close() -> None:
            raise RuntimeError("close failed")

        transaction.register_abort(fail_to_close)

        service.response.discard_tool_followup_prefetch(conn_id)

        assert st.tool_followup_prefetch_request is None
        assert st.tool_followup_prefetch_origin_response_key is None
        assert prefetch.response_key in st.closed_response_keys
        assert prefetch.response_key not in st.pending_response_keys

    def test_prefetch_claim_cleanup_failure_falls_back_to_fresh_generation(
        self,
        service,
        conn_id,
        text_prompt_queue,
    ):
        st = service._state(conn_id)
        transaction = ResponsePrefetchTransaction()
        prefetch = GenerateResponseRequest(
            runtime_config=st.runtime_config,
            prefetch_transaction=transaction,
        )
        st.tool_followup_prefetch_request = prefetch
        st.tool_followup_prefetch_origin_response_key = "response_origin"
        st.generation_done_tool_calls["response_origin"] = {"call_1"}
        st.mark_response_pending(prefetch.response_key)

        def fail_cleanup() -> None:
            raise RuntimeError("cleanup failed")

        transaction.complete(fail_cleanup)

        event = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))

        assert isinstance(event, ResponseCreatedEvent)
        assert st.tool_followup_prefetch_request is None
        assert st.tool_followup_prefetch_origin_response_key is None
        assert st.response_pending is False
        assert st.in_response is True
        assert prefetch.response_key in st.closed_response_keys
        assert "response_origin" not in st.generation_done_tool_calls
        replacement = text_prompt_queue.get_nowait()
        assert isinstance(replacement, GenerateResponseRequest)
        assert replacement.response_key != prefetch.response_key
        assert st.current_response_key == replacement.response_key

    @pytest.mark.parametrize("origin_active", [True, False])
    def test_failed_unclaimed_prefetch_is_discarded_before_standard_create(
        self,
        service,
        conn_id,
        text_prompt_queue,
        origin_active,
    ):
        st = service._state(conn_id)
        origin_key = "response_origin"
        if origin_active:
            st.in_response = True
            st.current_response_id = "resp_origin"
            st.current_response_key = origin_key
        st.generation_done_tool_calls[origin_key] = {"call_1"}
        prefetch = GenerateResponseRequest(
            runtime_config=st.runtime_config,
            prefetch_transaction=ResponsePrefetchTransaction(),
        )
        st.tool_followup_prefetch_request = prefetch
        st.tool_followup_prefetch_origin_response_key = origin_key
        st.mark_response_pending(prefetch.response_key)

        events = service.dispatch_pipeline_event(
            conn_id,
            ResponseGenerationDoneEvent(
                response_key=prefetch.response_key,
                succeeded=False,
            ),
        )

        assert events == []
        assert st.tool_followup_prefetch_request is None
        assert st.tool_followup_prefetch_origin_response_key is None
        assert prefetch.response_key in st.closed_response_keys
        assert prefetch.response_key not in st.pending_response_keys
        assert origin_key not in st.generation_done_tool_calls

        if origin_active:
            service.finish_response(conn_id, response_key=origin_key)
        created = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
        assert isinstance(created, ResponseCreatedEvent)
        replacement = text_prompt_queue.get_nowait()
        assert isinstance(replacement, GenerateResponseRequest)
        assert replacement.response_key != prefetch.response_key

    def test_discarded_prefetch_transaction_forces_immediate_create_fallback(
        self,
        service,
        conn_id,
        text_prompt_queue,
    ):
        st = service._state(conn_id)
        transaction = ResponsePrefetchTransaction()
        prefetch = GenerateResponseRequest(
            runtime_config=st.runtime_config,
            prefetch_transaction=transaction,
        )
        st.tool_followup_prefetch_request = prefetch
        st.tool_followup_prefetch_origin_response_key = "response_origin"
        st.generation_done_tool_calls["response_origin"] = {"call_1"}
        st.mark_response_pending(prefetch.response_key)
        transaction.discard()

        created = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))

        assert isinstance(created, ResponseCreatedEvent)
        replacement = text_prompt_queue.get_nowait()
        assert isinstance(replacement, GenerateResponseRequest)
        assert replacement.response_key != prefetch.response_key
        assert st.tool_followup_prefetch_request is None
        assert prefetch.response_key in st.closed_response_keys
        assert prefetch.response_key in st.closed_response_keys

    def test_prefetched_followup_preserves_logical_done_for_a_second_tool_round(
        self,
        service,
        conn_id,
        text_prompt_queue,
    ):
        st = service._state(conn_id)
        origin_key = "response_origin"
        first_call = RealtimeConversationItemFunctionCall(
            type="function_call",
            id="fc_first",
            call_id="call_first",
            name="first",
            arguments="{}",
        )
        assert st.runtime_config.chat.add_provisional_generation_items(origin_key, [first_call]) is not None
        st.in_response = True
        st.current_response_id = "resp_origin"
        st.current_response_key = origin_key
        st.pending_function_calls = {0: first_call}
        service.dispatch_pipeline_event(
            conn_id,
            ResponseGenerationDoneEvent(response_key=origin_key, call_ids=[first_call.call_id]),
        )
        service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={"type": "function_call_output", "call_id": first_call.call_id, "output": "first result"},
            ),
        )
        first_prefetch = text_prompt_queue.get_nowait()
        assert isinstance(first_prefetch, GenerateResponseRequest)

        second_call = RealtimeConversationItemFunctionCall(
            type="function_call",
            id="fc_second",
            call_id="call_second",
            name="second",
            arguments="{}",
        )
        assert (
            st.runtime_config.chat.add_provisional_generation_items(first_prefetch.response_key, [second_call])
            is not None
        )
        assert (
            service.dispatch_pipeline_event(
                conn_id,
                ResponseGenerationDoneEvent(
                    response_key=first_prefetch.response_key,
                    call_ids=[second_call.call_id],
                ),
            )
            == []
        )
        assert st.generation_done_tool_calls[first_prefetch.response_key] == {second_call.call_id}
        assert text_prompt_queue.empty()

        service.finish_response(conn_id, response_key=origin_key)
        assert isinstance(
            service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create")),
            ResponseCreatedEvent,
        )
        service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                response_key=first_prefetch.response_key,
                tools=[
                    {
                        "type": "function_call",
                        "id": second_call.id,
                        "call_id": second_call.call_id,
                        "name": second_call.name,
                        "arguments": second_call.arguments,
                    }
                ],
            ),
        )
        assert (
            service.handle_conversation_item_create(
                conn_id,
                ConversationItemCreateEvent(
                    type="conversation.item.create",
                    item={
                        "type": "function_call_output",
                        "call_id": second_call.call_id,
                        "output": "second result",
                    },
                ),
            )
            == []
        )

        second_prefetch = text_prompt_queue.get_nowait()
        assert isinstance(second_prefetch, GenerateResponseRequest)
        assert second_prefetch.response_key != first_prefetch.response_key
        assert st.tool_followup_prefetch_request is second_prefetch
        assert st.tool_followup_prefetch_origin_response_key == first_prefetch.response_key

    def test_late_generation_done_starts_prefetch_after_response_done(
        self,
        service,
        conn_id,
        text_prompt_queue,
    ):
        st = service._state(conn_id)
        response_key = "response_origin"
        call = RealtimeConversationItemFunctionCall(
            type="function_call",
            id="fc_lookup",
            call_id="call_lookup",
            name="lookup",
            arguments="{}",
        )
        assert st.runtime_config.chat.add_provisional_generation_items(response_key, [call]) is not None
        st.in_response = True
        st.current_response_id = "resp_origin"
        st.current_response_key = response_key
        st.pending_function_calls = {0: call}

        service.finish_response(conn_id, response_key=response_key)
        assert response_key in st.closed_response_keys
        assert response_key in st.completed_tool_response_keys

        service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={"type": "function_call_output", "call_id": call.call_id, "output": "result"},
            ),
        )
        assert text_prompt_queue.empty()

        service.dispatch_pipeline_event(
            conn_id,
            ResponseGenerationDoneEvent(response_key=response_key, call_ids=[call.call_id]),
        )

        prefetch = text_prompt_queue.get_nowait()
        assert isinstance(prefetch, GenerateResponseRequest)
        assert st.tool_followup_prefetch_request is prefetch
        assert response_key not in st.completed_tool_response_keys

    def test_late_generation_done_cannot_duplicate_an_already_started_followup(
        self,
        service,
        conn_id,
        text_prompt_queue,
    ):
        st = service._state(conn_id)
        response_key = "response_origin"
        call = RealtimeConversationItemFunctionCall(
            type="function_call",
            id="fc_lookup",
            call_id="call_lookup",
            name="lookup",
            arguments="{}",
        )
        assert st.runtime_config.chat.add_provisional_generation_items(response_key, [call]) is not None
        st.in_response = True
        st.current_response_id = "resp_origin"
        st.current_response_key = response_key
        st.pending_function_calls = {0: call}

        service.finish_response(conn_id, response_key=response_key)
        assert response_key in st.completed_tool_response_keys
        service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={"type": "function_call_output", "call_id": call.call_id, "output": "result"},
            ),
        )

        created = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
        assert isinstance(created, ResponseCreatedEvent)
        request = text_prompt_queue.get_nowait()
        assert isinstance(request, GenerateResponseRequest)

        deferred = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={
                "id": "msg_during_followup",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "new response context"}],
            },
        )
        assert service.handle_conversation_item_create(conn_id, deferred) == []
        assert len(st.deferred_items) == 1

        late_events = service.dispatch_pipeline_event(
            conn_id,
            ResponseGenerationDoneEvent(response_key=response_key, call_ids=[call.call_id]),
        )

        assert late_events == []
        assert text_prompt_queue.empty()
        assert st.tool_followup_prefetch_request is None
        assert st.current_response_key == request.response_key
        assert [item.id for item in st.deferred_items] == ["msg_during_followup"]

    def test_response_override_discards_incompatible_prefetch_and_generates_normally(
        self,
        service,
        conn_id,
        text_prompt_queue,
    ):
        st = service._state(conn_id)
        origin_key = "response_origin"
        call = RealtimeConversationItemFunctionCall(
            type="function_call",
            id="fc_lookup",
            call_id="call_lookup",
            name="lookup",
            arguments="{}",
        )
        assert st.runtime_config.chat.add_provisional_generation_items(origin_key, [call]) is not None
        st.in_response = True
        st.current_response_id = "resp_origin"
        st.current_response_key = origin_key
        st.pending_function_calls = {0: call}
        service.dispatch_pipeline_event(
            conn_id,
            ResponseGenerationDoneEvent(response_key=origin_key, call_ids=[call.call_id]),
        )
        service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={"type": "function_call_output", "call_id": call.call_id, "output": "result"},
            ),
        )
        prefetch = text_prompt_queue.get_nowait()
        nested_call = RealtimeConversationItemFunctionCall(
            type="function_call",
            id="fc_nested",
            call_id="call_nested",
            name="nested",
            arguments="{}",
        )
        assert st.runtime_config.chat.add_provisional_generation_items(prefetch.response_key, [nested_call]) is not None
        service.finish_response(conn_id, response_key=origin_key)

        created = service.handle_response_create(
            conn_id,
            ResponseCreateEvent(
                type="response.create",
                response={"output_modalities": ["text"], "instructions": "", "tools": []},
            ),
        )

        assert isinstance(created, ResponseCreatedEvent)
        request = text_prompt_queue.get_nowait()
        assert request.response_key != prefetch.response_key
        assert request.response is not None
        assert request.response.output_modalities == ["text"]
        assert request.response.instructions == ""
        assert request.response.tools == []
        assert prefetch.response_key in st.closed_response_keys
        assert not st.runtime_config.chat.has_pending_tool_calls()
        assert st.current_response_key == request.response_key

    def test_invalid_response_override_keeps_reusable_prefetch(
        self,
        service,
        conn_id,
    ):
        st = service._state(conn_id)
        transaction = ResponsePrefetchTransaction()
        prefetch = GenerateResponseRequest(
            runtime_config=st.runtime_config,
            prefetch_transaction=transaction,
        )
        st.tool_followup_prefetch_request = prefetch
        st.tool_followup_prefetch_origin_response_key = "response_origin"
        st.mark_response_pending(prefetch.response_key)

        rejected = service.handle_response_create(
            conn_id,
            ResponseCreateEvent(
                type="response.create",
                response={
                    "output_modalities": ["text"],
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "call_unknown",
                            "output": "invalid",
                        }
                    ],
                },
            ),
        )

        assert isinstance(rejected, RealtimeErrorEvent)
        assert rejected.error.type == "invalid_input_item"
        assert st.tool_followup_prefetch_request is prefetch
        assert st.response_pending is True
        assert prefetch.response_key not in st.closed_response_keys

        claimed = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
        assert isinstance(claimed, ResponseCreatedEvent)
        assert st.current_response_key == prefetch.response_key

    def test_context_change_restarts_unclaimed_tool_followup_prefetch(
        self,
        service,
        conn_id,
        text_prompt_queue,
    ):
        st = service._state(conn_id)
        call = RealtimeConversationItemFunctionCall(
            type="function_call",
            id="fc_lookup",
            call_id="call_lookup",
            name="lookup",
            arguments="{}",
        )
        assert st.runtime_config.chat.add_provisional_generation_items("response_origin", [call]) is not None
        service.dispatch_pipeline_event(
            conn_id,
            ResponseGenerationDoneEvent(response_key="response_origin", call_ids=[call.call_id]),
        )
        service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={"type": "function_call_output", "call_id": call.call_id, "output": "result"},
            ),
        )
        prefetch = text_prompt_queue.get_nowait()
        assert st.tool_followup_prefetch_request is prefetch
        st.generation_done_tool_calls[prefetch.response_key] = {"call_nested"}

        service.handle_conversation_item_create(
            conn_id,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "new context"}],
                },
            ),
        )

        replacement = text_prompt_queue.get_nowait()
        assert isinstance(replacement, GenerateResponseRequest)
        assert replacement.response_key != prefetch.response_key
        assert st.tool_followup_prefetch_request is replacement
        assert prefetch.response_key in st.closed_response_keys
        assert prefetch.response_key not in st.generation_done_tool_calls
        assert st.response_pending is True

    def test_response_create_while_implicit_response_pending(self, service, conn_id, text_prompt_queue):
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="implicit request"),
        )
        queued = text_prompt_queue.get_nowait()
        assert isinstance(queued, GenerateResponseRequest)
        assert service._state(conn_id).response_pending is True

        result = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))

        assert isinstance(result, RealtimeErrorEvent)
        assert result.error.type == "conversation_already_has_active_response"
        assert service._state(conn_id).current_response_id is None
        assert service._state(conn_id).response_pending is True
        assert text_prompt_queue.empty()

    def test_finishing_active_response_preserves_next_implicit_pending_key(self, service, conn_id, text_prompt_queue):
        service.response._ensure_response(conn_id, "response_a")
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="queued response B"),
        )
        request_b = text_prompt_queue.get_nowait()
        assert isinstance(request_b, GenerateResponseRequest)
        assert service._state(conn_id).pending_response_keys == {request_b.response_key}

        service.finish_response(conn_id, response_key="response_a")

        state = service._state(conn_id)
        assert state.response_pending is True
        assert state.pending_response_keys == {request_b.response_key}
        result = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
        assert isinstance(result, RealtimeErrorEvent)
        assert result.error.type == "conversation_already_has_active_response"

        created = service.response.on_assistant_response_done(
            conn_id,
            AssistantResponseDoneEvent(response_key=request_b.response_key),
        )
        assert isinstance(created[0], ResponseCreatedEvent)
        assert state.response_pending is False
        assert state.pending_response_keys == set()

    def test_response_create_stores_overrides(self, service, conn_id, runtime_config, text_prompt_queue):
        evt = ResponseCreateEvent(
            type="response.create",
            response={
                "instructions": "override instructions",
                "tool_choice": "auto",
            },
        )
        result = service.handle_response_create(conn_id, evt)
        assert isinstance(result, ResponseCreatedEvent)
        req = text_prompt_queue.get()
        assert isinstance(req, GenerateResponseRequest)
        assert req.response is not None
        assert req.response.instructions == "override instructions"
        assert req.response.tool_choice == "auto"
        assert req.runtime_config is runtime_config

    def test_response_create_preserves_latest_user_turn_timing(self, service, conn_id, text_prompt_queue):
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(
                transcript="hello",
                language_code="en",
                turn_id="turn_1",
                turn_revision=2,
                speech_stopped_at_s=123.0,
            ),
        )
        initial_req = text_prompt_queue.get()
        assert isinstance(initial_req, GenerateResponseRequest)
        assert initial_req.turn_id == "turn_1"
        assert initial_req.turn_revision == 2
        assert initial_req.speech_stopped_at_s == 123.0
        service.response._ensure_response(conn_id, initial_req.response_key)
        service.response._end_response(conn_id)

        result = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))

        assert isinstance(result, ResponseCreatedEvent)
        followup_req = text_prompt_queue.get()
        assert isinstance(followup_req, GenerateResponseRequest)
        assert followup_req.turn_id == "turn_1"
        assert followup_req.turn_revision == 2
        assert followup_req.speech_stopped_at_s == 123.0

    def test_response_create_rejects_complex_tool_choice(self, service, conn_id, runtime_config):
        evt = ResponseCreateEvent(
            type="response.create",
            response={
                "tool_choice": {"type": "function", "name": "my_func"},
            },
        )
        err = service.handle_response_create(conn_id, evt)
        assert isinstance(err, RealtimeErrorEvent)
        assert err.error.type == "tool_choice_not_supported"
        assert service._state(conn_id).in_response is False

    def test_response_create_accepts_valid_str_tool_choices(self, service, conn_id, text_prompt_queue):
        for choice in ("auto", "required", "none"):
            evt = ResponseCreateEvent(
                type="response.create",
                response={"tool_choice": choice},
            )
            result = service.handle_response_create(conn_id, evt)
            assert isinstance(result, ResponseCreatedEvent), f"Expected ResponseCreatedEvent for tool_choice={choice!r}"
            req = text_prompt_queue.get()
            assert isinstance(req, GenerateResponseRequest)
            assert req.response.tool_choice == choice
            service.response._end_response(conn_id)

    def test_response_create_with_image_input_items(self, service, conn_id, text_prompt_queue):
        evt = ResponseCreateEvent(
            type="response.create",
            response={
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Describe this image"},
                            {"type": "input_image", "image_url": "https://example.com/photo.jpg"},
                        ],
                    }
                ],
            },
        )
        result = service.handle_response_create(conn_id, evt)
        assert isinstance(result, ResponseCreatedEvent)
        gen_msg = text_prompt_queue.get()
        assert isinstance(gen_msg, GenerateResponseRequest)

    def test_response_create_rejects_invalid_function_call_output_in_input(self, service, conn_id, text_prompt_queue):
        evt = ResponseCreateEvent(
            type="response.create",
            response={
                "input": [
                    {"type": "function_call_output", "output": '{"x": 1}', "call_id": "call_bogus"},
                ],
            },
        )
        result = service.handle_response_create(conn_id, evt)
        assert isinstance(result, RealtimeErrorEvent)
        assert "call_bogus" in result.error.message
        assert service._state(conn_id).in_response is False

    def test_response_create_rejects_unresolved_function_call(self, service, conn_id, text_prompt_queue):
        chat = service._state(conn_id).runtime_config.chat
        chat.add_ordered_function_call(
            RealtimeConversationItemFunctionCall(
                type="function_call",
                call_id="call_pending",
                name="pending",
                arguments="{}",
            )
        )

        result = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))

        assert isinstance(result, RealtimeErrorEvent)
        assert result.error.type == "function_call_output_pending"
        assert service._state(conn_id).in_response is False
        assert text_prompt_queue.empty()

    def test_rejected_response_input_does_not_mutate_chat(self, service, conn_id, text_prompt_queue):
        chat = service._state(conn_id).runtime_config.chat
        chat.add_ordered_function_call(
            RealtimeConversationItemFunctionCall(
                type="function_call",
                call_id="call_pending",
                name="pending",
                arguments="{}",
            )
        )
        event = ResponseCreateEvent(
            type="response.create",
            response={"input": [self._user_input("must not survive rejection")]},
        )

        result = service.handle_response_create(conn_id, event)

        assert isinstance(result, RealtimeErrorEvent)
        assert result.error.type == "function_call_output_pending"
        assert [item.type for item in chat.buffer] == ["function_call"]
        assert chat.has_pending_tool_calls()
        assert text_prompt_queue.empty()

    def test_partially_resolved_response_input_does_not_mutate_chat(self, service, conn_id, text_prompt_queue):
        chat = service._state(conn_id).runtime_config.chat
        calls = [
            RealtimeConversationItemFunctionCall(
                type="function_call",
                call_id=f"call_pending_{index}",
                name="pending",
                arguments="{}",
            )
            for index in range(2)
        ]
        for call in calls:
            chat.add_ordered_function_call(call)
        event = ResponseCreateEvent(
            type="response.create",
            response={
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_pending_0",
                        "output": "done",
                    }
                ]
            },
        )

        result = service.handle_response_create(conn_id, event)

        assert isinstance(result, RealtimeErrorEvent)
        assert result.error.type == "function_call_output_pending"
        assert [item.type for item in chat.buffer] == ["function_call", "function_call"]
        assert all(call.status is None for call in calls)
        assert chat.has_pending_tool_calls()
        assert text_prompt_queue.empty()

    def test_invalid_response_input_batch_does_not_mutate_chat(self, service, conn_id, text_prompt_queue):
        chat = service._state(conn_id).runtime_config.chat
        event = ResponseCreateEvent(
            type="response.create",
            response={
                "input": [
                    self._user_input("must not survive invalid input"),
                    {
                        "type": "function_call_output",
                        "call_id": "call_missing",
                        "output": "done",
                    },
                ]
            },
        )

        result = service.handle_response_create(conn_id, event)

        assert isinstance(result, RealtimeErrorEvent)
        assert result.error.type == "invalid_input_item"
        assert chat.buffer == []
        assert text_prompt_queue.empty()

    def test_response_create_accepts_matching_function_output_in_input(self, service, conn_id, text_prompt_queue):
        chat = service._state(conn_id).runtime_config.chat
        chat.add_ordered_function_call(
            RealtimeConversationItemFunctionCall(
                type="function_call",
                call_id="call_pending",
                name="pending",
                arguments="{}",
            )
        )
        event = ResponseCreateEvent(
            type="response.create",
            response={
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_pending",
                        "output": "done",
                    }
                ]
            },
        )

        result = service.handle_response_create(conn_id, event)

        assert isinstance(result, ResponseCreatedEvent)
        assert not chat.has_pending_tool_calls()
        assert isinstance(text_prompt_queue.get_nowait(), GenerateResponseRequest)

    def test_double_response_create_rejected(self, service, conn_id, text_prompt_queue):
        """Second response.create is rejected because in_response is set immediately."""
        evt = ResponseCreateEvent(type="response.create")
        result1 = service.handle_response_create(conn_id, evt)
        assert isinstance(result1, ResponseCreatedEvent)
        result2 = service.handle_response_create(conn_id, evt)
        assert isinstance(result2, RealtimeErrorEvent)
        assert result2.error.type == "conversation_already_has_active_response"

    @staticmethod
    def _user_input(text):
        return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}

    def test_response_create_out_of_band_does_not_append_input_to_default_chat(
        self, service, conn_id, text_prompt_queue
    ):
        chat = service._state(conn_id).runtime_config.chat
        assert len(chat.buffer) == 0
        evt = ResponseCreateEvent(
            type="response.create",
            response={"conversation": "none", "input": [self._user_input("OOB question")]},
        )
        result = service.handle_response_create(conn_id, evt)
        assert isinstance(result, ResponseCreatedEvent)
        # Out-of-band: the default conversation is left untouched...
        assert len(chat.buffer) == 0
        # ...while the input still rides along on the queued request for the LM to use.
        req = text_prompt_queue.get()
        assert isinstance(req, GenerateResponseRequest)
        assert req.response.input is not None and len(req.response.input) == 1

    def test_response_create_in_band_appends_input_to_default_chat(self, service, conn_id, text_prompt_queue):
        chat = service._state(conn_id).runtime_config.chat
        evt = ResponseCreateEvent(type="response.create", response={"input": [self._user_input("in band")]})
        result = service.handle_response_create(conn_id, evt)
        assert isinstance(result, ResponseCreatedEvent)
        assert len(chat.buffer) == 1  # in-band input is threaded into the conversation

    def test_response_create_out_of_band_carries_null_turn(self, service, conn_id, text_prompt_queue):
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(
                transcript="hello",
                language_code="en",
                turn_id="turn_1",
                turn_revision=2,
                speech_stopped_at_s=123.0,
            ),
        )
        initial_req = text_prompt_queue.get()  # drain the STT-triggered request
        service.response._ensure_response(conn_id, initial_req.response_key)
        service.response._end_response(conn_id)

        result = service.handle_response_create(
            conn_id, ResponseCreateEvent(type="response.create", response={"conversation": "none"})
        )
        assert isinstance(result, ResponseCreatedEvent)
        req = text_prompt_queue.get()
        # Null turn identity makes every speculative-staleness gate treat it as always-latest.
        assert req.turn_id is None
        assert req.turn_revision is None
        assert req.speech_stopped_at_s is None

    def test_response_create_out_of_band_reports_null_conversation_id(self, service, conn_id):
        result = service.handle_response_create(
            conn_id, ResponseCreateEvent(type="response.create", response={"conversation": "none"})
        )
        assert isinstance(result, ResponseCreatedEvent)
        assert result.response.conversation_id is None
        done = [e for e in service.finish_response(conn_id) if isinstance(e, ResponseDoneEvent)]
        assert done and done[0].response.conversation_id is None

    @pytest.mark.parametrize("logical_done_before_response_done", [True, False])
    def test_out_of_band_tool_completion_never_starts_followup_prefetch(
        self,
        service,
        conn_id,
        text_prompt_queue,
        logical_done_before_response_done,
    ):
        result = service.handle_response_create(
            conn_id,
            ResponseCreateEvent(type="response.create", response={"conversation": "none"}),
        )
        assert isinstance(result, ResponseCreatedEvent)
        request = text_prompt_queue.get_nowait()
        call = RealtimeConversationItemFunctionCall(
            type="function_call",
            id="fc_oob",
            call_id="call_oob",
            name="lookup",
            arguments="{}",
        )
        state = service._state(conn_id)
        state.pending_function_calls[0] = call

        if not logical_done_before_response_done:
            service.finish_response(conn_id, response_key=request.response_key)
        events = service.dispatch_pipeline_event(
            conn_id,
            ResponseGenerationDoneEvent(response_key=request.response_key, call_ids=[call.call_id]),
        )

        assert events == []
        assert text_prompt_queue.empty()
        assert state.tool_followup_prefetch_request is None
        assert request.response_key not in state.generation_done_tool_calls
        assert request.response_key not in state.completed_tool_response_keys

    def test_response_create_in_band_reports_conversation_id(self, service, conn_id):
        result = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
        assert isinstance(result, ResponseCreatedEvent)
        assert result.response.conversation_id == service._state(conn_id).conversation_id


# ===================================================================
# Response cancel
# ===================================================================


class TestHandleResponseCancel:
    def test_cancel_active_response(self, service, conn_id, should_listen):
        should_listen.clear()
        service.response._ensure_response(conn_id)
        events = service.handle_response_cancel(conn_id)
        assert len(events) == 1
        assert isinstance(events[0], ResponseDoneEvent)
        assert events[0].response.status == "cancelled"
        assert events[0].response.status_details.reason == "client_cancelled"
        assert should_listen.is_set()

    def test_cancel_no_active_response(self, service, conn_id):
        events = service.handle_response_cancel(conn_id)
        assert events == []


# ===================================================================
# Outbound audio encoding
# ===================================================================


class TestEncodeAudioChunk:
    def test_begin_audio_output_reserves_assistant_item_for_media_transports(self, service, conn_id):
        _, item_id, output_index, events = service.begin_audio_output(conn_id)

        st = service._state(conn_id)
        assert isinstance(events[0], ResponseCreatedEvent)
        assert st.pending_assistant_item_id == item_id
        assert st.pending_assistant_output_index == output_index == 0
        assert st.last_item_id == item_id

        done = next(e for e in service.finish_response(conn_id) if isinstance(e, ResponseDoneEvent))
        assert done.response.output[0].id == item_id

    def test_first_chunk_emits_response_created_and_delta(self, service, conn_id):
        audio = _pcm_bytes(256)
        events = service.encode_audio_chunk(conn_id, audio)
        assert len(events) == 2
        assert isinstance(events[0], ResponseCreatedEvent)
        resp = events[0].response
        assert resp.status == "in_progress"
        assert resp.object == "realtime.response"
        assert resp.conversation_id is not None
        assert isinstance(events[1], ResponseAudioDeltaEvent)
        assert events[1].content_index == 0
        assert events[1].output_index == 0
        assert events[1].delta == base64.b64encode(audio).decode("ascii")

    def test_subsequent_chunks_keep_content_index(self, service, conn_id):
        service.encode_audio_chunk(conn_id, _pcm_bytes(256))  # first
        events = service.encode_audio_chunk(conn_id, _pcm_bytes(256))  # second
        assert len(events) == 1
        assert isinstance(events[0], ResponseAudioDeltaEvent)
        assert events[0].content_index == 0

    def test_response_created_includes_metadata(self, service, conn_id):
        from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

        service._state(conn_id).current_response_params = RealtimeResponseCreateParams(
            metadata={"key": "value"},
        )
        events = service.encode_audio_chunk(conn_id, _pcm_bytes(256))
        resp = events[0].response
        assert resp.metadata == {"key": "value"}


# ===================================================================
# Finish audio response
# ===================================================================


class TestFinishAudioResponse:
    def test_finish_without_audio_emits_only_response_done(self, service, conn_id):
        service.response._ensure_response(conn_id)
        events = service.finish_response(conn_id)
        assert len(events) == 1
        assert isinstance(events[0], ResponseDoneEvent)
        assert events[0].response.status == "completed"

    def test_finish_text_only_skips_audio_done(self, service, conn_id):
        from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

        service._state(conn_id).current_response_params = RealtimeResponseCreateParams(
            output_modalities=["text"],
        )
        service.response._ensure_response(conn_id)
        events = service.finish_response(conn_id)
        assert len(events) == 1
        assert isinstance(events[0], ResponseDoneEvent)
        assert events[0].response.status == "completed"
        assert not any(isinstance(e, ResponseAudioDoneEvent) for e in events)

    def test_finish_with_cancel_status(self, service, conn_id):
        service.response._ensure_response(conn_id)
        events = service.finish_response(conn_id, status="cancelled", reason="turn_detected")
        done = events[-1]
        assert done.response.status == "cancelled"
        assert done.response.status_details.reason == "turn_detected"

    def test_finish_resets_state(self, service, conn_id):
        from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

        service._state(conn_id).current_response_params = RealtimeResponseCreateParams(
            metadata={"k": "v"},
        )
        service.response._ensure_response(conn_id)
        service.finish_response(conn_id)
        st = service._state(conn_id)
        assert st.in_response is False
        assert st.current_response_id is None
        assert st.current_item_id is None
        assert st.current_response_params is None
        assert st.pending_assistant_item_id is None
        assert st.pending_assistant_output_index is None
        assert not st.pending_function_calls


class TestResponseDoneOutputItems:
    """response.done's response.output must carry the actual generated items,
    per https://platform.openai.com/docs/api-reference/realtime-server-events/session/updated —
    OpenAI's own docs: "response.done will also have the complete data we
    need to call our function." Without this, clients that read function
    calls from response.done (rather than only the incremental
    response.function_call_arguments.done event) never see them.
    """

    def test_output_includes_function_call_item(self, service, conn_id):
        service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                text="Sure, ending the call.",
                tools=[{"type": "function_call", "call_id": "call_1", "name": "endCall", "arguments": "{}"}],
            ),
        )
        events = service.finish_response(conn_id)
        done = next(e for e in events if isinstance(e, ResponseDoneEvent))
        function_calls = [
            item for item in done.response.output if isinstance(item, RealtimeConversationItemFunctionCall)
        ]
        assert len(function_calls) == 1
        assert function_calls[0].name == "endCall"
        assert function_calls[0].call_id == "call_1"
        assert function_calls[0].arguments == "{}"

    def test_function_call_only_response_skips_audio_done(self, service, conn_id):
        stream_events = service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                text="",
                tools=[
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "endCall",
                        "arguments": "{}",
                    }
                ],
            ),
        )

        terminal_events = service.finish_response(conn_id)
        done = next(e for e in terminal_events if isinstance(e, ResponseDoneEvent))
        function_event = next(e for e in stream_events if isinstance(e, ResponseFunctionCallArgumentsDoneEvent))

        item_done = next(e for e in stream_events if isinstance(e, ResponseOutputItemDoneEvent))
        assert [type(event) for event in terminal_events] == [ResponseDoneEvent]
        assert item_done.item.id == function_event.item_id
        assert [item.id for item in done.response.output] == [function_event.item_id]

    def test_output_includes_assistant_audio_message(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text="First sentence."))
        service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text="Second sentence."))
        events = service.finish_response(conn_id)
        done = next(e for e in events if isinstance(e, ResponseDoneEvent))
        messages = [item for item in done.response.output if isinstance(item, RealtimeConversationItemAssistantMessage)]
        assert len(messages) == 1
        assert messages[0].content[0].type == "output_audio"
        assert messages[0].content[0].transcript == "First sentence. Second sentence."

    def test_output_includes_assistant_text_message(self, service, conn_id):
        from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

        service._state(conn_id).current_response_params = RealtimeResponseCreateParams(
            output_modalities=["text"],
        )
        service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text="Hello there."))
        events = service.finish_response(conn_id)
        done = next(e for e in events if isinstance(e, ResponseDoneEvent))
        messages = [item for item in done.response.output if isinstance(item, RealtimeConversationItemAssistantMessage)]
        assert len(messages) == 1
        assert messages[0].content[0].type == "output_text"
        assert messages[0].content[0].text == "Hello there."

    def test_output_empty_when_response_has_no_content(self, service, conn_id):
        service.response._ensure_response(conn_id)
        events = service.finish_response(conn_id)
        done = next(e for e in events if isinstance(e, ResponseDoneEvent))
        assert not done.response.output

    def test_function_call_item_id_matches_its_arguments_done_event(self, service, conn_id):
        """A client correlating the streamed arguments event with the item in
        response.output by item_id must find the same id in both places.
        """
        stream_events = service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                text="One moment.",
                tools=[{"type": "function_call", "call_id": "call_1", "name": "endCall", "arguments": "{}"}],
            ),
        )
        args_done = next(e for e in stream_events if isinstance(e, ResponseFunctionCallArgumentsDoneEvent))

        done = next(e for e in service.finish_response(conn_id) if isinstance(e, ResponseDoneEvent))
        call_item = next(
            item for item in done.response.output if isinstance(item, RealtimeConversationItemFunctionCall)
        )
        assert call_item.id == args_done.item_id

    def test_every_output_item_has_a_distinct_id(self, service, conn_id):
        stream_events = service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                text="One moment.",
                tools=[
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "first",
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call",
                        "id": "fc_2",
                        "call_id": "call_2",
                        "name": "second",
                        "arguments": "{}",
                    },
                ],
            ),
        )
        args_done = [e for e in stream_events if isinstance(e, ResponseFunctionCallArgumentsDoneEvent)]
        done = next(e for e in service.finish_response(conn_id) if isinstance(e, ResponseDoneEvent))

        output_ids = [item.id for item in done.response.output]
        assert all(output_ids)
        assert len(set(output_ids)) == len(output_ids)
        assert [event.item_id for event in args_done] == ["fc_1", "fc_2"]
        assert output_ids[1:] == ["fc_1", "fc_2"]

    def test_output_indexes_match_final_items_across_pipeline_chunks(self, service, conn_id):
        text_events = service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text="One moment."))
        tool_events = service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                text="",
                tools=[
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "first",
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call",
                        "id": "fc_2",
                        "call_id": "call_2",
                        "name": "second",
                        "arguments": "{}",
                    },
                ],
            ),
        )
        terminal_events = service.finish_response(conn_id)
        done = next(e for e in terminal_events if isinstance(e, ResponseDoneEvent))
        output_events = [
            e
            for e in [*text_events, *tool_events]
            if isinstance(e, (ResponseAudioTranscriptDeltaEvent, ResponseFunctionCallArgumentsDoneEvent))
        ]

        assert [event.output_index for event in output_events] == [0, 1, 2]
        for event in output_events:
            assert done.response.output[event.output_index].id == event.item_id

    def test_output_order_is_preserved_when_tool_precedes_text(self, service, conn_id):
        tool_event = next(
            e
            for e in service.dispatch_pipeline_event(
                conn_id,
                AssistantOutputEvent(
                    text="",
                    tools=[
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "first",
                            "arguments": "{}",
                        }
                    ],
                ),
            )
            if isinstance(e, ResponseFunctionCallArgumentsDoneEvent)
        )
        text_event = next(
            e
            for e in service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text="After the call."))
            if isinstance(e, ResponseAudioTranscriptDeltaEvent)
        )
        terminal_events = service.finish_response(conn_id)
        done = next(e for e in terminal_events if isinstance(e, ResponseDoneEvent))

        assert tool_event.output_index == 0
        assert text_event.output_index == 1
        assert not any(isinstance(e, ResponseAudioDoneEvent) for e in terminal_events)
        assert [item.id for item in done.response.output] == [tool_event.item_id, text_event.item_id]

    def test_audio_delta_reuses_known_assistant_output_identity(self, service, conn_id):
        text_event = next(
            e
            for e in service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text="Speaking now."))
            if isinstance(e, ResponseAudioTranscriptDeltaEvent)
        )
        audio_delta = next(
            e for e in service.encode_audio_chunk(conn_id, _pcm_bytes(256)) if isinstance(e, ResponseAudioDeltaEvent)
        )
        done = next(e for e in service.finish_response(conn_id) if isinstance(e, ResponseDoneEvent))

        assert audio_delta.item_id == text_event.item_id
        assert audio_delta.output_index == text_event.output_index
        assert done.response.output[audio_delta.output_index].id == audio_delta.item_id

    def test_tool_boundary_closes_audio_first_output(self, service, conn_id):
        audio_delta = next(
            e for e in service.encode_audio_chunk(conn_id, _pcm_bytes(256)) if isinstance(e, ResponseAudioDeltaEvent)
        )
        tool_events = service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                tools=[
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "first",
                        "arguments": "{}",
                    }
                ],
            ),
        )
        audio_done = next(e for e in tool_events if isinstance(e, ResponseAudioDoneEvent))
        tool_event = next(e for e in tool_events if isinstance(e, ResponseFunctionCallArgumentsDoneEvent))
        text_event = next(
            e
            for e in service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text="Speaking now."))
            if isinstance(e, ResponseAudioTranscriptDeltaEvent)
        )

        terminal_events = service.finish_response(conn_id)
        done = next(e for e in terminal_events if isinstance(e, ResponseDoneEvent))

        assert audio_delta.output_index == audio_done.output_index == 0
        assert audio_delta.item_id == audio_done.item_id
        assert tool_event.output_index == 1
        assert text_event.output_index == 2
        assert [item.id for item in done.response.output] == [
            audio_delta.item_id,
            tool_event.item_id,
            text_event.item_id,
        ]

    def test_cancelled_audio_keeps_reserved_assistant_output_item(self, service, conn_id):
        tool_event = next(
            e
            for e in service.dispatch_pipeline_event(
                conn_id,
                AssistantOutputEvent(
                    text="",
                    tools=[
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "first",
                            "arguments": "{}",
                        }
                    ],
                ),
            )
            if isinstance(e, ResponseFunctionCallArgumentsDoneEvent)
        )
        audio_delta = next(
            e for e in service.encode_audio_chunk(conn_id, _pcm_bytes(256)) if isinstance(e, ResponseAudioDeltaEvent)
        )

        terminal_events = service.finish_response(conn_id, status="cancelled", reason="client_cancelled")
        done = next(e for e in terminal_events if isinstance(e, ResponseDoneEvent))

        assert tool_event.output_index == 0
        assert audio_delta.output_index == 1
        assert [item.id for item in done.response.output] == [tool_event.item_id, audio_delta.item_id]
        assert [item.status for item in done.response.output] == ["completed", "incomplete"]
        assistant = done.response.output[audio_delta.output_index]
        assert isinstance(assistant, RealtimeConversationItemAssistantMessage)
        assert assistant.content[0].type == "output_audio"
        assert assistant.content[0].transcript == ""

    def test_assistant_id_survives_non_interrupting_user_speech(self, service, conn_id):
        transcript_event = next(
            e
            for e in service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text="Still speaking."))
            if isinstance(e, ResponseAudioTranscriptDeltaEvent)
        )
        speech_event = next(
            e
            for e in service.dispatch_pipeline_event(
                conn_id,
                SpeechStartedEvent(interrupt_response=False),
            )
            if isinstance(e, InputAudioBufferSpeechStartedEvent)
        )
        done = next(e for e in service.finish_response(conn_id) if isinstance(e, ResponseDoneEvent))
        message = next(
            item for item in done.response.output if isinstance(item, RealtimeConversationItemAssistantMessage)
        )

        assert message.id == transcript_event.item_id
        assert message.id != speech_event.item_id

    def test_cancelled_response_preserves_completed_function_call(self, service, conn_id):
        service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                text="One moment.",
                tools=[{"type": "function_call", "call_id": "call_1", "name": "endCall", "arguments": "{}"}],
            ),
        )
        events = service.finish_response(conn_id, status="cancelled", reason="client_cancelled")
        done = next(e for e in events if isinstance(e, ResponseDoneEvent))
        assert done.response.output
        assert [item.status for item in done.response.output] == ["incomplete", "completed"]

    def test_cancelled_response_marks_unfinished_function_call_incomplete(self, service, conn_id):
        service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                text="",
                tools=[
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "endCall",
                        "arguments": "{}",
                        "status": "in_progress",
                    }
                ],
            ),
        )
        events = service.finish_response(conn_id, status="cancelled", reason="client_cancelled")
        item_done = next(e for e in events if isinstance(e, ResponseOutputItemDoneEvent))
        done = next(e for e in events if isinstance(e, ResponseDoneEvent))

        assert len(done.response.output) == 1
        assert item_done.item.id == done.response.output[0].id
        assert item_done.item.status == "incomplete"
        assert done.response.output[0].status == "incomplete"
        assert events.index(item_done) < events.index(done)


# ===================================================================
# Pipeline text translation
# ===================================================================


class TestDispatchPipelineEvent:
    # -- speech_started --

    def test_speech_started_emits_event(self, service, conn_id):
        events = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(),
        )
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, InputAudioBufferSpeechStartedEvent)
        assert evt.audio_start_ms == 0
        assert evt.item_id.startswith("item_")

    def test_speech_started_cancels_active_response(self, service, conn_id):
        service.response._ensure_response(conn_id)
        events = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(),
        )
        cancel_events = [e for e in events if isinstance(e, (ResponseAudioDoneEvent, ResponseDoneEvent))]
        assert len(cancel_events) == 1
        done = [e for e in cancel_events if isinstance(e, ResponseDoneEvent)][0]
        assert done.response.status == "cancelled"
        assert done.response.status_details.reason == "turn_detected"
        speech = [e for e in events if isinstance(e, InputAudioBufferSpeechStartedEvent)]
        assert len(speech) == 1

    def test_speech_started_no_response_emits_only_started(self, service, conn_id):
        """speech_started without active response emits only the started event."""
        events = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(),
        )
        assert len(events) == 1
        assert isinstance(events[0], InputAudioBufferSpeechStartedEvent)

    def test_speech_started_does_not_cancel_when_interrupt_disabled(self, service, conn_id):
        """With interrupt_response=False, speech_started emits the started event but does NOT cancel the active response."""
        from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

        service._state(conn_id).runtime_config.session.audio.input.turn_detection = ServerVad(
            type="server_vad",
            interrupt_response=False,
        )
        _, response_item_id = service.response._ensure_response(conn_id)
        events = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(),
        )
        assert len(events) == 1
        assert isinstance(events[0], InputAudioBufferSpeechStartedEvent)
        assert service._state(conn_id).in_response is True
        assert service._state(conn_id).current_item_id == response_item_id

    def test_speech_started_internal_non_interrupt_does_not_cancel(self, service, conn_id):
        _, response_item_id = service.response._ensure_response(conn_id)
        events = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(interrupt_response=False),
        )

        assert len(events) == 1
        assert isinstance(events[0], InputAudioBufferSpeechStartedEvent)
        assert service._state(conn_id).in_response is True
        assert service._state(conn_id).current_item_id == response_item_id
        done_events = service.finish_response(conn_id)
        assert len(done_events) == 1
        assert isinstance(done_events[0], ResponseDoneEvent)

    def test_consecutive_speech_cycles_get_distinct_item_ids(self, service, conn_id):
        """Each speech_started/stopped cycle generates a new unique item_id."""
        started_1 = service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        stopped_1 = service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent())

        started_2 = service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        stopped_2 = service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent())

        id_1 = started_1[0].item_id
        id_2 = started_2[0].item_id
        assert id_1 != id_2
        assert stopped_1[0].item_id == id_1
        assert stopped_2[0].item_id == id_2

    # -- speech_stopped --

    def test_speech_stopped_emits_event(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        events = service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(),
        )
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, InputAudioBufferSpeechStoppedEvent)
        assert evt.audio_end_ms == 0

    def test_speech_stopped_same_item_id_as_started(self, service, conn_id):
        started = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(),
        )
        stopped = service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(),
        )
        assert started[0].item_id == stopped[0].item_id

    def test_speech_stopped_stores_duration(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=2.5),
        )
        assert service._state(conn_id).input_audio_duration_s == 2.5

    def test_speech_stopped_zero_duration_not_stored(self, service, conn_id):
        """Phantom trigger (duration_s=0) emits stopped event but doesn't overwrite duration."""
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        events = service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(),
        )
        assert len(events) == 1
        assert isinstance(events[0], InputAudioBufferSpeechStoppedEvent)
        assert service._state(conn_id).input_audio_duration_s == 0.0

    # -- assistant_text --

    def test_assistant_text_emits_transcript_delta(self, service, conn_id):
        events = service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(text="Hello there"),
        )
        assert len(events) == 2
        assert isinstance(events[0], ResponseCreatedEvent)
        evt = events[1]
        assert isinstance(evt, ResponseAudioTranscriptDeltaEvent)
        assert evt.content_index == 0
        assert evt.output_index == 0
        assert evt.delta == "Hello there"

    def test_audio_transcript_deltas_match_single_terminal_done(self, service, conn_id):
        service.response._ensure_response(conn_id)
        first = service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text="Hello there."))
        second = service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text="How are you?"))
        service.encode_audio_chunk(conn_id, _pcm_bytes(256))

        deltas = [event for event in [*first, *second] if isinstance(event, ResponseAudioTranscriptDeltaEvent)]
        assert [event.delta for event in deltas] == ["Hello there.", " How are you?"]
        assert not any(isinstance(event, ResponseAudioTranscriptDoneEvent) for event in [*first, *second])

        terminal = service.finish_response(conn_id)
        transcript_done = [event for event in terminal if isinstance(event, ResponseAudioTranscriptDoneEvent)]
        response_done = next(event for event in terminal if isinstance(event, ResponseDoneEvent))
        assert len(transcript_done) == 1
        assert transcript_done[0].transcript == "".join(event.delta for event in deltas)
        assert (
            transcript_done[0].response_id,
            transcript_done[0].item_id,
            transcript_done[0].output_index,
            transcript_done[0].content_index,
        ) == (
            deltas[0].response_id,
            deltas[0].item_id,
            deltas[0].output_index,
            deltas[0].content_index,
        )
        assert response_done.response.output[transcript_done[0].output_index].id == transcript_done[0].item_id
        assert [event.type for event in terminal] == [
            "response.output_audio.done",
            "response.output_audio_transcript.done",
            "response.done",
        ]

    def test_audio_transcript_normalizes_chunk_whitespace_in_deltas_and_done(self, service, conn_id):
        service.response._ensure_response(conn_id)
        first = service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text="  Hello there.  \n"))
        whitespace = service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text=" \t\n"))
        second = service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text="  How are you?  "))

        deltas = [
            event for event in [*first, *whitespace, *second] if isinstance(event, ResponseAudioTranscriptDeltaEvent)
        ]
        assert [event.delta for event in deltas] == ["Hello there.", " How are you?"]

        terminal = service.finish_response(conn_id)
        transcript_done = next(event for event in terminal if isinstance(event, ResponseAudioTranscriptDoneEvent))
        assert transcript_done.transcript == "".join(event.delta for event in deltas) == "Hello there. How are you?"

    def test_cancelled_audio_transcript_emits_single_terminal_done(self, service, conn_id):
        service.response._ensure_response(conn_id)
        delta = service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text="partial"))[0]
        service.encode_audio_chunk(conn_id, _pcm_bytes(256))

        terminal = service.finish_response(conn_id, status="cancelled", reason="client_cancelled")
        transcript_done = [event for event in terminal if isinstance(event, ResponseAudioTranscriptDoneEvent)]
        response_done = next(event for event in terminal if isinstance(event, ResponseDoneEvent))

        assert isinstance(delta, ResponseAudioTranscriptDeltaEvent)
        assert len(transcript_done) == 1
        assert transcript_done[0].transcript == delta.delta == "partial"
        assert response_done.response.status == "cancelled"
        assert response_done.response.output[0].status == "incomplete"

    @pytest.mark.parametrize("status", ["failed", "incomplete"])
    def test_non_completed_audio_transcript_emits_single_terminal_done(self, service, conn_id, status):
        service.response._ensure_response(conn_id)
        delta = service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text="partial"))[0]
        service.encode_audio_chunk(conn_id, _pcm_bytes(256))

        terminal = service.finish_response(conn_id, status=status)
        transcript_done = [event for event in terminal if isinstance(event, ResponseAudioTranscriptDoneEvent)]
        response_done = next(event for event in terminal if isinstance(event, ResponseDoneEvent))

        assert len(transcript_done) == 1
        assert transcript_done[0].transcript == delta.delta == "partial"
        assert [event.type for event in terminal] == [
            "response.output_audio.done",
            "response.output_audio_transcript.done",
            "response.done",
        ]
        assert response_done.response.status == status
        assert response_done.response.output[0].status == "incomplete"

    def test_assistant_text_with_tools(self, service, conn_id):
        service.response._ensure_response(conn_id)
        events = service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                text="Let me check",
                tools=[
                    {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": '{"city": "Paris"}'},
                    {"type": "function_call", "call_id": "c2", "name": "get_time", "arguments": "{}"},
                ],
            ),
        )
        assert len(events) == 7
        assert isinstance(events[0], ResponseAudioTranscriptDeltaEvent)
        assert events[0].output_index == 0
        assert isinstance(events[1], ResponseOutputItemAddedEvent)
        assert events[1].item.call_id == "c1"
        assert isinstance(events[2], ResponseFunctionCallArgumentsDoneEvent)
        assert events[2].output_index == 1
        assert events[2].name == "get_weather"
        assert events[2].call_id == "c1"
        assert json.loads(events[2].arguments) == {"city": "Paris"}
        assert isinstance(events[3], ResponseOutputItemDoneEvent)
        assert events[3].item.status == "completed"
        assert events[3].item.call_id == "c1"
        assert isinstance(events[4], ResponseOutputItemAddedEvent)
        assert isinstance(events[5], ResponseFunctionCallArgumentsDoneEvent)
        assert events[5].output_index == 2
        assert isinstance(events[6], ResponseOutputItemDoneEvent)
        assert events[6].item.call_id == "c2"

    def test_assistant_text_tools_only(self, service, conn_id):
        service.response._ensure_response(conn_id)
        events = service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                text="",
                tools=[{"type": "function_call", "call_id": "c1", "name": "f1", "arguments": "{}"}],
            ),
        )
        assert len(events) == 3
        assert isinstance(events[0], ResponseOutputItemAddedEvent)
        assert isinstance(events[1], ResponseFunctionCallArgumentsDoneEvent)
        assert events[1].output_index == 0
        assert isinstance(events[2], ResponseOutputItemDoneEvent)
        assert events[2].output_index == 0

    def test_assistant_parts_preserve_tool_text_tool_text_order(self, service, conn_id):
        service.response._ensure_response(conn_id)
        events = service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                parts=[
                    AssistantToolCallPart(
                        tool={"type": "function_call", "call_id": "c1", "name": "first", "arguments": "{}"}
                    ),
                    AssistantTextPart(text="between"),
                    AssistantToolCallPart(
                        tool={"type": "function_call", "call_id": "c2", "name": "second", "arguments": "{}"}
                    ),
                    AssistantTextPart(text="after"),
                ]
            ),
        )

        assert [event.type for event in events] == [
            "response.output_item.added",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.output_audio_transcript.delta",
            "response.output_item.added",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.output_audio_transcript.delta",
        ]
        output_events = [
            event
            for event in events
            if isinstance(event, (ResponseAudioTranscriptDeltaEvent, ResponseFunctionCallArgumentsDoneEvent))
        ]
        assert [event.output_index for event in output_events] == [0, 1, 2, 3]
        assert len({event.item_id for event in output_events}) == 4

        terminal = service.finish_response(conn_id)
        transcript_done = [event for event in terminal if isinstance(event, ResponseAudioTranscriptDoneEvent)]
        response_done = next(event for event in terminal if isinstance(event, ResponseDoneEvent))
        assert [(event.output_index, event.transcript) for event in transcript_done] == [
            (1, "between"),
            (3, "after"),
        ]
        assert [item.type for item in response_done.response.output] == [
            "function_call",
            "message",
            "function_call",
            "message",
        ]
        assert [item.id for item in response_done.response.output] == [event.item_id for event in output_events]

    def test_assistant_part_indices_continue_across_pipeline_events(self, service, conn_id):
        service.response._ensure_response(conn_id)
        first = service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text="before"))
        second = service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(tools=[{"type": "function_call", "call_id": "c1", "name": "tool", "arguments": "{}"}]),
        )
        third = service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text="after"))

        function_event = next(event for event in second if isinstance(event, ResponseFunctionCallArgumentsDoneEvent))
        assert [first[0].output_index, function_event.output_index, third[0].output_index] == [0, 1, 2]
        assert len({first[0].item_id, function_event.item_id, third[0].item_id}) == 3

    def test_interleaved_audio_switches_output_identity_and_closes_each_item(self, service, conn_id):
        response_key = "response_1"
        first_text = service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(response_key=response_key, text="before"),
        )
        first_audio = [
            *service.encode_audio_chunk(conn_id, _pcm_bytes(256), response_key),
            *service.encode_audio_chunk(conn_id, _pcm_bytes(256), response_key),
        ]
        tool_events = service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                response_key=response_key,
                tools=[{"type": "function_call", "call_id": "c1", "name": "tool", "arguments": "{}"}],
            ),
        )
        second_text = service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(response_key=response_key, text="after"),
        )
        second_audio = service.encode_audio_chunk(conn_id, _pcm_bytes(256), response_key)
        terminal = service.finish_response(conn_id)

        deltas = [event for event in [*first_audio, *second_audio] if isinstance(event, ResponseAudioDeltaEvent)]
        audio_done = [event for event in [*tool_events, *terminal] if isinstance(event, ResponseAudioDoneEvent)]
        response_done = next(event for event in terminal if isinstance(event, ResponseDoneEvent))
        output_events = [
            event
            for event in [*first_text, *tool_events, *second_text]
            if isinstance(event, (ResponseAudioTranscriptDeltaEvent, ResponseFunctionCallArgumentsDoneEvent))
        ]
        assert [event.output_index for event in output_events] == [0, 1, 2]
        assert [event.output_index for event in deltas] == [0, 0, 2]
        assert [event.content_index for event in deltas] == [0, 0, 0]
        assert [event.output_index for event in audio_done] == [0, 2]
        assert [item.id for item in response_done.response.output] == [event.item_id for event in output_events]

    def test_early_tool_call_waits_for_preceding_text_but_not_its_tts(self, service, conn_id):
        response_key = "response_1"
        tool = AssistantToolCallPart(tool={"type": "function_call", "call_id": "c1", "name": "tool", "arguments": "{}"})

        # The side channel may outrun the TTS queue, but cannot overtake the
        # preceding assistant part in the public response.
        assert (
            service.dispatch_pipeline_event(
                conn_id,
                AssistantToolCallReadyEvent(
                    response_key=response_key,
                    output_sequence=1,
                    part=tool,
                ),
            )
            == []
        )
        events = service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                response_key=response_key,
                output_sequence=0,
                parts=[AssistantTextPart(text="One moment.")],
            ),
        )

        assert [event.type for event in events] == [
            "response.created",
            "response.output_audio_transcript.delta",
            "response.output_item.added",
            "response.function_call_arguments.done",
            "response.output_item.done",
        ]
        assert [event.output_index for event in events[1:]] == [0, 1, 1, 1]

        # Audio can continue after the function event. The ordered copy of the
        # tool call closes it later without exposing a duplicate call.
        service.encode_audio_chunk(conn_id, _pcm_bytes(256), response_key)
        ordered_tool = service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                response_key=response_key,
                output_sequence=1,
                parts=[tool],
            ),
        )
        assert [event.type for event in ordered_tool] == ["response.output_audio.done"]

        terminal = service.finish_response(conn_id)
        response_done = next(event for event in terminal if isinstance(event, ResponseDoneEvent))
        assert [item.type for item in response_done.response.output] == ["message", "function_call"]

    def test_later_audio_does_not_close_silent_intermediate_outputs(self, service, conn_id):
        response_key = "response_1"
        service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(response_key=response_key, text="a"),
        )
        first_audio = service.encode_audio_chunk(conn_id, _pcm_bytes(256), response_key)
        first_tool = service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                response_key=response_key,
                tools=[{"type": "function_call", "call_id": "c1", "name": "first", "arguments": "{}"}],
            ),
        )
        service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(response_key=response_key, text="b"))
        service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                response_key=response_key,
                tools=[{"type": "function_call", "call_id": "c2", "name": "second", "arguments": "{}"}],
            ),
        )
        service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(response_key=response_key, text="c"))
        last_audio = service.encode_audio_chunk(conn_id, _pcm_bytes(256), response_key)
        terminal = service.finish_response(conn_id)

        lifecycle = [
            (event.type, event.output_index)
            for event in [*first_audio, *first_tool, *last_audio, *terminal]
            if isinstance(event, (ResponseAudioDeltaEvent, ResponseAudioDoneEvent))
        ]
        assert lifecycle == [
            ("response.output_audio.delta", 0),
            ("response.output_audio.done", 0),
            ("response.output_audio.delta", 4),
            ("response.output_audio.done", 4),
        ]

    def test_text_only_interleaving_closes_each_text_item_before_response_done(self, service, conn_id):
        from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

        service._state(conn_id).current_response_params = RealtimeResponseCreateParams(
            output_modalities=["text"],
        )
        service.response._ensure_response(conn_id)
        events = service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                parts=[
                    AssistantTextPart(text="before"),
                    AssistantToolCallPart(
                        tool={"type": "function_call", "call_id": "c1", "name": "tool", "arguments": "{}"}
                    ),
                    AssistantTextPart(text="after"),
                ]
            ),
        )

        assert [event.type for event in events] == [
            "response.output_text.delta",
            "response.output_item.added",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.output_text.delta",
        ]
        assert [event.output_index for event in events] == [0, 1, 1, 1, 2]

        done_events = service.finish_response(conn_id)
        text_done = [event for event in done_events if isinstance(event, ResponseTextDoneEvent)]
        assert [(event.output_index, event.text) for event in text_done] == [(0, "before"), (2, "after")]
        response_done = done_events[-1]
        assert isinstance(response_done, ResponseDoneEvent)
        assert [item.type for item in response_done.response.output] == ["message", "function_call", "message"]

    def test_assistant_text_text_only_emits_text_events(self, service, conn_id):
        from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

        service._state(conn_id).current_response_params = RealtimeResponseCreateParams(
            output_modalities=["text"],
        )
        service.response._ensure_response(conn_id)
        events = service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(text="Hello there"),
        )
        # on_assistant_output streams only the delta now; the matching done is
        # emitted once at close in finish_response.
        assert len(events) == 1
        assert isinstance(events[0], ResponseTextDeltaEvent)
        assert events[0].content_index == 0
        assert events[0].output_index == 0
        assert events[0].delta == "Hello there"
        assert not any(isinstance(e, ResponseTextDoneEvent) for e in events)
        assert not any(isinstance(e, ResponseAudioTranscriptDoneEvent) for e in events)

        done_events = service.finish_response(conn_id)
        text_done = [e for e in done_events if isinstance(e, ResponseTextDoneEvent)]
        assert len(text_done) == 1
        assert text_done[0].content_index == 0
        assert text_done[0].output_index == 0
        assert text_done[0].text == "Hello there"

    def test_text_only_done_concatenates_streamed_parts(self, service, conn_id):
        from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

        service._state(conn_id).current_response_params = RealtimeResponseCreateParams(
            output_modalities=["text"],
        )
        service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text="Hello there. "))
        service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text="How are you?"))
        done_events = service.finish_response(conn_id)
        text_done = [e for e in done_events if isinstance(e, ResponseTextDoneEvent)]
        assert len(text_done) == 1
        # done.text concatenates the raw streamed parts verbatim (== sum of deltas).
        assert text_done[0].text == "Hello there. How are you?"

    def test_text_only_preserves_standalone_whitespace_deltas(self, service, conn_id):
        from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

        service._state(conn_id).current_response_params = RealtimeResponseCreateParams(
            output_modalities=["text"],
        )

        events = []
        for text in ("Hello", " ", "world", "\n"):
            events.extend(service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text=text)))

        deltas = [event.delta for event in events if isinstance(event, ResponseTextDeltaEvent)]
        assert deltas == ["Hello", " ", "world", "\n"]
        done_events = service.finish_response(conn_id)
        text_done = next(event for event in done_events if isinstance(event, ResponseTextDoneEvent))
        assert text_done.text == "Hello world\n"

    def test_text_only_emits_text_done_on_cancel(self, service, conn_id):
        from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

        service._state(conn_id).current_response_params = RealtimeResponseCreateParams(
            output_modalities=["text"],
        )
        service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text="partial"))
        done_events = service.finish_response(conn_id, status="cancelled", reason="client_cancelled")
        text_done = next(event for event in done_events if isinstance(event, ResponseTextDoneEvent))
        assert text_done.text == "partial"
        assert any(isinstance(e, ResponseDoneEvent) for e in done_events)

    def test_assistant_text_text_only_keeps_tool_events(self, service, conn_id):
        from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

        service._state(conn_id).current_response_params = RealtimeResponseCreateParams(
            output_modalities=["text"],
        )
        service.response._ensure_response(conn_id)
        events = service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                text="Let me check",
                tools=[{"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{}"}],
            ),
        )
        # No per-chunk done anymore: delta, then the tool event at output_index 1.
        assert isinstance(events[0], ResponseTextDeltaEvent)
        assert not any(isinstance(e, ResponseTextDoneEvent) for e in events)
        assert isinstance(events[1], ResponseOutputItemAddedEvent)
        tool_event = events[2]
        assert isinstance(tool_event, ResponseFunctionCallArgumentsDoneEvent)
        assert tool_event.output_index == 1
        assert tool_event.name == "get_weather"
        assert isinstance(events[3], ResponseOutputItemDoneEvent)
        assert events[3].item.call_id == "c1"

    def test_assistant_text_waits_for_pending_reopen_and_drops_confirmed_stale_turn(
        self,
        runtime_config,
        should_listen,
    ):
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(should_listen=should_listen, speculative_turns=tracker)
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config
        tracker.observe("turn_1", 0)
        candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
        done = Event()
        result = {}

        def dispatch():
            result["events"] = service.dispatch_pipeline_event(
                conn_id,
                AssistantOutputEvent(text="stale", turn_id="turn_1", turn_revision=0),
            )
            done.set()

        thread = Thread(target=dispatch)
        thread.start()

        assert not done.wait(0.05)
        assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)
        assert done.wait(1.0)
        thread.join(timeout=1.0)

        assert result["events"] == []
        assert service._state(conn_id).current_response_id is None
        service.unregister(conn_id)

    def test_assistant_text_waits_for_pending_reopen_and_emits_cancelled_reopen(
        self,
        runtime_config,
        should_listen,
    ):
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(should_listen=should_listen, speculative_turns=tracker)
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config
        tracker.observe("turn_1", 0)
        candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
        done = Event()
        result = {}

        def dispatch():
            result["events"] = service.dispatch_pipeline_event(
                conn_id,
                AssistantOutputEvent(text="latest", turn_id="turn_1", turn_revision=0),
            )
            done.set()

        thread = Thread(target=dispatch)
        thread.start()

        assert not done.wait(0.05)
        tracker.cancel_reopen_candidate("turn_1", candidate_revision)
        assert done.wait(1.0)
        thread.join(timeout=1.0)

        assert len(result["events"]) == 2
        assert isinstance(result["events"][0], ResponseCreatedEvent)
        assert isinstance(result["events"][1], ResponseAudioTranscriptDeltaEvent)
        assert result["events"][1].delta == "latest"
        assert tracker.is_committed("turn_1", 0)
        service.unregister(conn_id)

    def test_token_usage_does_not_wait_for_speculative_reopen(
        self,
        runtime_config,
        should_listen,
    ):
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(should_listen=should_listen, speculative_turns=tracker)
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config
        tracker.observe("turn_1", 0)
        candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
        done = Event()
        result = {}

        def dispatch():
            result["events"] = service.dispatch_pipeline_event(
                conn_id,
                TokenUsageEvent(input_tokens=10, output_tokens=5, turn_id="turn_1", turn_revision=0),
            )
            done.set()

        thread = Thread(target=dispatch)
        thread.start()

        assert done.wait(1.0)
        assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)
        thread.join(timeout=1.0)

        assert result["events"] == []
        assert service._state(conn_id).response_usage.input_tokens == 10
        assert service._state(conn_id).response_usage.output_tokens == 5
        service.unregister(conn_id)

    def test_try_dispatch_assistant_text_defers_pending_reopen(self, runtime_config, should_listen):
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(should_listen=should_listen, speculative_turns=tracker)
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config
        tracker.observe("turn_1", 0)
        candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)

        event = AssistantOutputEvent(text="latest", turn_id="turn_1", turn_revision=0)

        assert service.try_dispatch_pipeline_event(conn_id, event) is None
        assert service._state(conn_id).current_response_id is None

        tracker.cancel_reopen_candidate("turn_1", candidate_revision)
        events = service.try_dispatch_pipeline_event(conn_id, event)

        assert events is not None
        assert len(events) == 2
        assert isinstance(events[0], ResponseCreatedEvent)
        assert isinstance(events[1], ResponseAudioTranscriptDeltaEvent)
        assert events[1].delta == "latest"
        assert tracker.is_committed("turn_1", 0)
        service.unregister(conn_id)

    def test_try_dispatch_assistant_text_defers_reopen_grace(self, runtime_config, should_listen):
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(should_listen=should_listen, speculative_turns=tracker)
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config
        tracker.observe("turn_1", 0)
        tracker.start_reopen_grace("turn_1", 0, grace_s=0.05)

        event = AssistantOutputEvent(text="latest", turn_id="turn_1", turn_revision=0)

        assert service.should_defer_pipeline_event(event)
        assert service.try_dispatch_pipeline_event(conn_id, event) is None
        assert service._state(conn_id).current_response_id is None

        sleep(0.06)
        events = service.try_dispatch_pipeline_event(conn_id, event)

        assert events is not None
        assert len(events) == 2
        assert isinstance(events[0], ResponseCreatedEvent)
        assert isinstance(events[1], ResponseAudioTranscriptDeltaEvent)
        assert events[1].delta == "latest"
        assert tracker.is_committed("turn_1", 0)
        service.unregister(conn_id)

    def test_try_dispatch_token_usage_ignores_pending_reopen(self, runtime_config, should_listen):
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(should_listen=should_listen, speculative_turns=tracker)
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config
        tracker.observe("turn_1", 0)
        candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)

        event = TokenUsageEvent(input_tokens=10, output_tokens=5, turn_id="turn_1", turn_revision=0)

        assert service.try_dispatch_pipeline_event(conn_id, event) == []
        assert service._state(conn_id).response_usage.input_tokens == 10

        assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)
        assert service._state(conn_id).response_usage.output_tokens == 5
        service.unregister(conn_id)

    # -- partial_transcription --

    def test_partial_transcription_emits_stable_incremental_deltas_for_one_content_part(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        first = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="hello brave"),
        )
        second = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="hello brave new"),
        )
        third = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="hello brave new world"),
        )

        assert first == []
        assert isinstance(second[0], ConversationItemInputAudioTranscriptionDeltaEvent)
        assert second[0].content_index == 0
        assert second[0].delta == "hello"
        assert isinstance(third[0], ConversationItemInputAudioTranscriptionDeltaEvent)
        assert third[0].content_index == 0
        assert third[0].delta == " brave"

    def test_duplicate_partial_does_not_confirm_speculative_words(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        first = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="hello brave"),
        )
        duplicate = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="hello brave"),
        )
        extended = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="hello brave new"),
        )

        assert first == []
        assert duplicate == []
        assert extended[0].delta == "hello"

    def test_punctuation_revision_does_not_block_later_stable_deltas(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())

        first = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="Hey can you"),
        )
        second = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="Hey, can you hear"),
        )
        third = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="Hey, can you hear me?"),
        )

        assert first == []
        assert second[0].delta == "Hey can"
        assert third[0].delta == " you"

    def test_early_word_revision_is_replaced_before_it_reaches_the_wire(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())

        wrong = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="Nothing company."),
        )
        correction = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="Oh, nothing confused me."),
        )
        extended = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="Oh, nothing confused me. I was"),
        )

        assert wrong == []
        assert correction == []
        assert extended[0].delta == "Oh nothing confused"

    def test_stabilizer_resumes_after_one_incompatible_hypothesis(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(conn_id, PartialTranscriptionEvent(delta="The quick"))
        committed = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="The quick brown"),
        )
        revised = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="The swift brown"),
        )
        recovered = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="The swift brown fox"),
        )

        assert committed[0].delta == "The"
        assert revised == []
        assert recovered[0].delta == " swift"

    def test_new_input_item_resets_partial_transcription_delta(self, service, conn_id):
        first_started = service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(turn_id="turn_1"))
        service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="hello", turn_id="turn_1"),
        )

        second_started = service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(turn_id="turn_2"))
        service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="new item", turn_id="turn_2"),
        )
        second_partial = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="new item text", turn_id="turn_2"),
        )

        assert first_started[0].item_id != second_started[0].item_id
        assert second_partial[0].delta == "new"
        assert second_partial[0].content_index == 0

    def test_reopened_completed_turn_starts_a_new_transcription_item(self, service, conn_id):
        first_started = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="hello", turn_id="turn_1", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="hello", turn_id="turn_1", turn_revision=0),
        )

        reopened = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=1, reopened=True),
        )
        first_reopened_partial = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="hello again", turn_id="turn_1", turn_revision=1),
        )
        continued = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="hello again friend", turn_id="turn_1", turn_revision=1),
        )

        assert reopened[0].item_id != first_started[0].item_id
        assert first_reopened_partial == []
        assert continued[0].item_id == reopened[0].item_id
        assert continued[0].delta == "hello"
        assert continued[0].content_index == 0

    def test_completed_input_item_does_not_emit_later_deltas(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(turn_id="turn_1"))
        service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="hello", turn_id="turn_1"),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="hello", turn_id="turn_1"),
        )

        events = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="hello again", turn_id="turn_1"),
        )

        assert events == []

    def test_idless_partial_after_completion_does_not_borrow_assistant_item(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="hello"),
        )
        service.response._ensure_response(conn_id)
        assistant_item_id = service._state(conn_id).current_item_id

        events = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="late"),
        )

        assert events == []
        assert service._state(conn_id).current_item_id == assistant_item_id
        assert assistant_item_id not in service._state(conn_id).input_items

    def test_late_completion_and_overlapping_partial_keep_originating_item(self, service, conn_id):
        first_started = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1"),
        )
        service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="hello", turn_id="turn_1"),
        )
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=1.0, turn_id="turn_1"),
        )

        second_started = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_2"),
        )
        first_second_partial = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="world today", turn_id="turn_2"),
        )
        late_first_completion = service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="hello", turn_id="turn_1"),
        )
        second_second_partial = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="world today again", turn_id="turn_2"),
        )
        third_second_partial = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="world today again please", turn_id="turn_2"),
        )

        first_item_id = first_started[0].item_id
        second_item_id = second_started[0].item_id
        assert first_item_id != second_item_id
        assert late_first_completion[0].item_id == first_item_id
        assert late_first_completion[0].usage.seconds == 1.0
        assert first_second_partial == []
        assert second_second_partial[0].item_id == second_item_id
        assert second_second_partial[0].delta == "world"
        assert third_second_partial[0].item_id == second_item_id
        assert third_second_partial[0].delta == " today"
        input_items = service._state(conn_id).input_items
        assert first_item_id not in input_items
        assert input_items[second_item_id].transcript_prefix == "world today"

    def test_metadata_less_completion_keeps_the_current_input_item(self, service, conn_id):
        started = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="current", turn_id="turn_1", turn_revision=0),
        )

        completed = service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="metadata-less"),
        )

        current_item_id = started[0].item_id
        state = service._state(conn_id)
        assert completed[0].item_id == current_item_id
        assert current_item_id not in state.input_items
        assert state.current_input_item_id is None
        assert state.input_item_by_turn_revision == {}

    # -- transcription_completed --

    def test_transcription_completed_without_speech_start_preserves_legacy_fallback(
        self,
        service,
        conn_id,
        runtime_config,
        text_prompt_queue,
    ):
        events = service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(
                transcript="standalone final",
                turn_id="turn_1",
                turn_revision=2,
            ),
        )

        assert len(events) == 1
        assert isinstance(events[0], ConversationItemInputAudioTranscriptionCompletedEvent)
        assert events[0].transcript == "standalone final"
        assert service._state(conn_id).input_item_by_turn_revision == {}
        user_items = [item for item in runtime_config.chat.buffer if getattr(item, "role", None) == "user"]
        assert [item.content[0].text for item in user_items] == ["standalone final"]
        request = text_prompt_queue.get_nowait()
        assert request.turn_id == "turn_1"
        assert request.turn_revision == 2

    def test_transcription_completed_emits_event(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=3.2),
        )
        events = service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="hello world"),
        )
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, ConversationItemInputAudioTranscriptionCompletedEvent)
        assert evt.content_index == 0
        assert evt.transcript == "hello world"
        assert evt.usage.seconds == 3.2
        assert evt.usage.type == "duration"
        assert service._state(conn_id).response_pending is True

    def test_audio_input_completed_marks_response_pending_and_preserves_duration(
        self,
        service,
        conn_id,
        runtime_config,
        text_prompt_queue,
    ):
        audio = np.zeros(40000, dtype=np.float32)
        started = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=2.5, turn_id="turn_1", turn_revision=0),
        )

        events = service.dispatch_pipeline_event(
            conn_id,
            AudioInputCompletedEvent(
                audio=audio,
                audio_sample_rate=16000,
                audio_duration_s=2.5,
                turn_id="turn_1",
                turn_revision=0,
            ),
        )

        assert events == []
        state = service._state(conn_id)
        assert state.response_pending is True
        assert state.response_usage.audio_duration_s == 2.5
        assert state.current_input_item_id is None
        assert state.input_items == {}
        assert state.input_item_by_turn_revision == {}
        assert started[0].item_id not in state.input_items
        request = text_prompt_queue.get_nowait()
        assert isinstance(request, GenerateResponseRequest)
        assert request.runtime_config is runtime_config
        assert np.array_equal(request.audio, audio)
        assert request.audio_sample_rate == 16000
        assert request.turn_id == "turn_1"
        assert request.turn_revision == 0

        service.response._ensure_response(conn_id)
        assert state.input_audio_duration_s == 0.0
        assert state.response_usage.audio_duration_s == 2.5

    def test_empty_transcription_completed_emits_event_without_response(
        self,
        service,
        conn_id,
        runtime_config,
        text_prompt_queue,
    ):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=1.1),
        )
        events = service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="", language_code="en"),
        )

        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, ConversationItemInputAudioTranscriptionCompletedEvent)
        assert evt.transcript == ""
        assert evt.usage.seconds == 1.1
        assert text_prompt_queue.empty()
        assert runtime_config.chat.buffer == []
        assert service._state(conn_id).response_pending is False

    def test_revised_transcription_replaces_speculative_user_message(self, runtime_config, should_listen):
        text_prompt_queue = Queue()
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(
            text_prompt_queue=text_prompt_queue,
            should_listen=should_listen,
            speculative_turns=tracker,
        )
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config

        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=1.0, turn_id="turn_1", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="hello", turn_id="turn_1", turn_revision=0),
        )

        tracker.observe("turn_1", 1)
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=1, reopened=True),
        )
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=2.0, turn_id="turn_1", turn_revision=1),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="hello again", turn_id="turn_1", turn_revision=1),
        )

        user_items = [item for item in runtime_config.chat.buffer if getattr(item, "role", None) == "user"]
        assert len(user_items) == 1
        assert user_items[0].content[0].text == "hello again"
        first_req = text_prompt_queue.get_nowait()
        second_req = text_prompt_queue.get_nowait()
        assert first_req.turn_revision == 0
        assert second_req.turn_revision == 1
        assert service._state(conn_id).response_usage.audio_duration_s == 2.0
        service.unregister(conn_id)

    def test_empty_revised_transcription_removes_speculative_user_message(self, runtime_config, should_listen):
        text_prompt_queue = Queue()
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(
            text_prompt_queue=text_prompt_queue,
            should_listen=should_listen,
            speculative_turns=tracker,
        )
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config

        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=1.0, turn_id="turn_1", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="hello", turn_id="turn_1", turn_revision=0),
        )

        tracker.observe("turn_1", 1)
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=1, reopened=True),
        )
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=2.0, turn_id="turn_1", turn_revision=1),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="", turn_id="turn_1", turn_revision=1),
        )

        user_items = [item for item in runtime_config.chat.buffer if getattr(item, "role", None) == "user"]
        assert user_items == []
        first_req = text_prompt_queue.get_nowait()
        assert first_req.turn_revision == 0
        assert text_prompt_queue.empty()
        assert service._state(conn_id).response_usage.audio_duration_s == 2.0
        service.unregister(conn_id)

    def test_empty_first_revision_tracks_audio_for_later_nonempty_reopen(self, runtime_config, should_listen):
        text_prompt_queue = Queue()
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(
            text_prompt_queue=text_prompt_queue,
            should_listen=should_listen,
            speculative_turns=tracker,
        )
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config

        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=1.0, turn_id="turn_1", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="", turn_id="turn_1", turn_revision=0),
        )

        tracker.observe("turn_1", 1)
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=1, reopened=True),
        )
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=2.0, turn_id="turn_1", turn_revision=1),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="hello again", turn_id="turn_1", turn_revision=1),
        )

        user_items = [item for item in runtime_config.chat.buffer if getattr(item, "role", None) == "user"]
        assert len(user_items) == 1
        assert user_items[0].content[0].text == "hello again"
        req = text_prompt_queue.get_nowait()
        assert req.turn_revision == 1
        assert text_prompt_queue.empty()
        assert service._state(conn_id).response_usage.audio_duration_s == 2.0
        service.unregister(conn_id)

    def test_reopened_stt_reuses_item_when_old_completion_is_dropped(
        self,
        runtime_config,
        should_listen,
    ):
        text_prompt_queue = Queue()
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(
            text_prompt_queue=text_prompt_queue,
            should_listen=should_listen,
            speculative_turns=tracker,
        )
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config

        first_started = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="hello", turn_id="turn_1", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=1.0, turn_id="turn_1", turn_revision=0),
        )

        tracker.observe("turn_1", 1)
        second_started = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=1, reopened=True),
        )
        # BaseSTTHandler drops revision 0 after the reopen, so no terminal event
        # for that revision is dispatched here.
        first_reopened_partial = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="hello again", turn_id="turn_1", turn_revision=1),
        )
        second_partial = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="hello again friend", turn_id="turn_1", turn_revision=1),
        )
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=2.0, turn_id="turn_1", turn_revision=1),
        )
        completed = service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="hello again", turn_id="turn_1", turn_revision=1),
        )

        item_id = first_started[0].item_id
        assert second_started[0].item_id == item_id
        assert first_reopened_partial == []
        assert second_partial[0].item_id == item_id
        assert second_partial[0].delta == "hello"
        assert completed[0].item_id == item_id
        state = service._state(conn_id)
        assert item_id not in state.input_items
        assert state.current_input_item_id is None
        assert state.input_item_by_turn_revision == {}
        assert state.response_usage.audio_duration_s == 2.0
        user_items = [item for item in runtime_config.chat.buffer if getattr(item, "role", None) == "user"]
        assert len(user_items) == 1
        assert user_items[0].content[0].text == "hello again"
        assert text_prompt_queue.get_nowait().turn_revision == 1
        assert text_prompt_queue.empty()
        service.unregister(conn_id)

    def test_stale_transcription_revision_is_ignored(self, runtime_config, should_listen):
        text_prompt_queue = Queue()
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(
            text_prompt_queue=text_prompt_queue,
            should_listen=should_listen,
            speculative_turns=tracker,
        )
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config
        tracker.observe("turn_1", 1)

        events = service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="stale", turn_id="turn_1", turn_revision=0),
        )

        assert events == []
        assert runtime_config.chat.buffer == []
        assert text_prompt_queue.empty()
        assert service._state(conn_id).input_item_by_turn_revision == {}
        assert service._state(conn_id).input_items == {}
        service.unregister(conn_id)

    def test_stale_assistant_text_dropped_after_unanswered_reopen(self, runtime_config, should_listen):
        text_prompt_queue = Queue()
        tracker = SpeculativeTurnTracker()
        service = RealtimeService(
            text_prompt_queue=text_prompt_queue,
            should_listen=should_listen,
            speculative_turns=tracker,
        )
        conn_id = service.register()
        service._state(conn_id).runtime_config = runtime_config

        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=1.0, turn_id="turn_1", turn_revision=0),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="hello", turn_id="turn_1", turn_revision=0),
        )

        # The VAD reopens an unanswered turn past the grace window through the
        # same candidate protocol it uses for an in-grace reopen.
        candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
        assert candidate_revision == 1
        assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)

        events = service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(text="stale", turn_id="turn_1", turn_revision=0),
        )

        assert events == []
        assert service._state(conn_id).current_response_id is None

        service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(turn_id="turn_1", turn_revision=1, reopened=True),
        )
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=2.5, turn_id="turn_1", turn_revision=1),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="hello and more", turn_id="turn_1", turn_revision=1),
        )

        user_items = [item for item in runtime_config.chat.buffer if getattr(item, "role", None) == "user"]
        assert len(user_items) == 1
        assert user_items[0].content[0].text == "hello and more"
        service.unregister(conn_id)

    # -- response_failed --

    def test_response_failed_emits_error_and_failed_done(self, service, conn_id):
        service.response._ensure_response(conn_id)
        events = service.dispatch_pipeline_event(
            conn_id,
            ResponseFailedEvent(message="input must not be empty"),
        )
        # A top-level error carries the reason; the audio terminal closes the response.
        err = events[0]
        assert isinstance(err, RealtimeErrorEvent)
        assert err.error.message == "input must not be empty"
        assert err.error.type == "response_failed"
        assert service._state(conn_id).response_failed is True

        done = service.finish_response(conn_id)
        assert len(done) == 1
        assert done[0].response.status == "failed"
        assert done[0].response.status_details.error.type == "response_failed"
        # Slot released so the next response is not locked out.
        assert service._state(conn_id).in_response is False

    def test_failed_text_only_response_emits_text_done(self, service, conn_id):
        from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

        state = service._state(conn_id)
        state.current_response_params = RealtimeResponseCreateParams(output_modalities=["text"])
        service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text="partial"))
        service.dispatch_pipeline_event(conn_id, ResponseFailedEvent(message="provider failed"))

        events = service.finish_response(conn_id)

        text_done = next(event for event in events if isinstance(event, ResponseTextDoneEvent))
        response_done = next(event for event in events if isinstance(event, ResponseDoneEvent))
        assert text_done.text == "partial"
        assert response_done.response.status == "failed"
        assert response_done.response.output[0].status == "incomplete"
        assert response_done.response.status_details.error.type == "response_failed"

    def test_response_failed_while_pending_emits_error_and_failed_done(self, service, conn_id):
        service.dispatch_pipeline_event(
            conn_id,
            AudioInputCompletedEvent(
                audio=np.zeros(1600, dtype=np.float32),
                audio_duration_s=0.1,
            ),
        )
        assert service._state(conn_id).response_pending is True

        events = service.dispatch_pipeline_event(
            conn_id,
            ResponseFailedEvent(message="provider rejected audio"),
        )

        assert [event.type for event in events] == [
            "response.created",
            "error",
        ]
        created = events[0]
        assert isinstance(created, ResponseCreatedEvent)
        err = events[1]
        assert isinstance(err, RealtimeErrorEvent)
        assert err.error.message == "provider rejected audio"
        done = service.finish_response(conn_id)
        assert len(done) == 1
        assert done[0].response.status == "failed"
        assert done[0].response.id == created.response.id
        assert done[0].response.output == []
        state = service._state(conn_id)
        assert state.response_pending is False
        assert state.in_response is False

    def test_response_failed_without_active_response_is_noop(self, service, conn_id):
        # No active response (e.g. already closed): nothing to fail, emit nothing.
        events = service.dispatch_pipeline_event(
            conn_id,
            ResponseFailedEvent(message="too late"),
        )
        assert events == []

    def test_keyed_response_failure_survives_previous_response_completion(self, service, conn_id):
        service.response._ensure_response(conn_id, "first")
        service._state(conn_id).response_pending = True

        service.finish_response(conn_id, response_key="first")
        assert service._state(conn_id).response_pending is False

        events = service.dispatch_pipeline_event(
            conn_id,
            ResponseFailedEvent(
                message="second response failed",
                response_key="second",
                cancel_generation=2,
            ),
        )

        assert [event.type for event in events] == [
            "response.created",
            "error",
        ]
        done = service.finish_response(conn_id, response_key="second")
        assert len(done) == 1
        assert isinstance(done[0], ResponseDoneEvent)
        assert done[0].response.status == "failed"

    # -- unknown --

    def test_unknown_type_returns_empty(self, service, conn_id):
        from speech_to_speech.pipeline.events import PipelineEvent

        events = service.dispatch_pipeline_event(conn_id, PipelineEvent(type="something_else"))
        assert events == []


# ===================================================================
# Error helper
# ===================================================================


class TestMakeError:
    def test_make_error(self, service):
        err = service.make_error("oops", "my_error")
        assert isinstance(err, RealtimeErrorEvent)
        assert err.error.message == "oops"
        assert err.error.type == "my_error"
        assert err.event_id.startswith("event_")


# ===================================================================
# ID and state management
# ===================================================================


class TestIdAndStateManagement:
    def test_last_item_id_tracks_all_items(self, service, conn_id):
        st = service._state(conn_id)
        assert st.last_item_id is None

        # 1) speech_started sets last_item_id via dispatch_pipeline_event
        events = service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        input_id = events[0].item_id
        assert st.last_item_id == input_id

        # 2) assistant_text sets last_item_id via dispatch_pipeline_event
        events = service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text="hi"))
        output_id = st.current_item_id
        assert st.last_item_id == output_id

        # 3) handle_conversation_item_create updates last_item_id
        service.response._end_response(conn_id)
        evt = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={
                "id": "msg_manual",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "x"}],
            },
        )
        events = service.handle_conversation_item_create(conn_id, evt)
        assert st.last_item_id == events[0].item.id
        assert events[0].previous_item_id == output_id

    def test_content_index_stays_on_single_content_part(self, service, conn_id):
        service.response._start_item(conn_id)
        assert service.response._next_content_index(conn_id) == 0
        assert service.response._next_content_index(conn_id) == 0

        service.response._start_item(conn_id)
        assert service.response._next_content_index(conn_id) == 0

        service.response._ensure_response(conn_id)
        assert service.response._next_content_index(conn_id) == 0
        assert service.response._next_content_index(conn_id) == 0

        service.response._end_response(conn_id)
        service.response._ensure_response(conn_id)
        assert service.response._next_content_index(conn_id) == 0


# ===================================================================
# interrupt_response_enabled property
# ===================================================================


class TestInterruptResponseEnabled:
    def test_default_true_when_no_turn_detection(self, runtime_config):
        runtime_config.session.audio.input.turn_detection = None
        assert runtime_config.interrupt_response_enabled is True

    def test_true_when_server_vad_interrupt_true(self, runtime_config):
        from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

        runtime_config.session.audio.input.turn_detection = ServerVad(
            type="server_vad",
            interrupt_response=True,
        )
        assert runtime_config.interrupt_response_enabled is True

    def test_false_when_server_vad_interrupt_false(self, runtime_config):
        from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

        runtime_config.session.audio.input.turn_detection = ServerVad(
            type="server_vad",
            interrupt_response=False,
        )
        assert runtime_config.interrupt_response_enabled is False

    def test_default_true_when_server_vad_interrupt_none(self, runtime_config):
        from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

        runtime_config.session.audio.input.turn_detection = ServerVad(
            type="server_vad",
            interrupt_response=None,
        )
        assert runtime_config.interrupt_response_enabled is True

    def test_reads_dict_turn_detection(self, runtime_config):
        runtime_config.session.audio.input.turn_detection = {
            "type": "server_vad",
            "interrupt_response": False,
        }
        assert runtime_config.interrupt_response_enabled is False

    def test_dict_defaults_to_true(self, runtime_config):
        runtime_config.session.audio.input.turn_detection = {
            "type": "server_vad",
        }
        assert runtime_config.interrupt_response_enabled is True


# ===================================================================
# Usage metrics tracking (tokens + audio duration)
# ===================================================================


class TestUsageMetricsTracking:
    # -- token accumulation --

    def test_token_usage_accumulates_in_conn_state(self, service, conn_id):
        service.response._ensure_response(conn_id)
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=10, output_tokens=20),
        )
        usage = service._state(conn_id).response_usage
        assert usage.input_tokens == 10
        assert usage.output_tokens == 20

    def test_token_usage_accumulates_multiple(self, service, conn_id):
        service.response._ensure_response(conn_id)
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=5, output_tokens=10),
        )
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=3, output_tokens=7),
        )
        usage = service._state(conn_id).response_usage
        assert usage.input_tokens == 8
        assert usage.output_tokens == 17

    def test_token_usage_emits_no_events(self, service, conn_id):
        events = service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=10, output_tokens=20),
        )
        assert events == []

    def test_keyed_usage_waits_for_implicit_response_and_created_stays_zero(self, service, conn_id):
        response_key = "response_1"
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(response_key=response_key, input_tokens=11, output_tokens=7),
        )
        assert service._state(conn_id).response_usage.input_tokens == 0

        created = service.response.on_assistant_response_done(
            conn_id,
            AssistantResponseDoneEvent(response_key=response_key),
        )[0]
        assert isinstance(created, ResponseCreatedEvent)
        assert created.response.usage.input_tokens == 0
        assert created.response.usage.output_tokens == 0

        done = service.finish_response(conn_id, response_key=response_key)[0]
        assert isinstance(done, ResponseDoneEvent)
        assert done.response.usage.input_tokens == 11
        assert done.response.usage.output_tokens == 7

    def test_usage_for_closed_response_does_not_leak_into_next_response(self, service, conn_id):
        service.response._ensure_response(conn_id, "cancelled_response")
        service.finish_response(conn_id, status="cancelled", response_key="cancelled_response")
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(response_key="cancelled_response", input_tokens=11, output_tokens=7),
        )

        service.response._ensure_response(conn_id, "next_response")
        done = service.finish_response(conn_id, response_key="next_response")[0]

        assert isinstance(done, ResponseDoneEvent)
        assert done.response.usage.input_tokens == 0
        assert done.response.usage.output_tokens == 0
        assert service.total_usage.input_tokens == 11
        assert service.total_usage.output_tokens == 7

    def test_pending_response_cancellation_preserves_reported_usage_globally(self, service, conn_id):
        response_key = "pending_response"
        state = service._state(conn_id)
        state.mark_response_pending(response_key)
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(response_key=response_key, input_tokens=13, output_tokens=5),
        )
        service.dispatch_pipeline_event(
            conn_id,
            AssistantToolCallReadyEvent(
                response_key=response_key,
                output_sequence=1,
                part=AssistantToolCallPart(
                    tool={"type": "function_call", "call_id": "call_stale", "name": "lookup", "arguments": "{}"}
                ),
            ),
        )
        assert state.pending_early_tool_calls

        service.close_pending_responses(conn_id)

        assert state.pending_token_usage == {}
        assert state.pending_early_tool_calls == {}
        assert state.next_assistant_output_sequence == 0
        assert service.total_usage.input_tokens == 13
        assert service.total_usage.output_tokens == 5

    def test_response_done_reflects_token_usage(self, service, conn_id):
        service.response._ensure_response(conn_id)
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=100, output_tokens=50),
        )
        events = service.finish_response(conn_id)
        done_evt = events[-1]
        assert isinstance(done_evt, ResponseDoneEvent)
        assert done_evt.response.usage.input_tokens == 100
        assert done_evt.response.usage.output_tokens == 50
        assert done_evt.response.usage.total_tokens == 150

    def test_response_created_has_zero_tokens(self, service, conn_id):
        """ResponseCreatedEvent is emitted before any tokens are produced."""
        events = service.encode_audio_chunk(conn_id, _pcm_bytes(256))
        created_evt = events[0]
        assert isinstance(created_evt, ResponseCreatedEvent)
        assert created_evt.response.usage.input_tokens == 0
        assert created_evt.response.usage.output_tokens == 0
        assert created_evt.response.usage.total_tokens == 0

    def test_end_response_rolls_into_global(self, service, conn_id):
        service.response._ensure_response(conn_id)
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=10, output_tokens=20),
        )
        service.response._end_response(conn_id)
        assert service.total_usage.input_tokens == 10
        assert service.total_usage.output_tokens == 20
        usage = service._state(conn_id).response_usage
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    def test_multiple_responses_accumulate_global(self, service, conn_id):
        service.response._ensure_response(conn_id)
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=10, output_tokens=20),
        )
        service.response._end_response(conn_id)

        service.response._ensure_response(conn_id)
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=5, output_tokens=15),
        )
        service.response._end_response(conn_id)

        assert service.total_usage.input_tokens == 15
        assert service.total_usage.output_tokens == 35

    def test_unregister_rolls_partial_tokens_into_global(self, service):
        cid = service.register()
        service.response._ensure_response(cid)
        service.dispatch_pipeline_event(
            cid,
            TokenUsageEvent(input_tokens=7, output_tokens=3),
        )
        service.unregister(cid)
        assert service.total_usage.input_tokens == 7
        assert service.total_usage.output_tokens == 3

    def test_unregister_without_active_response_no_leak(self, service):
        cid = service.register()
        service.unregister(cid)
        assert service.total_usage.input_tokens == 0
        assert service.total_usage.output_tokens == 0

    def test_finish_response_resets_per_response_tokens(self, service, conn_id):
        """After finish_response, per-response counters are zero."""
        service.response._ensure_response(conn_id)
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=50, output_tokens=25),
        )
        service.finish_response(conn_id)
        usage = service._state(conn_id).response_usage
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert service.total_usage.input_tokens == 50
        assert service.total_usage.output_tokens == 25

    # -- audio duration accumulation --

    def test_transcription_completed_accumulates_duration(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent(duration_s=2.5))
        service.dispatch_pipeline_event(conn_id, TranscriptionCompletedEvent(transcript="hi"))
        assert service._state(conn_id).response_usage.audio_duration_s == 2.5

    def test_multiple_transcriptions_accumulate_duration(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent(duration_s=1.0))
        service.dispatch_pipeline_event(conn_id, TranscriptionCompletedEvent(transcript="a"))

        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent(duration_s=2.0))
        service.dispatch_pipeline_event(conn_id, TranscriptionCompletedEvent(transcript="b"))

        assert service._state(conn_id).response_usage.audio_duration_s == 3.0

    def test_end_response_rolls_duration_into_global(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent(duration_s=4.0))
        service.dispatch_pipeline_event(conn_id, TranscriptionCompletedEvent(transcript="x"))
        service.response._ensure_response(conn_id)
        service.response._end_response(conn_id)
        assert service.total_usage.audio_duration_s == 4.0
        assert service._state(conn_id).response_usage.audio_duration_s == 0.0

    def test_unregister_rolls_duration_into_global(self, service):
        cid = service.register()
        service.dispatch_pipeline_event(cid, SpeechStartedEvent())
        service.dispatch_pipeline_event(cid, SpeechStoppedEvent(duration_s=1.5))
        service.dispatch_pipeline_event(cid, TranscriptionCompletedEvent(transcript="y"))
        service.unregister(cid)
        assert service.total_usage.audio_duration_s == 1.5

    # -- responses_completed / responses_cancelled --

    def test_responses_completed_increments(self, service, conn_id):
        service.response._ensure_response(conn_id)
        service.finish_response(conn_id)
        assert service.total_usage.responses_completed == 1
        assert service.total_usage.responses_cancelled == 0

    def test_responses_cancelled_increments(self, service, conn_id):
        service.response._ensure_response(conn_id)
        service.finish_response(conn_id, status="cancelled", reason="turn_detected")
        assert service.total_usage.responses_cancelled == 1
        assert service.total_usage.responses_completed == 0

    def test_multiple_responses_accumulate_status_counters(self, service, conn_id):
        service.response._ensure_response(conn_id)
        service.finish_response(conn_id)
        service.response._ensure_response(conn_id)
        service.finish_response(conn_id, status="cancelled", reason="client_cancelled")
        service.response._ensure_response(conn_id)
        service.finish_response(conn_id)
        assert service.total_usage.responses_completed == 2
        assert service.total_usage.responses_cancelled == 1

    # -- tool_calls --

    def test_tool_calls_increments(self, service, conn_id):
        service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                text="",
                tools=[
                    {"type": "function_call", "call_id": "c1", "name": "f1", "arguments": "{}"},
                    {"type": "function_call", "call_id": "c2", "name": "f2", "arguments": "{}"},
                ],
            ),
        )
        assert service._state(conn_id).response_usage.tool_calls == 2

    def test_tool_calls_rolls_into_global(self, service, conn_id):
        service.response._ensure_response(conn_id)
        service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                text="",
                tools=[{"type": "function_call", "call_id": "c1", "name": "f1", "arguments": "{}"}],
            ),
        )
        service.finish_response(conn_id)
        assert service.total_usage.tool_calls == 1
        assert service._state(conn_id).response_usage.tool_calls == 0

    # -- connections --

    def test_connections_increments(self, service):
        assert service.total_usage.connections == 0
        cid1 = service.register()
        assert service.total_usage.connections == 1
        cid2 = service.register()
        assert service.total_usage.connections == 2
        service.unregister(cid1)
        service.unregister(cid2)

    # -- turns --

    def test_turns_increments(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        assert service._state(conn_id).response_usage.turns == 3

    def test_turns_rolls_into_global(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.response._ensure_response(conn_id)
        service.response._end_response(conn_id)
        assert service.total_usage.turns == 1
        assert service._state(conn_id).response_usage.turns == 0

    # -- errors_by_type --

    def test_errors_by_type_increments(self, service):
        service.make_error("msg", "type_a")
        service.make_error("msg", "type_a")
        service.make_error("msg", "type_b")
        assert service.total_usage.errors_by_type == {"type_a": 2, "type_b": 1}

    def test_total_errors_in_get_usage(self, service):
        service.make_error("msg", "type_a")
        service.make_error("msg", "type_b")
        usage = service.get_usage()
        assert usage["total_errors"] == 2
        assert usage["errors_by_type"] == {"type_a": 1, "type_b": 1}

    # -- get_usage --

    def test_get_usage(self, service, conn_id):
        # Speech cycle before response so speech_started doesn't cancel anything
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent(duration_s=3.0))
        service.dispatch_pipeline_event(conn_id, TranscriptionCompletedEvent(transcript="z"))

        service.response._ensure_response(conn_id)
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=10, output_tokens=20),
        )
        service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                text="hi",
                tools=[{"type": "function_call", "call_id": "c1", "name": "f1", "arguments": "{}"}],
            ),
        )
        service.finish_response(conn_id)
        service.make_error("oops", "some_error")
        usage = service.get_usage()
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 20
        assert usage["total_tokens"] == 30
        assert usage["audio_duration_s"] == 3.0
        assert usage["responses_completed"] == 1
        assert usage["responses_cancelled"] == 0
        assert usage["tool_calls"] == 1
        assert usage["turns"] == 1
        assert usage["connections"] >= 1
        assert usage["total_errors"] == 1
        assert usage["errors_by_type"] == {"some_error": 1}


# ===================================================================
# Chat image lifecycle
# ===================================================================


class TestChatImageLifecycle:
    """Tests for Chat.strip_images()."""

    def _make_chat(self):
        from speech_to_speech.LLM.chat import Chat

        return Chat(size=10)

    def _user_msg(self, *parts):
        from openai.types.realtime.realtime_conversation_item_user_message import (
            Content as UserContent,
        )
        from openai.types.realtime.realtime_conversation_item_user_message import (
            RealtimeConversationItemUserMessage,
        )

        content = []
        for p in parts:
            if p[0] == "text":
                content.append(UserContent(type="input_text", text=p[1]))
            elif p[0] == "image":
                content.append(UserContent(type="input_image", image_url=p[1]))
        return RealtimeConversationItemUserMessage(type="message", role="user", content=content)

    def test_strip_images_removes_image_parts(self):
        from speech_to_speech.LLM.chat import make_assistant_message

        chat = self._make_chat()
        chat.add_item(self._user_msg(("text", "What is this?"), ("image", "data:image/png;base64,abc")))
        chat.add_item(make_assistant_message("It's a cat."))
        chat.strip_images()
        user_msg = chat.buffer[0]
        assert len(user_msg.content) == 1
        assert user_msg.content[0].type == "input_text"
        assert user_msg.content[0].text == "What is this?"

    def test_strip_images_noop_on_text_only(self):
        from speech_to_speech.LLM.chat import make_assistant_message, make_user_message

        chat = self._make_chat()
        chat.add_item(make_user_message("hello"))
        chat.add_item(make_assistant_message("hi"))
        chat.strip_images()
        assert chat.buffer[0].content[0].text == "hello"
        assert chat.buffer[1].content[0].text == "hi"

    def test_strip_then_new_image_cycle(self):
        from speech_to_speech.LLM.chat import make_assistant_message

        chat = self._make_chat()
        chat.add_item(self._user_msg(("text", "look"), ("image", "old_url")))
        chat.add_item(make_assistant_message("I see it."))
        chat.strip_images()
        assert len(chat.buffer[0].content) == 1
        assert chat.buffer[0].content[0].type == "input_text"

        chat.add_item(self._user_msg(("text", "now this"), ("image", "new_url")))
        last_user = chat.buffer[-1]
        assert any(p.image_url == "new_url" for p in last_user.content)


# ===================================================================
# Chat tool call tracking
# ===================================================================


class TestChatToolCallTracking:
    """Tests for Chat._pending_tool_calls and append_tool_output."""

    def _make_chat(self, size=10):
        from speech_to_speech.LLM.chat import Chat

        return Chat(size=size)

    def _fc(self, call_id="call_1", name="f1"):
        from openai.types.realtime.realtime_conversation_item_function_call import (
            RealtimeConversationItemFunctionCall,
        )

        if not call_id.startswith("call_"):
            call_id = f"call_{call_id}"
        return RealtimeConversationItemFunctionCall(type="function_call", call_id=call_id, name=name, arguments="{}")

    def _fco(self, call_id="call_1"):
        from openai.types.realtime.realtime_conversation_item_function_call_output import (
            RealtimeConversationItemFunctionCallOutput,
        )

        if not call_id.startswith("call_"):
            call_id = f"call_{call_id}"
        return RealtimeConversationItemFunctionCallOutput(
            type="function_call_output", call_id=call_id, output='{"ok": true}'
        )

    def _user(self, text):
        from speech_to_speech.LLM.chat import make_user_message

        return make_user_message(text)

    def _assistant(self, text):
        from speech_to_speech.LLM.chat import make_assistant_message

        return make_assistant_message(text)

    def test_add_item_registers_pending_tool_call(self):
        chat = self._make_chat()
        fc = self._fc()
        chat.add_item(fc)
        assert "call_1" in chat._pending_tool_calls
        assert chat._pending_tool_calls["call_1"] is fc

    def test_append_tool_output_clears_pending(self):
        chat = self._make_chat()
        chat.add_item(self._fc())
        assert "call_1" in chat._pending_tool_calls
        chat.append_tool_output("call_1", self._fco())
        assert "call_1" not in chat._pending_tool_calls
        assert chat.buffer[-1].type == "function_call_output"

    def test_append_tool_output_reinjects_evicted_call(self):
        chat = self._make_chat(size=1)
        chat.add_item(self._user("hi"))
        chat.add_item(self._fc("call_x"))
        chat.add_item(self._assistant("ok"))
        chat.add_item(self._user("more"))
        chat.trim_if_needed()
        assert not any(getattr(e, "call_id", None) == "call_x" for e in chat.buffer)
        assert "call_x" in chat._pending_tool_calls

        chat.append_tool_output("call_x", self._fco("call_x"))
        assert chat._has_call_id_in_buffer("call_x")
        types = [e.type for e in chat.buffer]
        assert "function_call" in types
        assert "function_call_output" in types
        fc_idx = next(i for i, e in enumerate(chat.buffer) if e.type == "function_call")
        fco_idx = next(i for i, e in enumerate(chat.buffer) if e.type == "function_call_output")
        assert fc_idx < fco_idx

    def test_append_tool_output_rejects_unknown_call_id(self):
        from speech_to_speech.LLM.chat import ChatItemError

        chat = self._make_chat()
        with pytest.raises(ChatItemError, match="call_nope"):
            chat.append_tool_output("call_nope", self._fco("call_nope"))
        assert not any(getattr(e, "type", None) == "function_call_output" for e in chat.buffer)

    def test_copy_preserves_pending_tool_calls(self):
        chat = self._make_chat()
        chat.add_item(self._fc("call_a"))
        clone = chat.copy()
        assert "call_a" in clone._pending_tool_calls
        clone._pending_tool_calls.pop("call_a")
        assert "call_a" in chat._pending_tool_calls

    def test_reset_clears_pending_tool_calls(self):
        chat = self._make_chat()
        chat.add_item(self._fc())
        assert chat._pending_tool_calls
        chat.reset()
        assert chat._pending_tool_calls == {}
        assert chat.buffer == []

    # -- turn-based eviction --

    def test_eviction_removes_complete_turn(self):
        chat = self._make_chat(size=1)
        chat.add_item(self._user("turn 1"))
        chat.add_item(self._assistant("thinking"))
        chat.add_item(self._fc("c1"))
        chat.add_item(self._fco("c1"))
        chat.add_item(self._assistant("done"))
        assert len(chat.buffer) == 5

        chat.add_item(self._user("turn 2"))
        chat.trim_if_needed()
        from openai.types.realtime.realtime_conversation_item_user_message import (
            RealtimeConversationItemUserMessage,
        )

        user_msgs = [e for e in chat.buffer if isinstance(e, RealtimeConversationItemUserMessage)]
        assert len(user_msgs) == 1
        assert user_msgs[0].content[0].text == "turn 2"
        assert not any(getattr(e, "call_id", None) == "call_c1" and e.type == "function_call" for e in chat.buffer)

    def test_eviction_preserves_size_user_turns(self):
        from openai.types.realtime.realtime_conversation_item_user_message import (
            RealtimeConversationItemUserMessage,
        )

        chat = self._make_chat(size=2)
        chat.add_item(self._user("t1"))
        chat.add_item(self._assistant("r1"))
        chat.add_item(self._user("t2"))
        chat.add_item(self._assistant("let me check"))
        chat.add_item(self._fc("c2"))
        chat.add_item(self._fco("c2"))
        chat.add_item(self._assistant("here"))
        assert chat._user_turn_count == 2

        chat.add_item(self._user("t3"))
        chat.trim_if_needed()
        assert chat._user_turn_count == 2
        user_texts = [e.content[0].text for e in chat.buffer if isinstance(e, RealtimeConversationItemUserMessage)]
        assert user_texts == ["t2", "t3"]

    def test_pending_tool_calls_cleaned_after_reinjection(self):
        chat = self._make_chat(size=1)
        chat.add_item(self._user("hi"))
        chat.add_item(self._fc("call_z"))
        chat.add_item(self._user("bye"))
        assert "call_z" in chat._pending_tool_calls

        chat.append_tool_output("call_z", self._fco("call_z"))
        assert chat._has_call_id_in_buffer("call_z")
