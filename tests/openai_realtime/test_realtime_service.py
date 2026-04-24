"""Unit tests for api.openai_realtime.service.RealtimeService.

Every public method is exercised and the emitted OpenAI Realtime events are
validated for correct type, attributes, and state transitions.
"""

import base64
import json

import pytest
from openai.types.realtime import (
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    RealtimeErrorEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCancelEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    SessionCreatedEvent,
    SessionUpdateEvent,
)

from speech_to_speech.api.openai_realtime.service import (
    CHUNK_SIZE_BYTES,
)
from speech_to_speech.pipeline.events import (
    AssistantTextEvent,
    PartialTranscriptionEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TokenUsageEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.messages import GenerateResponseRequest

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
    def _text_event(self, text: str = "hello", item_id: str = "item_abc") -> ConversationItemCreateEvent:
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
        chat = service._state(conn_id).runtime_config.chat.to_list()
        assert chat[-1]["role"] == "user"
        assert chat[-1]["content"] == [{"type": "input_text", "text": "hi"}]

    def test_text_input_previous_item_id_chain(self, service, conn_id):
        e1 = service.handle_conversation_item_create(conn_id, self._text_event("a", "item_1"))
        e2 = service.handle_conversation_item_create(conn_id, self._text_event("b", "item_2"))
        assert e1[0].previous_item_id is None
        assert e2[0].previous_item_id == "item_1"

    def test_function_call_output_forwarded(self, service, conn_id, text_prompt_queue):
        service._state(conn_id).runtime_config.chat.append(
            {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{}"}
        )
        evt = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={"type": "function_call_output", "output": '{"result": 42}', "call_id": "call_1"},
        )
        events = service.handle_conversation_item_create(conn_id, evt)
        assert len(events) == 1
        assert isinstance(events[0], ConversationItemCreatedEvent)
        chat = service._state(conn_id).runtime_config.chat.to_list()
        assert chat[-1] == {"type": "function_call_output", "call_id": "call_1", "output": '{"result": 42}'}

    def test_function_call_output_rejected_for_unknown_call_id(self, service, conn_id, text_prompt_queue):
        evt = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={"type": "function_call_output", "output": '{"result": 42}', "call_id": "call_unknown"},
        )
        events = service.handle_conversation_item_create(conn_id, evt)
        assert len(events) == 1
        assert isinstance(events[0], RealtimeErrorEvent)
        assert "call_unknown" in events[0].error.message
        chat = service._state(conn_id).runtime_config.chat.to_list()
        assert not any(entry.get("type") == "function_call_output" for entry in chat)

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
        chat = service._state(conn_id).runtime_config.chat.to_list()
        last = chat[-1]
        assert last["role"] == "user"
        assert isinstance(last["content"], list)
        assert last["content"][0]["type"] == "input_image"
        assert last["content"][0]["image_url"] == "https://example.com/img.png"

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
        chat = service._state(conn_id).runtime_config.chat.to_list()
        last = chat[-1]
        assert last["role"] == "user"
        assert isinstance(last["content"], list)
        assert len(last["content"]) == 2
        assert last["content"][0] == {"type": "input_text", "text": "What is this?"}
        assert last["content"][1]["type"] == "input_image"
        assert last["content"][1]["image_url"] == "data:image/png;base64,abc123"


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

    def test_double_response_create_rejected(self, service, conn_id, text_prompt_queue):
        """Second response.create is rejected because in_response is set immediately."""
        evt = ResponseCreateEvent(type="response.create")
        result1 = service.handle_response_create(conn_id, evt)
        assert isinstance(result1, ResponseCreatedEvent)
        result2 = service.handle_response_create(conn_id, evt)
        assert isinstance(result2, RealtimeErrorEvent)
        assert result2.error.type == "conversation_already_has_active_response"


# ===================================================================
# Response cancel
# ===================================================================


class TestHandleResponseCancel:
    def test_cancel_active_response(self, service, conn_id, should_listen):
        should_listen.clear()
        service.response._ensure_response(conn_id)
        events = service.handle_response_cancel(conn_id)
        assert len(events) == 2
        assert isinstance(events[0], ResponseAudioDoneEvent)
        assert isinstance(events[1], ResponseDoneEvent)
        assert events[1].response.status == "cancelled"
        assert events[1].response.status_details.reason == "client_cancelled"
        assert should_listen.is_set()

    def test_cancel_no_active_response(self, service, conn_id):
        events = service.handle_response_cancel(conn_id)
        assert events == []


# ===================================================================
# Outbound audio encoding
# ===================================================================


class TestEncodeAudioChunk:
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

    def test_subsequent_chunks_increment_content_index(self, service, conn_id):
        service.encode_audio_chunk(conn_id, _pcm_bytes(256))  # first
        events = service.encode_audio_chunk(conn_id, _pcm_bytes(256))  # second
        assert len(events) == 1
        assert isinstance(events[0], ResponseAudioDeltaEvent)
        assert events[0].content_index == 1

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
    def test_finish_emits_audio_done_and_response_done(self, service, conn_id):
        service.response._ensure_response(conn_id)
        events = service.finish_audio_response(conn_id)
        assert len(events) == 2
        assert isinstance(events[0], ResponseAudioDoneEvent)
        assert events[0].content_index == 0
        assert isinstance(events[1], ResponseDoneEvent)
        assert events[1].response.status == "completed"

    def test_finish_with_cancel_status(self, service, conn_id):
        service.response._ensure_response(conn_id)
        events = service.finish_audio_response(conn_id, status="cancelled", reason="turn_detected")
        done = events[1]
        assert done.response.status == "cancelled"
        assert done.response.status_details.reason == "turn_detected"

    def test_finish_resets_state(self, service, conn_id):
        from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

        service._state(conn_id).current_response_params = RealtimeResponseCreateParams(
            metadata={"k": "v"},
        )
        service.response._ensure_response(conn_id)
        service.finish_audio_response(conn_id)
        st = service._state(conn_id)
        assert st.in_response is False
        assert st.current_response_id is None
        assert st.current_item_id is None
        assert st.current_response_params is None


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
        assert len(cancel_events) == 2
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
        service.response._ensure_response(conn_id)
        events = service.dispatch_pipeline_event(
            conn_id,
            SpeechStartedEvent(),
        )
        assert len(events) == 1
        assert isinstance(events[0], InputAudioBufferSpeechStartedEvent)
        assert service._state(conn_id).in_response is True

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

    def test_assistant_text_emits_transcript_done(self, service, conn_id):
        events = service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(text="Hello there"),
        )
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, ResponseAudioTranscriptDoneEvent)
        assert evt.content_index == 0
        assert evt.output_index == 0
        assert evt.transcript == "Hello there"

    def test_assistant_text_with_tools(self, service, conn_id):
        events = service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(
                text="Let me check",
                tools=[
                    {"call_id": "c1", "name": "get_weather", "arguments": '{"city": "Paris"}'},
                    {"call_id": "c2", "name": "get_time", "arguments": "{}"},
                ],
            ),
        )
        assert len(events) == 3
        assert isinstance(events[0], ResponseAudioTranscriptDoneEvent)
        assert events[0].output_index == 0
        assert isinstance(events[1], ResponseFunctionCallArgumentsDoneEvent)
        assert events[1].output_index == 1
        assert events[1].name == "get_weather"
        assert events[1].call_id == "c1"
        assert json.loads(events[1].arguments) == {"city": "Paris"}
        assert isinstance(events[2], ResponseFunctionCallArgumentsDoneEvent)
        assert events[2].output_index == 2

    def test_assistant_text_tools_only(self, service, conn_id):
        events = service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(
                text="",
                tools=[{"call_id": "c1", "name": "f1", "arguments": "{}"}],
            ),
        )
        assert len(events) == 1
        assert isinstance(events[0], ResponseFunctionCallArgumentsDoneEvent)
        assert events[0].output_index == 0

    # -- partial_transcription --

    def test_partial_transcription_emits_delta(self, service, conn_id):
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        e1 = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="hel"),
        )
        e2 = service.dispatch_pipeline_event(
            conn_id,
            PartialTranscriptionEvent(delta="lo"),
        )
        assert isinstance(e1[0], ConversationItemInputAudioTranscriptionDeltaEvent)
        assert e1[0].content_index == 0
        assert e1[0].delta == "hel"
        assert isinstance(e2[0], ConversationItemInputAudioTranscriptionDeltaEvent)
        assert e2[0].content_index == 1

    # -- transcription_completed --

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
        events = service.dispatch_pipeline_event(conn_id, AssistantTextEvent(text="hi"))
        output_id = st.current_item_id
        assert st.last_item_id == output_id

        # 3) handle_conversation_item_create updates last_item_id
        service.response._end_response(conn_id)
        evt = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={
                "id": "item_manual",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "x"}],
            },
        )
        events = service.handle_conversation_item_create(conn_id, evt)
        assert st.last_item_id == "item_manual"
        assert events[0].previous_item_id == output_id

    def test_content_index_resets_on_new_item(self, service, conn_id):
        service.response._start_item(conn_id)
        assert service.response._next_content_index(conn_id) == 0
        assert service.response._next_content_index(conn_id) == 1

        service.response._start_item(conn_id)
        assert service.response._next_content_index(conn_id) == 0

        service.response._ensure_response(conn_id)
        assert service.response._next_content_index(conn_id) == 0
        assert service.response._next_content_index(conn_id) == 1

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

    def test_response_done_reflects_token_usage(self, service, conn_id):
        service.response._ensure_response(conn_id)
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=100, output_tokens=50),
        )
        events = service.finish_audio_response(conn_id)
        done_evt = events[1]
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

    def test_finish_audio_response_resets_per_response_tokens(self, service, conn_id):
        """After finish_audio_response, per-response counters are zero."""
        service.response._ensure_response(conn_id)
        service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(input_tokens=50, output_tokens=25),
        )
        service.finish_audio_response(conn_id)
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
        service.finish_audio_response(conn_id)
        assert service.total_usage.responses_completed == 1
        assert service.total_usage.responses_cancelled == 0

    def test_responses_cancelled_increments(self, service, conn_id):
        service.response._ensure_response(conn_id)
        service.finish_audio_response(conn_id, status="cancelled", reason="turn_detected")
        assert service.total_usage.responses_cancelled == 1
        assert service.total_usage.responses_completed == 0

    def test_multiple_responses_accumulate_status_counters(self, service, conn_id):
        service.response._ensure_response(conn_id)
        service.finish_audio_response(conn_id)
        service.response._ensure_response(conn_id)
        service.finish_audio_response(conn_id, status="cancelled", reason="client_cancelled")
        service.response._ensure_response(conn_id)
        service.finish_audio_response(conn_id)
        assert service.total_usage.responses_completed == 2
        assert service.total_usage.responses_cancelled == 1

    # -- tool_calls --

    def test_tool_calls_increments(self, service, conn_id):
        service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(
                text="",
                tools=[
                    {"call_id": "c1", "name": "f1", "arguments": "{}"},
                    {"call_id": "c2", "name": "f2", "arguments": "{}"},
                ],
            ),
        )
        assert service._state(conn_id).response_usage.tool_calls == 2

    def test_tool_calls_rolls_into_global(self, service, conn_id):
        service.response._ensure_response(conn_id)
        service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(
                text="",
                tools=[{"call_id": "c1", "name": "f1", "arguments": "{}"}],
            ),
        )
        service.finish_audio_response(conn_id)
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
            AssistantTextEvent(
                text="hi",
                tools=[{"call_id": "c1", "name": "f1", "arguments": "{}"}],
            ),
        )
        service.finish_audio_response(conn_id)
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

    def test_strip_images_removes_image_parts(self):
        chat = self._make_chat()
        chat.append(
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is this?"},
                    {"type": "input_image", "image_url": "data:image/png;base64,abc"},
                ],
            }
        )
        chat.append({"role": "assistant", "content": "It's a cat."})
        chat.strip_images()
        user_msg = chat.buffer[0]
        assert user_msg["content"] == [{"type": "input_text", "text": "What is this?"}]

    def test_strip_images_noop_on_text_only(self):
        chat = self._make_chat()
        chat.append({"role": "user", "content": "hello"})
        chat.append({"role": "assistant", "content": "hi"})
        chat.strip_images()
        assert chat.buffer[0]["content"] == "hello"
        assert chat.buffer[1]["content"] == "hi"

    def test_strip_then_new_image_cycle(self):
        chat = self._make_chat()
        chat.append(
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "look"},
                    {"type": "input_image", "image_url": "old_url"},
                ],
            }
        )
        chat.append({"role": "assistant", "content": "I see it."})
        chat.strip_images()
        assert chat.buffer[0]["content"] == [{"type": "input_text", "text": "look"}]

        chat.append(
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "now this"},
                    {"type": "input_image", "image_url": "new_url"},
                ],
            }
        )
        last_user = chat.buffer[-1]
        assert isinstance(last_user["content"], list)
        assert any(p.get("image_url") == "new_url" for p in last_user["content"])


# ===================================================================
# Chat tool call tracking
# ===================================================================


class TestChatToolCallTracking:
    """Tests for Chat._pending_tool_calls and append_tool_output."""

    def _make_chat(self, size=10):
        from speech_to_speech.LLM.chat import Chat

        return Chat(size=size)

    def _fc(self, call_id="call_1", name="f1"):
        return {"type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"}

    def _fco(self, call_id="call_1"):
        return {"type": "function_call_output", "call_id": call_id, "output": '{"ok": true}'}

    def test_append_registers_pending_tool_call(self):
        chat = self._make_chat()
        fc = self._fc()
        chat.append(fc)
        assert "call_1" in chat._pending_tool_calls
        assert chat._pending_tool_calls["call_1"] is fc

    def test_append_tool_output_clears_pending(self):
        chat = self._make_chat()
        chat.append(self._fc())
        assert "call_1" in chat._pending_tool_calls
        err = chat.append_tool_output("call_1", self._fco())
        assert err is None
        assert "call_1" not in chat._pending_tool_calls
        assert chat.buffer[-1]["type"] == "function_call_output"

    def test_append_tool_output_reinjects_evicted_call(self):
        chat = self._make_chat(size=1)
        # size=1 keeps 1 user turn. The 2nd user message evicts the 1st turn
        # (user + fc + assistant), but _pending_tool_calls retains the fc.
        chat.append({"role": "user", "content": "hi"})
        chat.append(self._fc("call_x"))
        chat.append({"role": "assistant", "content": "ok"})
        chat.append({"role": "user", "content": "more"})
        assert not any(e.get("call_id") == "call_x" for e in chat.buffer)
        assert "call_x" in chat._pending_tool_calls

        err = chat.append_tool_output("call_x", self._fco("call_x"))
        assert err is None
        assert "call_x" not in chat._pending_tool_calls
        types = [e.get("type") for e in chat.buffer]
        assert "function_call" in types
        assert "function_call_output" in types
        fc_idx = next(i for i, e in enumerate(chat.buffer) if e.get("type") == "function_call")
        fco_idx = next(i for i, e in enumerate(chat.buffer) if e.get("type") == "function_call_output")
        assert fc_idx < fco_idx

    def test_append_tool_output_rejects_unknown_call_id(self):
        chat = self._make_chat()
        err = chat.append_tool_output("call_nope", self._fco("call_nope"))
        assert err is not None
        assert "call_nope" in err
        assert not any(e.get("type") == "function_call_output" for e in chat.buffer)

    def test_copy_preserves_pending_tool_calls(self):
        chat = self._make_chat()
        chat.append(self._fc("call_a"))
        clone = chat.copy()
        assert "call_a" in clone._pending_tool_calls
        # Mutating the clone does not affect the original
        clone._pending_tool_calls.pop("call_a")
        assert "call_a" in chat._pending_tool_calls

    def test_reset_clears_pending_tool_calls(self):
        chat = self._make_chat()
        chat.append(self._fc())
        assert chat._pending_tool_calls
        chat.reset()
        assert chat._pending_tool_calls == {}
        assert chat.buffer == []

    # -- transformers / Chat Completions tool_calls format --

    def _assistant_with_tool_calls(self, *call_ids):
        """Build an assistant message with tool_calls (local LLM format)."""
        return {
            "role": "assistant",
            "tool_calls": [
                {"type": "function", "id": cid, "function": {"name": f"fn_{cid}", "arguments": {}}} for cid in call_ids
            ],
        }

    def test_append_registers_pending_from_tool_calls_format(self):
        chat = self._make_chat()
        msg = self._assistant_with_tool_calls("tc_1", "tc_2")
        chat.append(msg)
        assert "tc_1" in chat._pending_tool_calls
        assert "tc_2" in chat._pending_tool_calls

    def test_append_tool_output_accepts_tool_calls_format_in_buffer(self):
        chat = self._make_chat()
        chat.append(self._assistant_with_tool_calls("tc_1"))
        err = chat.append_tool_output("tc_1", self._fco("tc_1"))
        assert err is None
        assert chat.buffer[-1] == self._fco("tc_1")
        assert "tc_1" not in chat._pending_tool_calls

    def test_append_tool_output_reinjects_evicted_tool_calls_format(self):
        chat = self._make_chat(size=1)
        chat.append({"role": "user", "content": "hi"})
        chat.append(self._assistant_with_tool_calls("tc_x"))
        chat.append({"role": "user", "content": "a"})
        assert not any(e.get("tool_calls") for e in chat.buffer)
        assert "tc_x" in chat._pending_tool_calls

        err = chat.append_tool_output("tc_x", self._fco("tc_x"))
        assert err is None
        assert "tc_x" not in chat._pending_tool_calls
        assert any(
            e.get("role") == "assistant" and any(tc.get("id") == "tc_x" for tc in e.get("tool_calls", []))
            for e in chat.buffer
        )
        assert chat.buffer[-1] == self._fco("tc_x")

    # -- turn-based eviction --

    def test_eviction_removes_complete_turn(self):
        chat = self._make_chat(size=1)
        chat.append({"role": "user", "content": "turn 1"})
        chat.append({"role": "assistant", "content": "thinking"})
        chat.append(self._fc("c1"))
        chat.append(self._fco("c1"))
        chat.append({"role": "assistant", "content": "done"})
        # Still 1 user turn, no eviction yet
        assert len(chat.buffer) == 5

        chat.append({"role": "user", "content": "turn 2"})
        # 2 user turns > size=1 → oldest turn fully evicted
        user_msgs = [e for e in chat.buffer if e.get("role") == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "turn 2"
        assert not any(e.get("content") == "thinking" for e in chat.buffer)
        assert not any(e.get("call_id") == "c1" and e.get("type") == "function_call" for e in chat.buffer)

    def test_eviction_preserves_size_user_turns(self):
        chat = self._make_chat(size=2)
        # Turn 1: short
        chat.append({"role": "user", "content": "t1"})
        chat.append({"role": "assistant", "content": "r1"})
        # Turn 2: long (with tool call)
        chat.append({"role": "user", "content": "t2"})
        chat.append({"role": "assistant", "content": "let me check"})
        chat.append(self._fc("c2"))
        chat.append(self._fco("c2"))
        chat.append({"role": "assistant", "content": "here"})
        assert chat._count_user_turns() == 2

        # Turn 3: triggers eviction of turn 1
        chat.append({"role": "user", "content": "t3"})
        assert chat._count_user_turns() == 2
        user_contents = [e["content"] for e in chat.buffer if e.get("role") == "user"]
        assert user_contents == ["t2", "t3"]

    def test_pending_tool_calls_cleaned_after_reinjection(self):
        chat = self._make_chat(size=1)
        chat.append({"role": "user", "content": "hi"})
        chat.append(self._fc("call_z"))
        chat.append({"role": "user", "content": "bye"})
        assert "call_z" in chat._pending_tool_calls

        chat.append_tool_output("call_z", self._fco("call_z"))
        assert "call_z" not in chat._pending_tool_calls
