"""Unit tests for api.openai_realtime.service.RealtimeService.

Every public method is exercised and the emitted OpenAI Realtime events are
validated for correct type, attributes, and state transitions.
"""

import base64
import json

import pytest

from openai.types.realtime import (
    ConversationItemCreateEvent,
    ConversationItemCreatedEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    RealtimeErrorEvent,
    RealtimeResponse,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCancelEvent,
    ResponseCreateEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    SessionCreatedEvent,
    SessionUpdateEvent,
)

from api.openai_realtime.service import (
    CHUNK_SIZE_BYTES,
    ConnState,
    RealtimeService,
)


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
        sid = service.register("c1")
        assert sid.startswith("session_")
        st = service._state("c1")
        assert st.conversation_id.startswith("conv_")
        assert st.in_response is False
        assert st.last_item_id is None
        service.unregister("c1")

    def test_unregister_removes_state(self, service):
        service.register("c2")
        service.unregister("c2")
        with pytest.raises(KeyError):
            service._state("c2")

    def test_build_session_created(self, service, conn_id, runtime_config):
        runtime_config.voice = "echo"
        runtime_config.instructions = "Be helpful"
        runtime_config.tools = [{"type": "function", "name": "get_weather"}]
        runtime_config.tool_choice = "auto"
        runtime_config.turn_detection = {"type": "server_vad"}
        runtime_config.input_audio_transcription = {"model": "whisper-1"}

        evt = service.build_session_created(conn_id)
        assert isinstance(evt, SessionCreatedEvent)
        assert evt.event_id.startswith("event_")
        assert evt.session is not None
        assert evt.session.instructions == "Be helpful"
        assert evt.session.tools is not None
        assert evt.session.tool_choice == "auto"
        assert evt.session.audio.output.voice == "echo"
        assert evt.session.audio.input.turn_detection.type == "server_vad"
        assert evt.session.audio.input.transcription.model == "whisper-1"


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
        return SessionUpdateEvent(type="session.update", session=session_fields)

    def test_session_update_voice(self, service, runtime_config):
        evt = self._make_update(
            audio={"output": {"voice": "shimmer"}},
        )
        service.handle_session_update(evt)
        assert runtime_config.voice == "shimmer"

    def test_session_update_instructions(self, service, runtime_config):
        service.handle_session_update(self._make_update(instructions="Be concise"))
        assert runtime_config.instructions == "Be concise"

    def test_session_update_tools_and_tool_choice(self, service, runtime_config):
        tools = [{"type": "function", "name": "f1"}]
        service.handle_session_update(self._make_update(tools=tools, tool_choice="required"))
        assert runtime_config.tools is not None
        assert runtime_config.tool_choice == "required"

    def test_session_update_nested_audio_format(self, service, runtime_config):
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
        service.handle_session_update(evt)
        assert runtime_config.voice == "nova"
        assert runtime_config.turn_detection.type == "server_vad"


# ===================================================================
# Conversation item create
# ===================================================================

class TestHandleConversationItemCreate:
    def _text_event(self, text: str = "hello", item_id: str = "item_abc") -> ConversationItemCreateEvent:
        return ConversationItemCreateEvent(
            type="conversation.item.create",
            item={
                "id": item_id,
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        )

    def test_text_input_emits_conversation_item_created(
        self, service, conn_id, text_prompt_queue, should_listen,
    ):
        should_listen.set()
        events = service.handle_conversation_item_create(conn_id, self._text_event("hi"))
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, ConversationItemCreatedEvent)
        assert evt.previous_item_id is None  # first item
        assert evt.item.role == "user"
        assert evt.item.status == "completed"
        assert evt.item.content[0].type == "input_text"
        assert evt.item.content[0].text == "hi"
        assert not text_prompt_queue.empty()
        assert text_prompt_queue.get() == "hi"
        assert not should_listen.is_set()

    def test_text_input_previous_item_id_chain(self, service, conn_id):
        e1 = service.handle_conversation_item_create(conn_id, self._text_event("a", "item_1"))
        e2 = service.handle_conversation_item_create(conn_id, self._text_event("b", "item_2"))
        assert e1[0].previous_item_id is None
        assert e2[0].previous_item_id == "item_1"

    def test_function_call_output_forwarded(self, service, conn_id, text_prompt_queue):
        evt = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={"type": "function_call_output", "output": '{"result": 42}', "call_id": "call_1"},
        )
        events = service.handle_conversation_item_create(conn_id, evt)
        assert events == []
        assert text_prompt_queue.get() == '{"result": 42}'

    def test_input_image_ignored(self, service, conn_id):
        evt = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "url": "https://example.com/img.png"}],
            },
        )
        events = service.handle_conversation_item_create(conn_id, evt)
        assert events == []


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
        err = service.handle_response_create(conn_id, evt)
        assert err is None

    def test_response_create_while_active(self, service, conn_id):
        service._state(conn_id).in_response = True
        evt = ResponseCreateEvent(type="response.create")
        err = service.handle_response_create(conn_id, evt)
        assert isinstance(err, RealtimeErrorEvent)
        assert err.error.type == "conversation_already_has_active_response"

    def test_response_create_stores_overrides(self, service, conn_id, runtime_config):
        evt = ResponseCreateEvent(
            type="response.create",
            response={
                "instructions": "override instructions",
                "tool_choice": {"type": "function", "name": "my_func"},
            },
        )
        service.handle_response_create(conn_id, evt)
        assert runtime_config.response_instructions == "override instructions"
        assert runtime_config.response_tool_choice is not None


# ===================================================================
# Response cancel
# ===================================================================

class TestHandleResponseCancel:
    def test_cancel_active_response(self, service, conn_id, should_listen):
        should_listen.clear()
        service._ensure_response(conn_id)
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
        service._state(conn_id).response_metadata = {"key": "value"}
        events = service.encode_audio_chunk(conn_id, _pcm_bytes(256))
        resp = events[0].response
        assert resp.metadata == {"key": "value"}


# ===================================================================
# Finish audio response
# ===================================================================

class TestFinishAudioResponse:
    def test_finish_emits_audio_done_and_response_done(self, service, conn_id):
        service._ensure_response(conn_id)
        events = service.finish_audio_response(conn_id)
        assert len(events) == 2
        assert isinstance(events[0], ResponseAudioDoneEvent)
        assert events[0].content_index == 0
        assert isinstance(events[1], ResponseDoneEvent)
        assert events[1].response.status == "completed"

    def test_finish_with_cancel_status(self, service, conn_id):
        service._ensure_response(conn_id)
        events = service.finish_audio_response(conn_id, status="cancelled", reason="turn_detected")
        done = events[1]
        assert done.response.status == "cancelled"
        assert done.response.status_details.reason == "turn_detected"

    def test_finish_resets_state(self, service, conn_id):
        service._state(conn_id).response_metadata = {"k": "v"}
        service._ensure_response(conn_id)
        service.finish_audio_response(conn_id)
        st = service._state(conn_id)
        assert st.in_response is False
        assert st.current_response_id is None
        assert st.current_output_item_id is None
        assert st.response_metadata is None


# ===================================================================
# Pipeline text translation
# ===================================================================

class TestTranslatePipelineText:

    # -- speech_started --

    def test_speech_started_emits_event(self, service, conn_id):
        events = service.translate_pipeline_text(
            conn_id, {"type": "speech_started", "audio_start_ms": 1000},
        )
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, InputAudioBufferSpeechStartedEvent)
        assert evt.audio_start_ms == 1000
        assert evt.item_id.startswith("item_")

    def test_speech_started_cancels_active_response(self, service, conn_id):
        service._ensure_response(conn_id)
        events = service.translate_pipeline_text(
            conn_id, {"type": "speech_started", "audio_start_ms": 0},
        )
        cancel_events = [e for e in events if isinstance(e, (ResponseAudioDoneEvent, ResponseDoneEvent))]
        assert len(cancel_events) == 2
        done = [e for e in cancel_events if isinstance(e, ResponseDoneEvent)][0]
        assert done.response.status == "cancelled"
        assert done.response.status_details.reason == "turn_detected"
        speech = [e for e in events if isinstance(e, InputAudioBufferSpeechStartedEvent)]
        assert len(speech) == 1

    # -- speech_stopped --

    def test_speech_stopped_emits_event(self, service, conn_id):
        service.translate_pipeline_text(conn_id, {"type": "speech_started", "audio_start_ms": 0})
        events = service.translate_pipeline_text(
            conn_id, {"type": "speech_stopped", "audio_end_ms": 2000},
        )
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, InputAudioBufferSpeechStoppedEvent)
        assert evt.audio_end_ms == 2000

    def test_speech_stopped_same_item_id_as_started(self, service, conn_id):
        started = service.translate_pipeline_text(
            conn_id, {"type": "speech_started", "audio_start_ms": 0},
        )
        stopped = service.translate_pipeline_text(
            conn_id, {"type": "speech_stopped", "audio_end_ms": 500},
        )
        assert started[0].item_id == stopped[0].item_id

    def test_speech_stopped_stores_duration(self, service, conn_id):
        service.translate_pipeline_text(conn_id, {"type": "speech_started"})
        service.translate_pipeline_text(
            conn_id, {"type": "speech_stopped", "duration_s": 2.5},
        )
        assert service._state(conn_id).input_audio_duration_s == 2.5

    # -- assistant_text --

    def test_assistant_text_emits_transcript_done(self, service, conn_id):
        events = service.translate_pipeline_text(
            conn_id, {"type": "assistant_text", "text": "Hello there"},
        )
        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, ResponseAudioTranscriptDoneEvent)
        assert evt.content_index == 0
        assert evt.output_index == 0
        assert evt.transcript == "Hello there"

    def test_assistant_text_with_tools(self, service, conn_id):
        events = service.translate_pipeline_text(
            conn_id,
            {
                "type": "assistant_text",
                "text": "Let me check",
                "tools": [
                    {"call_id": "c1", "name": "get_weather", "arguments": {"city": "Paris"}},
                    {"call_id": "c2", "name": "get_time", "arguments": {}},
                ],
            },
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
        events = service.translate_pipeline_text(
            conn_id,
            {
                "type": "assistant_text",
                "text": "",
                "tools": [{"call_id": "c1", "name": "f1", "arguments": {}}],
            },
        )
        assert len(events) == 1
        assert isinstance(events[0], ResponseFunctionCallArgumentsDoneEvent)
        assert events[0].output_index == 0

    # -- partial_transcription --

    def test_partial_transcription_emits_delta(self, service, conn_id):
        service.translate_pipeline_text(conn_id, {"type": "speech_started"})
        e1 = service.translate_pipeline_text(
            conn_id, {"type": "partial_transcription", "delta": "hel"},
        )
        e2 = service.translate_pipeline_text(
            conn_id, {"type": "partial_transcription", "delta": "lo"},
        )
        assert isinstance(e1[0], ConversationItemInputAudioTranscriptionDeltaEvent)
        assert e1[0].content_index == 0
        assert e1[0].delta == "hel"
        assert isinstance(e2[0], ConversationItemInputAudioTranscriptionDeltaEvent)
        assert e2[0].content_index == 1

    # -- transcription_completed --

    def test_transcription_completed_emits_event(self, service, conn_id):
        service.translate_pipeline_text(conn_id, {"type": "speech_started"})
        service.translate_pipeline_text(
            conn_id, {"type": "speech_stopped", "duration_s": 3.2},
        )
        events = service.translate_pipeline_text(
            conn_id, {"type": "transcription_completed", "transcript": "hello world"},
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
        events = service.translate_pipeline_text(conn_id, {"type": "something_else"})
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

        # 1) _start_input_item updates last_item_id
        input_id = service._start_input_item(conn_id)
        assert st.last_item_id == input_id

        # 2) _ensure_response updates last_item_id
        _, output_id = service._ensure_response(conn_id)
        assert st.last_item_id == output_id

        # 3) handle_conversation_item_create updates last_item_id
        service._end_response(conn_id)
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
        service._start_input_item(conn_id)
        assert service._next_input_content_index(conn_id) == 0
        assert service._next_input_content_index(conn_id) == 1

        service._start_input_item(conn_id)
        assert service._next_input_content_index(conn_id) == 0

        service._ensure_response(conn_id)
        assert service._next_output_content_index(conn_id) == 0
        assert service._next_output_content_index(conn_id) == 1

        service._end_response(conn_id)
        service._ensure_response(conn_id)
        assert service._next_output_content_index(conn_id) == 0
