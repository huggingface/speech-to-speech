import base64
import json
import logging
import uuid
import numpy as np
from pydantic import BaseModel, Field
from queue import Queue
from scipy.signal import resample_poly
from threading import Event as ThreadingEvent
from typing import Literal, Optional, Union

from pydantic import ValidationError

from openai.types.realtime.realtime_response import Audio, AudioOutput
from openai.types.realtime.realtime_conversation_item_user_message import (
    RealtimeConversationItemUserMessage,
    Content as UserMessageContent,
)
from openai.types.realtime.realtime_response_status import RealtimeResponseStatus
from openai.types.realtime.realtime_response_usage import RealtimeResponseUsage
from openai.types.realtime.conversation_item_input_audio_transcription_completed_event import (
    UsageTranscriptTextUsageDuration,
)
from openai.types.realtime.realtime_transcription_session_create_request import (
    RealtimeTranscriptionSessionCreateRequest,
)

from openai.types.realtime import (
    InputAudioBufferAppendEvent,
    SessionUpdateEvent,
    ConversationItemCreateEvent,
    ConversationItemCreatedEvent,
    ResponseCreateEvent,
    ResponseCancelEvent,
    SessionCreatedEvent,
    RealtimeSessionCreateRequest,
    RealtimeErrorEvent,
    RealtimeError,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    ResponseCreatedEvent,
    RealtimeResponse,
    ResponseDoneEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent
)

from api.openai_realtime.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512
BYTES_PER_SAMPLE = 2
CHUNK_SIZE_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE

_ResponseStatus = Literal["completed", "cancelled", "failed", "incomplete", "in_progress"]
_StatusReason = Literal["turn_detected", "client_cancelled", "max_output_tokens", "content_filter"]


def _resample(audio_int16: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample int16 PCM audio between sample rates using polyphase filtering."""
    if from_rate == to_rate:
        return audio_int16
    samples = np.frombuffer(audio_int16, dtype=np.int16).astype(np.float32) / 32768.0
    gcd = np.gcd(to_rate, from_rate)
    resampled = resample_poly(samples, up=to_rate // gcd, down=from_rate // gcd)
    return np.clip(resampled * 32768, -32768, 32767).astype(np.int16).tobytes()

_EVENT_TYPE_TO_MODEL: dict[str, type[BaseModel]] = {
    "input_audio_buffer.append": InputAudioBufferAppendEvent,
    "session.update": SessionUpdateEvent,
    "conversation.item.create": ConversationItemCreateEvent,
    "response.create": ResponseCreateEvent,
    "response.cancel": ResponseCancelEvent,
}

ClientEvent = Union[
    InputAudioBufferAppendEvent,
    SessionUpdateEvent,
    ConversationItemCreateEvent,
    ResponseCreateEvent,
    ResponseCancelEvent,
]

ServerEvent = Union[
    SessionCreatedEvent,
    RealtimeErrorEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    ConversationItemCreatedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent
]

RealtimeEvent = Union[ClientEvent, ServerEvent]

def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"



class ConnState(BaseModel):
    """Per-connection mutable state, including all protocol-level IDs."""
    session_id: str = Field(default_factory=lambda: _generate_id("session"))
    conversation_id: str = Field(default_factory=lambda: _generate_id("conv"))
    in_response: bool = False
    audio_buffer_has_data: bool = False
    audio_remainder: bytes = b""
    current_response_id: str | None = None
    current_item_id: str | None = None
    content_index: int = 0
    input_audio_duration_s: float = 0.0
    last_item_id: str | None = None
    response_metadata: dict[str, str] | None = None


class RealtimeService:
    """Translates between OpenAI Realtime protocol events and internal pipeline messages.

    One instance is shared across all WebSocket connections.  Per-connection
    state (response lifecycle, audio buffer) is tracked internally by
    connection id.
    """

    def __init__(
        self,
        runtime_config: RuntimeConfig | None = None,
        text_prompt_queue: Queue | None = None,
        should_listen: ThreadingEvent | None = None,
    ):
        self.runtime_config = runtime_config or RuntimeConfig()
        self.text_prompt_queue = text_prompt_queue
        self.should_listen = should_listen
        self._conns: dict[str, ConnState] = {}

    # ── Connection lifecycle ─────────────────────

    def register(self) -> str:
        """Register a new connection and return its session_id."""
        state = ConnState()
        self._conns[state.session_id] = state
        return state.session_id

    def unregister(self, conn_id: str) -> None:
        self._conns.pop(conn_id, None)

    def _state(self, conn_id: str) -> ConnState:
        return self._conns[conn_id]

    @property
    def connection_ids(self) -> list[str]:
        return list(self._conns)

    def build_session_created(self, conn_id: str) -> SessionCreatedEvent:
        """Build a SessionCreatedEvent populated with the current RuntimeConfig."""
        session = self.runtime_config.session or RealtimeSessionCreateRequest(type="realtime")
        return SessionCreatedEvent(
            type="session.created",
            event_id=self._next_event_id(),
            session=session,
        )

    # ── ID management ────────────────────────────

    @staticmethod
    def _next_event_id() -> str:
        return _generate_id("event")

    def _ensure_response(self, conn_id: str) -> tuple[str, str]:
        """Ensure a response and output item exist, creating them if needed."""
        st = self._state(conn_id)
        if st.current_response_id is None:
            st.current_response_id = _generate_id("resp")
            self._start_item(conn_id)
            st.in_response = True
        return st.current_response_id, self._current_item_id(conn_id)

    def _end_response(self, conn_id: str) -> None:
        st = self._state(conn_id)
        st.current_response_id = None
        st.current_item_id = None
        st.content_index = 0
        st.in_response = False
        st.response_metadata = None

    def _start_item(self, conn_id: str) -> str:
        """Generate a new item ID, reset content index, and store it."""
        st = self._state(conn_id)
        item_id = _generate_id("item")
        st.current_item_id = item_id
        st.content_index = 0
        st.input_audio_duration_s = 0.0
        return item_id

    def _current_item_id(self, conn_id: str) -> str:
        return self._state(conn_id).current_item_id or self._start_item(conn_id)

    def _next_content_index(self, conn_id: str) -> int:
        """Return the current content index and advance it."""
        st = self._state(conn_id)
        idx = st.content_index
        st.content_index += 1
        return idx

    def _build_response(self, conn_id: str, status: _ResponseStatus, reason: _StatusReason | None = None) -> RealtimeResponse:
        """Build a fully-populated RealtimeResponse from the current connection state."""
        st = self._state(conn_id)
        status_details = None
        if reason or status in ("completed", "cancelled", "incomplete", "failed"):
            status_details = RealtimeResponseStatus(type=status, reason=reason)  # type: ignore[arg-type]
        audio_cfg = self.runtime_config.session.audio
        audio_output = audio_cfg.output if audio_cfg is not None else None
        voice = audio_output.voice if audio_output is not None else None
        return RealtimeResponse(
            id=st.current_response_id,
            object="realtime.response",
            status=status,
            status_details=status_details,
            audio=Audio(output=AudioOutput(voice=voice)),
            conversation_id=st.conversation_id,
            metadata=st.response_metadata,
            usage=RealtimeResponseUsage(
                input_tokens=0, output_tokens=0, total_tokens=0,
            ),
        )

    # ── Client event parsing ─────────────────────

    def parse_client_event(self, raw: dict) -> Optional[ClientEvent]:
        event_type: str | None = raw.get("type")
        if event_type is None:
            logger.warning("Client event missing 'type' field")
            return None
        model_cls = _EVENT_TYPE_TO_MODEL.get(event_type)
        if model_cls is None:
            logger.warning(f"Unknown client event type: {event_type}")
            return None
        try:
            return model_cls.model_validate(raw)  # type: ignore[return-value]
        except ValidationError as e:
            logger.error(f"Invalid {event_type} payload: {e}")
            return None

    # ── Client event handlers ────────────────────

    def handle_audio_append(self, conn_id: str, event: InputAudioBufferAppendEvent) -> list[bytes]:
        """Decode base64 audio, resample to pipeline rate, and split into 512-sample PCM16 chunks for the VAD."""
        try:
            pcm_bytes = base64.b64decode(event.audio)
        except Exception as e:
            logger.error(f"Base64 decode error: {e}")
            return []

        audio_cfg = self.runtime_config.session.audio
        if audio_cfg is not None and audio_cfg.input is not None:
            client_in_rate = getattr(audio_cfg.input.format, "rate", None) or PIPELINE_SAMPLE_RATE
        else:
            client_in_rate = PIPELINE_SAMPLE_RATE
        pcm_bytes = _resample(pcm_bytes, client_in_rate, PIPELINE_SAMPLE_RATE)

        st = self._state(conn_id)
        pcm_bytes = st.audio_remainder + pcm_bytes

        chunks = []
        for i in range(0, len(pcm_bytes), CHUNK_SIZE_BYTES):
            chunk = pcm_bytes[i : i + CHUNK_SIZE_BYTES]
            if len(chunk) == CHUNK_SIZE_BYTES:
                chunks.append(chunk)
            else:
                st.audio_remainder = chunk
                break
        else:
            st.audio_remainder = b""

        if chunks:
            st.audio_buffer_has_data = True
        return chunks

    def handle_session_update(self, event: SessionUpdateEvent) -> Optional[RealtimeErrorEvent]:
        """Apply session config changes to the shared RuntimeConfig.

        Only ``RealtimeSessionCreateRequest`` sessions are accepted;
        ``RealtimeTranscriptionSessionCreateRequest`` sessions not yet supported.
        Incoming fields are deep-merged into the existing session so that
        partial updates preserve previously-set values.
        """
        s = event.session
        if s is None:
            return None

        if isinstance(s, RealtimeTranscriptionSessionCreateRequest):
            return self.make_error(
                message="Only 'realtime' session type is supported; transcription sessions are not.",
                _type="invalid_session_type",
            )

        model = getattr(s, "model", None)
        if model is not None:
            logger.info(f"Session model set to: {model}")

        current = self.runtime_config.session
        if current is None:
            self.runtime_config.session = s
        else:
            self.runtime_config.apply_session_update(s)
        logger.info("Session configuration updated")
        return None

    def handle_conversation_item_create(self, conn_id: str, event: ConversationItemCreateEvent) -> list[ServerEvent]:
        """Inject a text message or function-call output into the LLM context.

        Items are added to the LLM chat context but do NOT trigger response
        generation on their own.  A subsequent ``response.create`` event is
        required to trigger the model.

        Supported content part types:
        - ``input_text``: added to LLM context (generation deferred to response.create).
        - ``input_image``: acknowledged but not processed (image support is
          not implemented in the local pipeline). A log entry is emitted so
          callers can observe the event.
        - ``function_call_output``: added to LLM context without triggering generation.
        """
        events: list[ServerEvent] = []
        item = event.item

        if item.type == "message" and item.content:
            for part in item.content:
                if part.type == "input_text" and part.text:
                    if self.text_prompt_queue:
                        self.text_prompt_queue.put(("__ADD_TO_CONTEXT__", "user", part.text))
                        logger.info(f"Injected text to LLM: {part.text!r:.80}")
                    if self.should_listen:
                        self.should_listen.clear()
                    st = self._state(conn_id)
                    events.append(
                        ConversationItemCreatedEvent(
                            type="conversation.item.created",
                            event_id=self._next_event_id(),
                            previous_item_id=st.last_item_id,
                            item=RealtimeConversationItemUserMessage(
                                id=item.id,
                                type="message",
                                role="user",
                                object="realtime.item",
                                status="completed",
                                content=[UserMessageContent(type="input_text", text=part.text)],
                            ),
                        )
                    )
                    st.last_item_id = item.id
                elif part.type == "input_image":
                    # Image content is received but not forwarded to the
                    # local pipeline — vision is not supported server-side.
                    logger.info(
                        "Received input_image content part; "
                        "ignoring — image processing not yet supported in local pipeline",
                    )

        elif item.type == "function_call_output" and item.output:
            if self.text_prompt_queue:
                self.text_prompt_queue.put(("__FUNCTION_RESULT__", item.output))
                logger.info("Injected function_call_output to LLM context")

        return events

    def handle_audio_commit(self, conn_id: str) -> RealtimeErrorEvent | None:
        """Commit the audio buffer. Returns an error if no audio was appended."""
        st = self._state(conn_id)
        if not st.audio_buffer_has_data:
            return self.make_error(
                message="Input audio buffer is empty, nothing to commit.",
                _type="input_audio_buffer_commit_empty",
            )
        st.audio_buffer_has_data = False
        logger.debug("Audio buffer committed")
        return None

    def handle_response_create(self, conn_id: str, event: ResponseCreateEvent) -> RealtimeErrorEvent | None:
        """Trigger a response. Returns an error if a response is already in progress.

        Per-response overrides (instructions, tool_choice) travel inside the
        ``__GENERATE_RESPONSE__`` sentinel tuple so they are atomically paired
        with the correct LLM generation — no shared mutable state to race on.
        """
        if self._state(conn_id).in_response:
            return self.make_error(
                message="Cannot create response while another response is in progress.",
                _type="conversation_already_has_active_response",
            )
        override_instructions: str | None = None
        override_tool_choice: str | None = None
        if event.response:
            cfg = self.runtime_config
            if cfg.session is None:
                cfg.session = RealtimeSessionCreateRequest(type="realtime")
            if event.response.instructions:
                override_instructions = event.response.instructions
                logger.info(
                    "Per-response instructions (%d chars): %s...",
                    len(override_instructions),
                    override_instructions[:60],
                )
            if event.response.tool_choice:
                if not isinstance(event.response.tool_choice, str):
                    return self.make_error(
                        message="Only string tool_choice values are supported for now (auto, required, none).",
                        _type="tool_choice_not_supported",
                    )
                override_tool_choice = event.response.tool_choice
                logger.info("Per-response tool_choice: %s", override_tool_choice)
            if hasattr(event.response, "metadata") and event.response.metadata:
                self._state(conn_id).response_metadata = event.response.metadata
                logger.info("Per-response metadata stored (%d keys)", len(event.response.metadata))
        if self.text_prompt_queue:
            self.text_prompt_queue.put(
                ("__GENERATE_RESPONSE__", override_instructions, override_tool_choice)
            )
        logger.debug("response.create received, LLM generation triggered")
        return None

    def handle_response_cancel(self, conn_id: str) -> list[ServerEvent]:
        """Cancel the in-progress response and re-enable listening."""
        events = self.finish_audio_response(conn_id, status="cancelled", reason="client_cancelled")
        if self.should_listen:
            self.should_listen.set()
        logger.info("Response cancelled, listening re-enabled")
        return events

    # ── Outbound audio encoding ──────────────────

    def encode_audio_chunk(self, conn_id: str, audio: bytes) -> list[ServerEvent]:
        """Encode a raw PCM audio chunk, emitting ResponseCreated on the first chunk."""
        events: list[ServerEvent] = []
        was_new = not self._state(conn_id).in_response
        resp_id, item_id = self._ensure_response(conn_id)
        if was_new:
            events.append(ResponseCreatedEvent(
                type="response.created",
                event_id=self._next_event_id(),
                response=self._build_response(conn_id, "in_progress"),
            ))
        audio_cfg = self.runtime_config.session.audio
        if audio_cfg is not None and audio_cfg.output is not None:
            client_out_rate = getattr(audio_cfg.output.format, "rate", None) or PIPELINE_SAMPLE_RATE
        else:
            client_out_rate = PIPELINE_SAMPLE_RATE
        audio = _resample(audio, PIPELINE_SAMPLE_RATE, client_out_rate)
        b64 = base64.b64encode(audio).decode("ascii")
        events.append(ResponseAudioDeltaEvent(
            type="response.output_audio.delta",
            event_id=self._next_event_id(),
            content_index=self._next_content_index(conn_id),
            delta=b64,
            item_id=item_id,
            output_index=0,
            response_id=resp_id,
        ))
        return events

    def finish_audio_response(
        self, conn_id: str, status: _ResponseStatus = "completed", reason: _StatusReason | None = None,
    ) -> list[ServerEvent]:
        """Close the current response (audio done + response done)."""
        st = self._state(conn_id)
        events: list[ServerEvent] = []
        if st.in_response:
            resp_id, item_id = self._ensure_response(conn_id)
            events.append(ResponseAudioDoneEvent(
                type="response.output_audio.done",
                event_id=self._next_event_id(),
                content_index=0,
                item_id=item_id,
                output_index=0,
                response_id=resp_id,
            ))
            events.append(ResponseDoneEvent(
                type="response.done",
                event_id=self._next_event_id(),
                response=self._build_response(conn_id, status, reason),
            ))
            self._end_response(conn_id)
        return events

    # ── Pipeline text -> protocol events ─────────

    def translate_pipeline_text(self, conn_id: str, msg: dict) -> list[ServerEvent]:
        """Convert an internal text_output_queue message to protocol ServerEvent(s)."""
        events: list[ServerEvent] = []
        msg_type = msg.get("type")

        if msg_type == "speech_started":
            if self._state(conn_id).in_response and self.runtime_config.interrupt_response_enabled:
                events.extend(self.finish_audio_response(conn_id, status="cancelled", reason="turn_detected"))
            input_item_id = self._start_item(conn_id)
            self._state(conn_id).last_item_id = input_item_id
            events.append(InputAudioBufferSpeechStartedEvent(
                type="input_audio_buffer.speech_started",
                event_id=self._next_event_id(),
                audio_start_ms=msg.get("audio_start_ms", 0),
                item_id=input_item_id,
            ))

        elif msg_type == "speech_stopped":
            duration_s = msg.get("duration_s", 0.0)
            if duration_s:
                self._state(conn_id).input_audio_duration_s = duration_s
            events.append(InputAudioBufferSpeechStoppedEvent(
                type="input_audio_buffer.speech_stopped",
                event_id=self._next_event_id(),
                audio_end_ms=msg.get("audio_end_ms", msg.get("audio_stop_ms", 0)),
                item_id=self._current_item_id(conn_id),
            ))

        elif msg_type == "assistant_text":
            resp_id, item_id = self._ensure_response(conn_id)
            self._state(conn_id).last_item_id = item_id
            output_idx = 0
            text = msg.get("text", "")
            if text:
                events.append(ResponseAudioTranscriptDoneEvent(
                    type="response.output_audio_transcript.done",
                    event_id=self._next_event_id(),
                    content_index=0,
                    item_id=item_id,
                    output_index=output_idx,
                    response_id=resp_id,
                    transcript=text,
                ))
                output_idx += 1
            tools = msg.get("tools")
            if tools:
                for tool in tools:
                    logger.info(f"Tool: {tool}")
                    events.append(
                        ResponseFunctionCallArgumentsDoneEvent(
                            type="response.function_call_arguments.done",
                            event_id=self._next_event_id(),
                            call_id=tool.get("call_id", ""),
                            name=tool.get("name", ""),
                            arguments=json.dumps(tool.get("arguments", {})),
                            item_id=item_id,
                            output_index=output_idx,
                            response_id=resp_id,
                        )
                    )
                    output_idx += 1

        elif msg_type == "partial_transcription":
            events.append(
                ConversationItemInputAudioTranscriptionDeltaEvent(
                    type="conversation.item.input_audio_transcription.delta",
                    event_id=self._next_event_id(),
                    content_index=self._next_content_index(conn_id),
                    item_id=self._current_item_id(conn_id),
                    delta=msg.get("delta", ""),
                )
            )

        elif msg_type == "transcription_completed":
            st = self._state(conn_id)
            events.append(
                ConversationItemInputAudioTranscriptionCompletedEvent(
                    type="conversation.item.input_audio_transcription.completed",
                    event_id=self._next_event_id(),
                    content_index=0,
                    item_id=self._current_item_id(conn_id),
                    transcript=msg.get("transcript", ""),
                    usage=UsageTranscriptTextUsageDuration(
                        seconds=st.input_audio_duration_s, type="duration",
                    ),
                )
            )

        else:
            logger.debug(f"Unhandled pipeline text message type: {msg_type}")

        return events

    # ── Helpers ───────────────────────────────────

    @staticmethod
    def make_error(message: str, _type: str) -> RealtimeErrorEvent:
        return RealtimeErrorEvent(
            type="error",
            error=RealtimeError(message=message, type=_type),
            event_id=RealtimeService._next_event_id(),
        )
