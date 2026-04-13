import logging
from pydantic import BaseModel, Field
from queue import Queue
from threading import Event as ThreadingEvent
from typing import Callable, Literal, Optional, Union

from pydantic import ValidationError

from openai.types.realtime import (
    InputAudioBufferAppendEvent,
    SessionUpdateEvent,
    ConversationItemCreateEvent,
    ConversationItemCreatedEvent,
    ResponseCreateEvent,
    ResponseCancelEvent,
    SessionCreatedEvent,
    RealtimeErrorEvent,
    RealtimeError,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent,
)

from api.openai_realtime.runtime_config import RuntimeConfig
from api.openai_realtime.handlers import (
    AudioHandler,
    ConversationHandler,
    ResponseHandler,
    SessionHandler,
)
from api.openai_realtime.handlers.base import _generate_id

logger = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512
BYTES_PER_SAMPLE = 2
CHUNK_SIZE_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE

_ResponseStatus = Literal["completed", "cancelled", "failed", "incomplete", "in_progress"]
_StatusReason = Literal["turn_detected", "client_cancelled", "max_output_tokens", "content_filter"]

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
    ResponseFunctionCallArgumentsDoneEvent,
]

RealtimeEvent = Union[ClientEvent, ServerEvent]


class UsageMetrics(BaseModel):
    """Per-response usage counters.

    Supports ``+=`` for rolling per-response metrics into a global total
    and ``reset()`` for clearing per-response state after rollup.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    audio_duration_s: float = 0.0
    responses_completed: int = 0
    responses_cancelled: int = 0
    tool_calls: int = 0
    turns: int = 0

    def __iadd__(self, other: "UsageMetrics") -> "UsageMetrics":
        for field in UsageMetrics.model_fields:
            setattr(self, field, getattr(self, field) + getattr(other, field))
        return self

    def reset(self) -> None:
        for field, info in UsageMetrics.model_fields.items():
            setattr(self, field, info.default)


class GlobalUsageMetrics(UsageMetrics):
    """Server-wide metrics that extend per-response counters with
    connection and error tracking."""
    connections: int = 0
    # connection duration in seconds.
    # latency tts, llm, vad, stt (mean, max, p90)
    errors_by_type: dict[str, int] = Field(default_factory=dict)

    def record_error(self, error_type: str) -> None:
        self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1

    @property
    def total_errors(self) -> int:
        return sum(self.errors_by_type.values())


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
    response_usage: UsageMetrics = Field(default_factory=UsageMetrics)


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
        self.total_usage = GlobalUsageMetrics()

        self.audio = AudioHandler(self)
        self.session = SessionHandler(self)
        self.response = ResponseHandler(self)
        self.conversation = ConversationHandler(self)

        self._pipeline_dispatch: dict[str, Callable[[str, dict], list[ServerEvent]]] = {
            "speech_started": self.audio.on_speech_started,
            "speech_stopped": self.audio.on_speech_stopped,
            "assistant_text": self.response.on_assistant_text,
            "token_usage": self._on_token_usage,
            "partial_transcription": self.conversation.on_partial_transcription,
            "transcription_completed": self.conversation.on_transcription_completed,
        }

    # ── Connection lifecycle ─────────────────────

    def register(self) -> str:
        """Register a new connection and return its session_id."""
        state = ConnState()
        self._conns[state.session_id] = state
        self.total_usage.connections += 1
        return state.session_id

    def unregister(self, conn_id: str) -> None:
        st = self._conns.pop(conn_id, None)
        if st is not None:
            self.total_usage += st.response_usage
            logger.info(
                "Session %s unregistered — cumulative: input_tokens=%d, output_tokens=%d, audio=%.2fs",
                conn_id, self.total_usage.input_tokens, self.total_usage.output_tokens,
                self.total_usage.audio_duration_s,
            )

    def _state(self, conn_id: str) -> ConnState:
        return self._conns[conn_id]

    @property
    def connection_ids(self) -> list[str]:
        return list(self._conns)

    # ── Client event parsing ─────────────────────

    @staticmethod
    def _next_event_id() -> str:
        return _generate_id("event")

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

    def build_session_created(self, conn_id: str) -> SessionCreatedEvent:
        return self.session.build_session_created(conn_id)

    def handle_session_update(self, conn_id: str, event: SessionUpdateEvent) -> Optional[RealtimeErrorEvent]:
        return self.session.handle_session_update(conn_id, event)

    def handle_audio_append(self, conn_id: str, event: InputAudioBufferAppendEvent) -> list[bytes]:
        return self.audio.handle_audio_append(conn_id, event)

    def handle_audio_commit(self, conn_id: str) -> RealtimeErrorEvent | None:
        return self.audio.handle_audio_commit(conn_id)

    def encode_audio_chunk(self, conn_id: str, audio: bytes) -> list[ServerEvent]:
        return self.audio.encode_audio_chunk(conn_id, audio)

    def handle_response_create(self, conn_id: str, event: ResponseCreateEvent) -> ServerEvent | None:
        return self.response.handle_response_create(conn_id, event)

    def handle_response_cancel(self, conn_id: str) -> list[ServerEvent]:
        return self.response.handle_response_cancel(conn_id)

    def finish_audio_response(
        self, conn_id: str, status: _ResponseStatus = "completed", reason: _StatusReason | None = None,
    ) -> list[ServerEvent]:
        return self.response.finish_audio_response(conn_id, status, reason)

    def handle_conversation_item_create(self, conn_id: str, event: ConversationItemCreateEvent) -> list[ServerEvent]:
        return self.conversation.handle_conversation_item_create(conn_id, event)

    def dispatch_pipeline_event(self, conn_id: str, msg: dict) -> list[ServerEvent]:
        """Route a pipeline text_output_queue message to the appropriate handler."""
        msg_type = msg.get("type")
        handler = self._pipeline_dispatch.get(msg_type)
        if handler is None:
            logger.debug("Unhandled pipeline message type: %s", msg_type)
            return []
        return handler(conn_id, msg)

    # ── Metrics ────────────────────────────────────

    def _on_token_usage(self, conn_id: str, msg: dict) -> list[ServerEvent]:
        """Accumulate input/output token counts on the connection's usage metrics."""
        st = self._state(conn_id)
        st.response_usage.input_tokens += msg.get("input_tokens", 0)
        st.response_usage.output_tokens += msg.get("output_tokens", 0)
        logger.info(
            "Token usage (response): input=%d, output=%d",
            st.response_usage.input_tokens, st.response_usage.output_tokens,
        )
        return []

    def get_usage(self) -> dict:
        """Return cumulative usage metrics across all completed responses."""
        data = self.total_usage.model_dump()
        data["total_tokens"] = data["input_tokens"] + data["output_tokens"]
        data["total_errors"] = self.total_usage.total_errors
        return data

    # ── Error ───────────────────────────────────

    def make_error(self, message: str, _type: str) -> RealtimeErrorEvent:
        self.total_usage.record_error(_type)
        return RealtimeErrorEvent(
            type="error",
            error=RealtimeError(message=message, type=_type),
            event_id=self._next_event_id(),
        )
