import logging
from collections.abc import Mapping
from queue import Queue
from threading import Event as ThreadingEvent
from typing import Any, Callable, Literal, Optional, TypeVar, Union

from openai.types.realtime import (
    ConversationItem,
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferCommitEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    OutputAudioBufferClearEvent,
    RealtimeError,
    RealtimeErrorEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCancelEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    SessionCreatedEvent,
    SessionUpdateEvent,
)
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from speech_to_speech.api.openai_realtime.handlers import (
    AudioHandler,
    ConversationHandler,
    ResponseHandler,
    SessionHandler,
)
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.pipeline.events import (
    AssistantTextEvent,
    PartialTranscriptionEvent,
    PipelineEvent,
    ResponseFailedEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TokenUsageEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.messages import GenerateResponseRequest
from speech_to_speech.pipeline.queue_types import TextPromptItem
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.utils.utils import _generate_id

logger = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512
BYTES_PER_SAMPLE = 2
CHUNK_SIZE_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE

_ResponseStatus = Literal["completed", "cancelled", "failed", "incomplete", "in_progress"]
_StatusReason = Literal["turn_detected", "client_cancelled", "max_output_tokens", "content_filter"]

_EVENT_TYPE_TO_MODEL: dict[str, type[BaseModel]] = {
    "input_audio_buffer.append": InputAudioBufferAppendEvent,
    "input_audio_buffer.commit": InputAudioBufferCommitEvent,
    "output_audio_buffer.clear": OutputAudioBufferClearEvent,
    "session.update": SessionUpdateEvent,
    "conversation.item.create": ConversationItemCreateEvent,
    "response.create": ResponseCreateEvent,
    "response.cancel": ResponseCancelEvent,
}

ClientEvent = Union[
    InputAudioBufferAppendEvent,
    InputAudioBufferCommitEvent,
    OutputAudioBufferClearEvent,
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
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
]

RealtimeEvent = Union[ClientEvent, ServerEvent]


_UsageMetricsT = TypeVar("_UsageMetricsT", bound="UsageMetrics")


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

    def __iadd__(self: _UsageMetricsT, other: "UsageMetrics") -> _UsageMetricsT:
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str = Field(default_factory=lambda: _generate_id("session"))
    conversation_id: str = Field(default_factory=lambda: _generate_id("conv"))
    runtime_config: RuntimeConfig = Field(default_factory=RuntimeConfig)
    in_response: bool = False
    response_pending: bool = False
    audio_buffer_has_data: bool = False
    audio_remainder: bytes = b""
    current_response_id: Optional[str] = None
    current_item_id: Optional[str] = None
    content_index: int = 0
    input_content_index: int = 0
    input_audio_duration_s: float = 0.0
    last_item_id: Optional[str] = None
    current_response_params: RealtimeResponseCreateParams | None = None
    pending_output_text_parts: list[str] = Field(default_factory=list)
    response_usage: UsageMetrics = Field(default_factory=UsageMetrics)
    speculative_turn_id: Optional[str] = None
    speculative_turn_revision: Optional[int] = None
    speculative_user_turn_id: Optional[str] = None
    speculative_user_turn_revision: Optional[int] = None
    speculative_user_speech_stopped_at_s: Optional[float] = None
    speculative_user_item_id: Optional[str] = None
    speculative_input_item_id: Optional[str] = None
    speculative_audio_duration_s: float = 0.0
    # Client conversation.item.create items that arrived while a response was
    # generating. Applying them mid-generation races the LLM handler's chat
    # write-back (cross-thread), so they are buffered here and flushed in order
    # once the response completes. See ConversationHandler.flush_deferred_items.
    deferred_items: list[ConversationItem] = Field(default_factory=list)


class RealtimeService:
    """Translates between OpenAI Realtime protocol events and internal pipeline messages.

    One instance is shared across all WebSocket connections.  Per-connection
    state (response lifecycle, audio buffer) is tracked internally by
    connection id.
    """

    def __init__(
        self,
        text_prompt_queue: Queue[TextPromptItem] | None = None,
        should_listen: ThreadingEvent | None = None,
        chat_size: int = 10,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        self.text_prompt_queue = text_prompt_queue
        self.should_listen = should_listen
        self._chat_size = chat_size
        self.speculative_turns = speculative_turns
        self._conns: dict[str, ConnState] = {}
        self.total_usage = GlobalUsageMetrics()

        self.audio = AudioHandler(self)
        self.session = SessionHandler(self)
        self.response = ResponseHandler(self)
        self.conversation = ConversationHandler(self)

        self._pipeline_dispatch: dict[type[PipelineEvent], Callable[..., list[ServerEvent]]] = {
            SpeechStartedEvent: self.audio.on_speech_started,
            SpeechStoppedEvent: self.audio.on_speech_stopped,
            TokenUsageEvent: self._on_token_usage,
            PartialTranscriptionEvent: self.conversation.on_partial_transcription,
            TranscriptionCompletedEvent: self._on_transcription_completed,
            ResponseFailedEvent: self._on_response_failed,
        }

    # ── Connection lifecycle ─────────────────────

    def register(self) -> str:
        """Register a new connection and return its session_id."""
        if self.speculative_turns:
            self.speculative_turns.reset()
        state = ConnState(runtime_config=RuntimeConfig(chat=Chat(self._chat_size)))
        self._conns[state.session_id] = state
        self.total_usage.connections += 1
        return state.session_id

    def unregister(self, conn_id: str) -> None:
        st = self._conns.pop(conn_id, None)
        if st is not None:
            # Suppress any in-flight compaction splice so a daemon worker can't
            # mutate a Chat tied to a closed session, and don't make further
            # billable LLM calls on its behalf once the splice is suppressed.
            st.runtime_config.chat.close()
            self.total_usage += st.response_usage
            logger.info(
                "Session %s unregistered — cumulative: input_tokens=%d, output_tokens=%d, audio=%.2fs",
                conn_id,
                self.total_usage.input_tokens,
                self.total_usage.output_tokens,
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

    def parse_client_event(self, raw: Mapping[str, object]) -> Optional[ClientEvent]:
        raw_type = raw.get("type")
        event_type: Optional[str] = raw_type if isinstance(raw_type, str) else None
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

    def append_pcm(self, conn_id: str, pcm_bytes: bytes, src_rate: int) -> list[bytes]:
        return self.audio.append_pcm(conn_id, pcm_bytes, src_rate)

    def handle_audio_commit(self, conn_id: str) -> RealtimeErrorEvent | None:
        return self.audio.handle_audio_commit(conn_id)

    def begin_audio_response(self, conn_id: str) -> tuple[str, str, list[ServerEvent]]:
        return self.audio.begin_audio_response(conn_id)

    def encode_audio_chunk(self, conn_id: str, audio: bytes) -> list[ServerEvent]:
        return self.audio.encode_audio_chunk(conn_id, audio)

    def handle_response_create(self, conn_id: str, event: ResponseCreateEvent) -> ServerEvent | None:
        return self.response.handle_response_create(conn_id, event)

    def handle_response_cancel(self, conn_id: str) -> list[ServerEvent]:
        return self.response.handle_response_cancel(conn_id)

    def finish_response(
        self,
        conn_id: str,
        status: _ResponseStatus = "completed",
        reason: _StatusReason | None = None,
    ) -> list[ServerEvent]:
        return self.response.finish_response(conn_id, status, reason)

    def handle_conversation_item_create(self, conn_id: str, event: ConversationItemCreateEvent) -> list[ServerEvent]:
        return self.conversation.handle_conversation_item_create(conn_id, event)

    def dispatch_pipeline_event(self, conn_id: str, event: PipelineEvent) -> list[ServerEvent]:
        """Route a pipeline text_output_queue event to the appropriate handler."""
        events = self._dispatch_pipeline_event(conn_id, event, wait_for_pending_reopen=True)
        return [] if events is None else events

    def try_dispatch_pipeline_event(self, conn_id: str, event: PipelineEvent) -> list[ServerEvent] | None:
        """Non-blocking dispatch.

        Returns ``None`` when dispatch must be retried after a speculative
        reopen candidate resolves.
        """
        return self._dispatch_pipeline_event(conn_id, event, wait_for_pending_reopen=False)

    def should_defer_pipeline_event(self, event: PipelineEvent) -> bool:
        if self.speculative_turns is None or not isinstance(event, (AssistantTextEvent, TokenUsageEvent)):
            return False
        return self.speculative_turns.has_pending_reopen_or_grace(
            getattr(event, "turn_id", None),
            getattr(event, "turn_revision", None),
        )

    def _dispatch_pipeline_event(
        self,
        conn_id: str,
        event: PipelineEvent,
        *,
        wait_for_pending_reopen: bool,
    ) -> list[ServerEvent] | None:
        is_stale = self._is_stale_turn_event(event, wait_for_pending_reopen=wait_for_pending_reopen)
        if is_stale is None:
            return None
        if is_stale:
            logger.info(
                "Ignoring stale %s for turn=%s rev=%s",
                event.type,
                getattr(event, "turn_id", None),
                getattr(event, "turn_revision", None),
            )
            return []

        self._observe_turn_event(event)
        if isinstance(event, AssistantTextEvent):
            return self.response.on_assistant_text(
                conn_id,
                event,
                wait_for_pending_reopen=wait_for_pending_reopen,
            )
        handler = self._pipeline_dispatch.get(type(event))
        if handler is None:
            logger.debug("Unhandled pipeline event type: %s", type(event).__name__)
            return []
        return handler(conn_id, event)

    def _is_stale_turn_event(self, event: PipelineEvent, *, wait_for_pending_reopen: bool = True) -> bool | None:
        if self.speculative_turns is None:
            return False
        if not isinstance(
            event,
            (PartialTranscriptionEvent, TranscriptionCompletedEvent, AssistantTextEvent, TokenUsageEvent),
        ):
            return False
        turn_id = getattr(event, "turn_id", None)
        turn_revision = getattr(event, "turn_revision", None)
        if isinstance(event, (AssistantTextEvent, TokenUsageEvent)):
            is_latest: bool | None
            if wait_for_pending_reopen:
                is_latest = self.speculative_turns.is_latest_after_reopen_grace(turn_id, turn_revision)
            else:
                is_latest = self.speculative_turns.try_is_latest_after_reopen_grace(turn_id, turn_revision)
            if is_latest is None:
                return None
            return not is_latest
        return not self.speculative_turns.is_latest(turn_id, turn_revision)

    def _observe_turn_event(self, event: PipelineEvent) -> None:
        if self.speculative_turns is None:
            return
        self.speculative_turns.observe(
            getattr(event, "turn_id", None),
            getattr(event, "turn_revision", None),
        )

    # ── STT → LM bridge ────────────────────────────

    def _on_transcription_completed(self, conn_id: str, event: TranscriptionCompletedEvent) -> list[ServerEvent]:
        """Handle a final STT transcription: emit protocol event, append to chat, trigger LM."""
        st = self._state(conn_id)
        same_speculative_turn = event.turn_id is not None and event.turn_id == st.speculative_user_turn_id
        if same_speculative_turn:
            st.response_usage.audio_duration_s -= st.speculative_audio_duration_s
        else:
            st.speculative_audio_duration_s = 0.0

        events = self.conversation.on_transcription_completed(conn_id, event)
        if event.turn_id is not None:
            st.speculative_audio_duration_s = st.input_audio_duration_s

        cfg = st.runtime_config
        transcript = event.transcript
        if transcript:
            if same_speculative_turn and st.speculative_user_item_id:
                replaced = cfg.chat.replace_user_message_text(st.speculative_user_item_id, transcript)
                if not replaced:
                    item = cfg.chat.add_item(make_user_message(transcript))
                    st.speculative_user_item_id = item.id
            else:
                item = cfg.chat.add_item(make_user_message(transcript))
                st.speculative_user_item_id = item.id
        elif same_speculative_turn and st.speculative_user_item_id:
            cfg.chat.remove_user_message(st.speculative_user_item_id)
            st.speculative_user_item_id = None
        elif event.turn_id is not None and event.turn_id != st.speculative_user_turn_id:
            st.speculative_user_item_id = None

        if event.turn_id is not None:
            st.speculative_user_turn_id = event.turn_id
            st.speculative_user_turn_revision = event.turn_revision
            st.speculative_user_speech_stopped_at_s = event.speech_stopped_at_s

        queue = self.text_prompt_queue
        if queue and transcript:
            st.response_pending = True
            queue.put(
                GenerateResponseRequest(
                    runtime_config=cfg,
                    language_code=event.language_code,
                    turn_id=event.turn_id,
                    turn_revision=event.turn_revision,
                    speech_stopped_at_s=event.speech_stopped_at_s,
                )
            )

        return events

    # ── Metrics ────────────────────────────────────

    def _on_token_usage(self, conn_id: str, event: TokenUsageEvent) -> list[ServerEvent]:
        """Accumulate input/output token counts on the connection's usage metrics."""
        if self.speculative_turns and not self.speculative_turns.is_latest(
            event.turn_id,
            event.turn_revision,
        ):
            logger.debug("Dropping stale token usage for turn=%s rev=%s", event.turn_id, event.turn_revision)
            return []
        st = self._state(conn_id)
        st.response_usage.input_tokens += event.input_tokens
        st.response_usage.output_tokens += event.output_tokens
        logger.info(
            "Token usage (response): input=%d, output=%d",
            st.response_usage.input_tokens,
            st.response_usage.output_tokens,
        )
        return []

    def _on_response_failed(self, conn_id: str, event: ResponseFailedEvent) -> list[ServerEvent]:
        """Surface the failure to the client and close the response as ``failed``.

        Emitted when generation failed (e.g. invalid out-of-band input, or the
        provider rejecting an empty context). A top-level ``error`` event carries
        the human-readable reason — ``response.done.status_details.error`` only
        has code/type, no message — then ``finish_response`` closes the slot.

        Idempotent: gated on an active response, and ``finish_response`` is itself
        a no-op once the slot is closed, so a later EndOfResponse-driven close does
        nothing.
        """
        logger.info("Response failed: %s", event.message)
        if not self._state(conn_id).in_response:
            return []
        events: list[ServerEvent] = [self.make_error(event.message, "response_failed")]
        events.extend(self.response.finish_response(conn_id, status="failed"))
        return events

    def get_usage(self) -> dict[str, Any]:
        """Return cumulative usage metrics across all completed responses."""
        data = self.total_usage.model_dump()
        data["total_tokens"] = data["input_tokens"] + data["output_tokens"]
        data["total_errors"] = self.total_usage.total_errors
        return data

    # ── Error ───────────────────────────────────

    def make_error(self, message: str, _type: str) -> RealtimeErrorEvent:
        self.total_usage.record_error(_type)
        return build_error_event(message, _type)


def build_error_event(message: str, error_type: str) -> RealtimeErrorEvent:
    """Construct a RealtimeErrorEvent without touching any service-instance state.

    Used by the websocket route handler on pool rejection, where no unit's
    service should be charged with the error in its usage metrics.
    """
    return RealtimeErrorEvent(
        type="error",
        error=RealtimeError(message=message, type=error_type),
        event_id=_generate_id("event"),
    )
