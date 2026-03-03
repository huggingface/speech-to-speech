import base64
import json
import logging
from dataclasses import dataclass
from queue import Queue
from threading import Event as ThreadingEvent
from typing import Optional, Union

from pydantic import ValidationError

from openai.types.realtime.realtime_response_create_params import ToolChoiceOptions, ToolChoiceMcp
from openai.types.realtime.realtime_response import Audio, AudioOutput

from openai.types.realtime import (
    InputAudioBufferAppendEvent,
    SessionUpdateEvent,
    ConversationItemCreateEvent,
    ResponseCreateEvent,
    ResponseCancelEvent,
    SessionCreatedEvent,
    RealtimeErrorEvent,
    RealtimeError,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ResponseCreatedEvent,
    RealtimeResponse,
    ResponseDoneEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent
)

from api.openai_realtime.protocol import (
    ConversationItemInputAudioTranscriptionPartial, # Should be use ConversationItemInputAudioTranscriptionDeltaEvent instead
)
from api.openai_realtime.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

CHUNK_SAMPLES = 512
BYTES_PER_SAMPLE = 2
CHUNK_SIZE_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE

_EVENT_TYPE_TO_MODEL: dict[str, type] = {
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
    ConversationItemInputAudioTranscriptionPartial,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent
]

RealtimeEvent = Union[ClientEvent, ServerEvent]

@dataclass
class _ConnState:
    """Per-connection mutable state."""
    in_response: bool = False
    audio_buffer_has_data: bool = False


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
        self._conns: dict[str, _ConnState] = {}

    # ── Connection lifecycle ─────────────────────

    def register(self, conn_id: str) -> None:
        self._conns[conn_id] = _ConnState()

    def unregister(self, conn_id: str) -> None:
        self._conns.pop(conn_id, None)

    def _state(self, conn_id: str) -> _ConnState:
        return self._conns[conn_id]

    @property
    def connection_ids(self) -> list[str]:
        return list(self._conns)

    # ── Client event parsing ─────────────────────

    def parse_client_event(self, raw: dict) -> Optional[ClientEvent]:
        event_type = raw.get("type")
        model_cls = _EVENT_TYPE_TO_MODEL.get(event_type)
        if model_cls is None:
            logger.warning(f"Unknown client event type: {event_type}")
            return None
        try:
            return model_cls.model_validate(raw)
        except ValidationError as e:
            logger.error(f"Invalid {event_type} payload: {e}")
            return None

    # ── Client event handlers ────────────────────

    def handle_audio_append(self, conn_id: str, event: InputAudioBufferAppendEvent) -> list[bytes]:
        """Decode base64 audio and split into 512-sample PCM16 chunks for the VAD."""
        try:
            pcm_bytes = base64.b64decode(event.audio)
        except Exception as e:
            logger.error(f"Base64 decode error: {e}")
            return []

        chunks = []
        for i in range(0, len(pcm_bytes), CHUNK_SIZE_BYTES):
            chunk = pcm_bytes[i : i + CHUNK_SIZE_BYTES]
            if len(chunk) == CHUNK_SIZE_BYTES:
                chunks.append(chunk)
        if chunks:
            self._state(conn_id).audio_buffer_has_data = True
        return chunks

    def handle_session_update(self, event: SessionUpdateEvent) -> None:
        """Apply session config changes to the shared RuntimeConfig.

        Supports both the nested format from the OpenAI SDK::

            session.audio.input.turn_detection
            session.audio.input.transcription
            session.audio.output.voice

        and flat fields for simple clients.  Nested ``session`` values take
        priority over flat fields when both are present.
        """
        cfg = self.runtime_config

        s = event.session or {}
        audio = s.get("audio") or {}
        audio_in = audio.get("input") or {}
        audio_out = audio.get("output") or {}

        model = event.model
        instructions = s.get("instructions") or event.instructions
        voice = audio_out.get("voice") or event.voice
        turn_detection = audio_in.get("turn_detection") or event.turn_detection
        transcription = audio_in.get("transcription") or event.input_audio_transcription
        tools = s.get("tools") or event.tools
        tool_choice = s.get("tool_choice") or event.tool_choice

        if model is not None:
            logger.info(f"Session model set to: {model}")
        if voice is not None:
            cfg.voice = voice
            logger.info(f"Voice updated to: {voice}")
        if instructions is not None:
            cfg.instructions = instructions
            logger.info(f"Instructions updated ({len(instructions)} chars)")
        if turn_detection is not None:
            cfg.turn_detection = turn_detection
            logger.info(f"Turn detection updated: {turn_detection}")
        if transcription is not None:
            cfg.input_audio_transcription = transcription
            logger.info("Input audio transcription config updated")
        if tools is not None:
            cfg.tools = tools
            logger.info(f"Tools updated ({len(tools)} tools)")
        if tool_choice is not None:
            cfg.tool_choice = tool_choice
            logger.info(f"Tool choice updated to: {tool_choice}")

    def handle_conversation_item_create(self, conn_id: str, event: ConversationItemCreateEvent) -> list[ServerEvent]:
        """Inject a text message or function-call output directly into the LLM queue.

        Supported content part types:
        - ``input_text``: forwarded to the LLM text-prompt queue.
        - ``input_image``: acknowledged but not processed (image support is
          not implemented in the local pipeline). A log entry is emitted so
          callers can observe the event.
        """
        events: list[ServerEvent] = []
        item = event.item

        if item.type == "message" and item.content:
            for part in item.content:
                if part.type == "input_text" and part.text:
                    if self.text_prompt_queue:
                        self.text_prompt_queue.put(part.text)
                        logger.info(f"Injected text to LLM: {part.text!r:.80}")
                    if self.should_listen:
                        self.should_listen.clear()
                    events.append(
                        ConversationItemInputAudioTranscriptionCompletedEvent(content_index=0, event_id=conn_id, item_id=item.id, transcript=part.text)
                    )
                elif part.type == "input_image":
                    # Image content is received but not forwarded to the
                    # local pipeline — vision is not supported server-side.
                    logger.info(
                        "Received input_image content part; "
                        "ignoring — image processing not yet supported in local pipeline",
                    )

        elif item.type == "function_call_output" and item.output:
            if self.text_prompt_queue:
                self.text_prompt_queue.put(item.output)
                logger.info("Injected function_call_output to LLM")

        return events

    def handle_audio_commit(self, conn_id: str) -> RealtimeErrorEvent | None:
        """Commit the audio buffer. Returns an error if no audio was appended."""
        st = self._state(conn_id)
        if not st.audio_buffer_has_data:
            return self.make_error(
                message="Input audio buffer is empty, nothing to commit.",
                _type="input_audio_buffer_commit_empty",
                event_id=conn_id
            )
        st.audio_buffer_has_data = False
        logger.debug("Audio buffer committed")
        return None

    def handle_response_create(self, conn_id: str, event: ResponseCreateEvent) -> RealtimeErrorEvent | None:
        """Trigger a response. Returns an error if a response is already in progress.

        Per-response overrides (instructions, tool_choice) from the ``response``
        field are written to ``RuntimeConfig`` so pipeline handlers can pick
        them up via ``consume_response_overrides()``.
        """
        if self._state(conn_id).in_response:
            return self.make_error(
                message="Cannot create response while another response is in progress.",
                _type="conversation_already_has_active_response",
                event_id=conn_id
            )
        if event.response:
            cfg = self.runtime_config
            if event.response.instructions:
                cfg.response_instructions = event.response.instructions
                logger.info(
                    "Per-response instructions set (%d chars): %s...",
                    len(event.response.instructions),
                    event.response.instructions[:60],
                )
            if event.response.tool_choice:
                if isinstance(event.response.tool_choice, ToolChoiceOptions) or isinstance(event.response.tool_choice, ToolChoiceMcp):
                    logger.warning("ToolChoiceMcp or ToolChoiceOptions is not yet supported")
                    return self.make_error(
                        message="ToolChoiceMcp or ToolChoiceOptions is not yet supported",
                        _type="tool_choice_not_supported",
                        event_id=conn_id
                    )
                cfg.response_tool_choice = event.response.tool_choice
                logger.info("Per-response tool_choice set to: %s", event.response.tool_choice)
        logger.debug("response.create received")
        return None

    def handle_response_cancel(self, conn_id: str) -> list[ServerEvent]:
        """Cancel the in-progress response and re-enable listening."""
        events = self.finish_audio_response(conn_id)
        if self.should_listen:
            self.should_listen.set()
        logger.info("Response cancelled, listening re-enabled")
        return events

    # ── Outbound audio encoding ──────────────────

    def encode_audio_chunk(self, conn_id: str, audio: bytes) -> list[ServerEvent]:
        """Encode a raw PCM audio chunk, emitting ResponseCreated on the first chunk."""
        st = self._state(conn_id)
        events: list[ServerEvent] = []
        if not st.in_response:
            events.append(ResponseCreatedEvent(event_id=conn_id, response=RealtimeResponse(id=conn_id, audio=Audio(output=AudioOutput(voice=self.runtime_config.voice)))))
            st.in_response = True
        b64 = base64.b64encode(audio).decode("ascii")
        # TODO: should be add to events list when the model-generated audio is updated.
        events.append(ResponseAudioDeltaEvent(event_id=conn_id, content_index=0, delta=b64, item_id=conn_id, output_index=0, response_id=conn_id))
        return events

    def finish_audio_response(self, conn_id: str) -> list[ServerEvent]:
        """Close the current response (audio done + response done)."""
        st = self._state(conn_id)
        events: list[ServerEvent] = []
        if st.in_response:
            events.append(ResponseAudioDoneEvent(event_id=conn_id, content_index=0, item_id=conn_id, output_index=0, response_id=conn_id))
            events.append(ResponseDoneEvent(
                event_id=conn_id,
                response=RealtimeResponse(id=conn_id, audio=Audio(output=AudioOutput(voice=self.runtime_config.voice))),
            ))
            st.in_response = False
        return events

    # ── Pipeline text -> protocol events ─────────

    def translate_pipeline_text(self, conn_id: str, msg: dict) -> list[ServerEvent]:
        """Convert an internal text_output_queue message to protocol ServerEvent(s)."""
        events: list[ServerEvent] = []
        msg_type = msg.get("type")

        if msg_type == "speech_started":
            st = self._state(conn_id)
            if st.in_response:
                events.extend(self.finish_audio_response(conn_id))
            events.append(InputAudioBufferSpeechStartedEvent(audio_start_ms=msg.get("audio_start_ms", 0), item_id=msg.get("item_id", ""), event_id=conn_id))

        elif msg_type == "speech_stopped":
            events.append(InputAudioBufferSpeechStoppedEvent(audio_stop_ms=msg.get("audio_stop_ms", 0), item_id=msg.get("item_id", ""), event_id=conn_id))

        elif msg_type == "assistant_text":
            tools = msg.get("tools")
            if tools:
                for tool in tools:
                    events.append(
                        ResponseFunctionCallArgumentsDoneEvent(
                            event_id=conn_id,
                            call_id=tool.get("call_id", ""),
                            name=tool.get("name", ""),
                            arguments=json.dumps(tool.get("arguments", {})),
                            item_id=conn_id,
                            output_index=0,
                            response_id=conn_id,
                        )
                    )
            text = msg.get("text", "")
            if text:
                events.append(ResponseAudioTranscriptDoneEvent(event_id=conn_id, content_index=0, item_id=conn_id, output_index=0, response_id=conn_id, transcript=text))

        elif msg_type == "partial_transcription":
            events.append(
                ConversationItemInputAudioTranscriptionPartial(
                    content_index=0,
                    event_id=conn_id,
                    item_id=msg.get("item_id", ""),
                    transcript=msg.get("transcript", "")
                )
            )

        elif msg_type == "transcription_completed":
            events.append(
                ConversationItemInputAudioTranscriptionCompletedEvent(content_index=0, event_id=conn_id, item_id=msg.get("item_id", ""), transcript=msg.get("transcript", ""))
            )

        else:
            logger.debug(f"Unhandled pipeline text message type: {msg_type}")

        return events

    # ── Helpers ───────────────────────────────────

    @staticmethod
    def make_error(message: str, _type: str, event_id: str) -> RealtimeErrorEvent:
        return RealtimeErrorEvent(error=RealtimeError(message=message, type=_type), event_id=event_id)
