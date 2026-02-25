import base64
import json
import logging
from typing import Optional

from pydantic import ValidationError

from api.openai_realtime.protocol import (
    ClientEvent,
    ConversationItemInputAudioTranscriptionCompleted,
    ConversationItemInputAudioTranscriptionPartial,
    ErrorEvent,
    InputAudioBufferAppend,
    InputAudioBufferCommit,
    InputAudioBufferSpeechStarted,
    InputAudioBufferSpeechStopped,
    ResponseAudioDelta,
    ResponseAudioDone,
    ResponseAudioTranscriptDone,
    ResponseCreated,
    ResponseDone,
    ResponseFunctionCallArgumentsDone,
    ServerEvent,
    SessionUpdate,
)

logger = logging.getLogger(__name__)

CHUNK_SAMPLES = 512
BYTES_PER_SAMPLE = 2
CHUNK_SIZE_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE

_EVENT_TYPE_TO_MODEL: dict[str, type] = {
    "input_audio_buffer.append": InputAudioBufferAppend,
    "input_audio_buffer.commit": InputAudioBufferCommit,
    "session.update": SessionUpdate,
}


class RealtimeService:
    """Translates between OpenAI Realtime protocol events and internal pipeline messages."""

    def __init__(self):
        self.session_config: dict = {}
        self._in_response: bool = False

    # ── Client event handling ────────────────────

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

    def handle_audio_append(self, event: InputAudioBufferAppend) -> list[bytes]:
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
        return chunks

    def handle_session_update(self, event: SessionUpdate) -> None:
        if event.model is not None:
            self.session_config["model"] = event.model
        logger.info(f"Session config updated: {self.session_config}")

    # ── Outbound audio encoding ──────────────────

    def encode_audio_chunk(self, audio: bytes) -> list[ServerEvent]:
        """Encode a raw PCM audio chunk, emitting ResponseCreated on the first chunk."""
        events: list[ServerEvent] = []
        if not self._in_response:
            events.append(ResponseCreated())
            self._in_response = True
        b64 = base64.b64encode(audio).decode("ascii")
        events.append(ResponseAudioDelta(delta=b64))
        return events

    def finish_audio_response(self) -> list[ServerEvent]:
        """Close the current response (audio done + response done)."""
        events: list[ServerEvent] = []
        if self._in_response:
            events.append(ResponseAudioDone())
            events.append(ResponseDone())
            self._in_response = False
        return events

    # ── Pipeline text -> protocol events ─────────

    def translate_pipeline_text(self, msg: dict) -> list[ServerEvent]:
        """Convert an internal text_output_queue message to protocol ServerEvent(s)."""
        events: list[ServerEvent] = []
        msg_type = msg.get("type")

        if msg_type == "speech_started":
            if self._in_response:
                events.extend(self.finish_audio_response())
            events.append(InputAudioBufferSpeechStarted())

        elif msg_type == "speech_stopped":
            events.append(InputAudioBufferSpeechStopped())

        elif msg_type == "assistant_text":
            tools = msg.get("tools")
            if tools:
                for tool in tools:
                    events.append(
                        ResponseFunctionCallArgumentsDone(
                            call_id=tool.get("call_id", ""),
                            name=tool.get("name", ""),
                            arguments=json.dumps(tool.get("arguments", {})),
                        )
                    )
            text = msg.get("text", "")
            if text:
                events.append(ResponseAudioTranscriptDone(transcript=text))

        elif msg_type == "partial_transcription":
            events.append(
                ConversationItemInputAudioTranscriptionPartial(
                    transcript=msg.get("transcript", "")
                )
            )

        elif msg_type == "transcription_completed":
            events.append(
                ConversationItemInputAudioTranscriptionCompleted(
                    transcript=msg.get("transcript", "")
                )
            )

        else:
            logger.debug(f"Unhandled pipeline text message type: {msg_type}")

        return events

    # ── Helpers ───────────────────────────────────

    @staticmethod
    def make_error(message: str, code: str | None = None) -> ErrorEvent:
        return ErrorEvent(error=message, code=code)
