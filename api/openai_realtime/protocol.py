import time
import uuid
from typing import Literal, Union

from pydantic import BaseModel, Field


def _random_id() -> str:
    return uuid.uuid4().hex[:24]


# ──────────────────────────────────────────────
# Shared / auxiliary models
# ──────────────────────────────────────────────

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ──────────────────────────────────────────────
# Client -> Server events
# ──────────────────────────────────────────────

class InputAudioBufferAppend(BaseModel):
    """Append base64-encoded PCM16 @ 16 kHz audio to the buffer."""

    type: Literal["input_audio_buffer.append"] = "input_audio_buffer.append"
    audio: str


class InputAudioBufferCommit(BaseModel):
    """Signal the server to process the accumulated audio buffer."""

    type: Literal["input_audio_buffer.commit"] = "input_audio_buffer.commit"
    final: bool = False


class SessionUpdate(BaseModel):
    """Update session configuration (e.g. model selection)."""

    type: Literal["session.update"] = "session.update"
    model: str | None = None


# ──────────────────────────────────────────────
# Server -> Client events
# ──────────────────────────────────────────────

class SessionCreated(BaseModel):
    """Sent once when the WebSocket connection is established."""

    type: Literal["session.created"] = "session.created"
    id: str = Field(default_factory=lambda: f"sess-{_random_id()}")
    created: int = Field(default_factory=lambda: int(time.time()))


class ErrorEvent(BaseModel):
    """Error notification."""

    type: Literal["error"] = "error"
    error: str
    code: str | None = None


# ── Input-audio-buffer lifecycle ──────────────

class InputAudioBufferSpeechStarted(BaseModel):
    """VAD detected the start of user speech."""

    type: Literal["input_audio_buffer.speech_started"] = "input_audio_buffer.speech_started"


class InputAudioBufferSpeechStopped(BaseModel):
    """VAD detected the end of user speech."""

    type: Literal["input_audio_buffer.speech_stopped"] = "input_audio_buffer.speech_stopped"


# ── Transcription events ─────────────────────

class TranscriptionDelta(BaseModel):
    """Incremental transcription text (streaming)."""

    type: Literal["transcription.delta"] = "transcription.delta"
    delta: str


class TranscriptionDone(BaseModel):
    """Final transcription with optional usage stats."""

    type: Literal["transcription.done"] = "transcription.done"
    text: str
    usage: UsageInfo | None = None


class ConversationItemInputAudioTranscriptionPartial(BaseModel):
    """Partial (real-time) transcription of user speech."""

    type: Literal["conversation.item.input_audio_transcription.partial"] = (
        "conversation.item.input_audio_transcription.partial"
    )
    transcript: str


class ConversationItemInputAudioTranscriptionCompleted(BaseModel):
    """Final transcription after user finishes speaking."""

    type: Literal["conversation.item.input_audio_transcription.completed"] = (
        "conversation.item.input_audio_transcription.completed"
    )
    transcript: str


# ── Response lifecycle ────────────────────────

class ResponseCreated(BaseModel):
    """A new response has been created."""

    type: Literal["response.created"] = "response.created"


class ResponseDone(BaseModel):
    """The response generation is complete (audio may still be playing)."""

    type: Literal["response.done"] = "response.done"


class ResponseCompleted(BaseModel):
    """Text-only response completion."""

    type: Literal["response.completed"] = "response.completed"


# ── Response audio ────────────────────────────

class ResponseAudioDelta(BaseModel):
    """Incremental base64-encoded audio chunk (GA event name)."""

    type: Literal["response.audio.delta"] = "response.audio.delta"
    delta: str


class ResponseOutputAudioDelta(BaseModel):
    """Incremental base64-encoded audio chunk (GA alias)."""

    type: Literal["response.output_audio.delta"] = "response.output_audio.delta"
    delta: str


class ResponseAudioDone(BaseModel):
    """Audio stream finished (GA event name)."""

    type: Literal["response.audio.done"] = "response.audio.done"


class ResponseOutputAudioDone(BaseModel):
    """Audio stream finished (GA alias)."""

    type: Literal["response.output_audio.done"] = "response.output_audio.done"


class ResponseAudioCompleted(BaseModel):
    """Audio stream finished (legacy event name)."""

    type: Literal["response.audio.completed"] = "response.audio.completed"


# ── Response transcripts (assistant speech) ───

class ResponseAudioTranscriptDone(BaseModel):
    """Final transcript of the assistant's spoken response (GA)."""

    type: Literal["response.audio_transcript.done"] = "response.audio_transcript.done"
    transcript: str


class ResponseOutputAudioTranscriptDone(BaseModel):
    """Final transcript of the assistant's spoken response (GA alias)."""

    type: Literal["response.output_audio_transcript.done"] = "response.output_audio_transcript.done"
    transcript: str


# ── Tool / function calling ───────────────────

class ResponseFunctionCallArgumentsDone(BaseModel):
    """All arguments for a function call have been streamed."""

    type: Literal["response.function_call_arguments.done"] = "response.function_call_arguments.done"
    call_id: str
    name: str
    arguments: str


# ──────────────────────────────────────────────
# Discriminated union of all events
# ──────────────────────────────────────────────

ClientEvent = Union[
    InputAudioBufferAppend,
    InputAudioBufferCommit,
    SessionUpdate,
]

ServerEvent = Union[
    SessionCreated,
    ErrorEvent,
    InputAudioBufferSpeechStarted,
    InputAudioBufferSpeechStopped,
    TranscriptionDelta,
    TranscriptionDone,
    ConversationItemInputAudioTranscriptionPartial,
    ConversationItemInputAudioTranscriptionCompleted,
    ResponseCreated,
    ResponseDone,
    ResponseCompleted,
    ResponseAudioDelta,
    ResponseOutputAudioDelta,
    ResponseAudioDone,
    ResponseOutputAudioDone,
    ResponseAudioCompleted,
    ResponseAudioTranscriptDone,
    ResponseOutputAudioTranscriptDone,
    ResponseFunctionCallArgumentsDone,
]

RealtimeEvent = Union[ClientEvent, ServerEvent]
