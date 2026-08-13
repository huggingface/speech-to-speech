"""Typed events flowing from pipeline handlers to the realtime router.

VAD and transcription events use ``text_output_queue``; response-dependent
assistant and token-usage events share the TTS output queue with their audio.
The router dispatches both through ``RealtimeService``.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from pydantic import BaseModel, ConfigDict, Field, model_validator

from speech_to_speech.pipeline.messages import (
    AssistantOutputPart,
    AssistantToolCallPart,
    _normalize_assistant_output_fields,
)


class PipelineEvent(BaseModel):
    """Base for protocol-neutral events consumed by the realtime router.

    The ``type`` field mirrors the former dict ``"type"`` key and acts as a
    Pydantic discriminator.
    """

    type: str


# ── VAD events ────────────────────────────────────────────────────────


class SpeechStartedEvent(PipelineEvent):
    type: Literal["speech_started"] = "speech_started"
    audio_start_ms: int = 0
    turn_id: str | None = None
    turn_revision: int | None = None
    reopened: bool = False
    interrupt_response: bool = Field(default=True, exclude=True)


class SpeechStoppedEvent(PipelineEvent):
    type: Literal["speech_stopped"] = "speech_stopped"
    duration_s: float = 0.0
    audio_end_ms: int = 0
    turn_id: str | None = None
    turn_revision: int | None = None


# ── Transcription events (TranscriptionNotifier) ─────────────────────


class PartialTranscriptionEvent(PipelineEvent):
    """Latest cumulative STT hypothesis for one input-audio item.

    ``delta`` is retained for public API compatibility. Its value is a
    cumulative internal hypothesis, not an incremental wire-protocol delta.
    """

    type: Literal["partial_transcription"] = "partial_transcription"
    delta: str
    turn_id: str | None = None
    turn_revision: int | None = None


class TranscriptionCompletedEvent(PipelineEvent):
    type: Literal["transcription_completed"] = "transcription_completed"
    transcript: str
    language_code: Optional[str] = None
    turn_id: str | None = None
    turn_revision: int | None = None
    speech_stopped_at_s: float | None = Field(default=None, exclude=True)


# ── Direct audio events (AudioInputNotifier) ─────────────────────────


class AudioInputCompletedEvent(PipelineEvent):
    """Final VAD audio awaiting realtime response bookkeeping."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["audio_input_completed"] = "audio_input_completed"
    audio: np.ndarray = Field(exclude=True)
    audio_sample_rate: int = 16000
    audio_duration_s: float = 0.0
    turn_id: str | None = None
    turn_revision: int | None = None
    speech_stopped_at_s: float | None = Field(default=None, exclude=True)


# ── LLM output events (LMOutputProcessor) ────────────────────────────


class AssistantOutputEvent(PipelineEvent):
    """Internal ordered assistant output awaiting API-specific serialization."""

    type: Literal["assistant_text"] = "assistant_text"
    parts: list[AssistantOutputPart] = Field(default_factory=list)
    text: str = ""
    tools: list[ResponseFunctionToolCall] = Field(default_factory=list)
    turn_id: str | None = None
    turn_revision: int | None = None
    # Response generation that produced this text, mirroring AudioOutput. Lets the
    # send loop discard stale assistant text by the same generation-aware rule as
    # audio, instead of blanket-dropping while cancel_scope.discarding is set.
    cancel_generation: int | None = None
    response_key: str | None = Field(default=None, exclude=True, repr=False)
    # Monotonic part position assigned by LMOutputProcessor. It lets the
    # realtime side channel expose a later tool call as soon as every earlier
    # model part has reached the ordered TTS path, without letting it overtake
    # a preceding assistant message.
    output_sequence: int | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def _normalize_ordered_parts(self) -> "AssistantOutputEvent":
        legacy_views = _normalize_assistant_output_fields(self.parts, self.text, self.tools, self.model_fields_set)
        if legacy_views is not None:
            self.text, self.tools = legacy_views
        return self


# Kept for callers that imported the original public pipeline symbol.
AssistantTextEvent = AssistantOutputEvent


class AssistantToolCallReadyEvent(PipelineEvent):
    """A tool call that can be exposed without waiting for preceding TTS.

    The matching :class:`AssistantOutputEvent` still travels through the TTS
    queue and owns audio-finalization order. This side-channel event only
    makes the standard function-call arguments event available early enough
    for clients to execute the tool while speech synthesis continues.
    """

    type: Literal["assistant_tool_call_ready"] = "assistant_tool_call_ready"
    part: AssistantToolCallPart
    output_sequence: int
    turn_id: str | None = None
    turn_revision: int | None = None
    cancel_generation: int | None = None
    response_key: str | None = Field(default=None, exclude=True, repr=False)


class TokenUsageEvent(PipelineEvent):
    type: Literal["token_usage"] = "token_usage"
    input_tokens: int = 0
    output_tokens: int = 0
    turn_id: str | None = None
    turn_revision: int | None = None
    cancel_generation: int | None = None
    response_key: str | None = Field(default=None, exclude=True, repr=False)


class AssistantResponseDoneEvent(PipelineEvent):
    """Marks the end of ordered assistant output for one response."""

    type: Literal["assistant_response_done"] = "assistant_response_done"
    response_key: str | None = Field(default=None, exclude=True, repr=False)
    turn_id: str | None = None
    turn_revision: int | None = None
    cancel_generation: int | None = None


class ResponseGenerationDoneEvent(PipelineEvent):
    """Internal signal that LM generation ended before downstream TTS drains.

    This event travels on the side channel and is never serialized to a
    Realtime client.  ``call_ids`` identifies tool calls produced by the
    generation so the server can safely precompute a follow-up once their
    outputs have arrived.
    """

    type: Literal["response_generation_done"] = "response_generation_done"
    response_key: str | None = Field(default=None, exclude=True, repr=False)
    call_ids: list[str] = Field(default_factory=list)
    succeeded: bool = True
    turn_id: str | None = None
    turn_revision: int | None = None
    cancel_generation: int | None = None


class ResponseFailedEvent(PipelineEvent):
    """Signals that a response could not be generated (e.g. invalid out-of-band input).

    Dispatched to the service so it can close the in-progress response with
    ``status="failed"`` instead of the usual ``completed``.
    """

    type: Literal["response_failed"] = "response_failed"
    message: str = ""
    turn_id: str | None = None
    turn_revision: int | None = None
    cancel_generation: int | None = None
    response_key: str | None = Field(default=None, exclude=True, repr=False)
