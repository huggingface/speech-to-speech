"""Single source of truth for inter-component pipeline messages.

Typed :class:`PipelineMessage` subclasses replace the ad-hoc tuples that
previously flowed between STT, LLM, LMOutputProcessor and TTS stages.
Binary sentinels carried on the audio/output queue are plain ``bytes``
constants.
"""

from __future__ import annotations

from time import perf_counter
from typing import Annotated, Final, Literal, Optional, TypeAlias

import numpy as np
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from pydantic import BaseModel, ConfigDict, Field, model_validator

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig

# ── Base class ────────────────────────────────────────────────────────


class PipelineMessage(BaseModel):
    """Base for all typed pipeline messages.

    The ``tag`` field acts as a Pydantic discriminator so a ``Union`` of
    subtypes can be validated from raw dicts when needed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tag: str


# ── VAD → STT ─────────────────────────────────────────────────────────


class VADAudio(PipelineMessage):
    """Audio segment from VAD, with optional mode for realtime transcription."""

    tag: Literal["vad_audio"] = "vad_audio"
    audio: np.ndarray
    mode: Literal["progressive", "final"] | None = None
    turn_id: str | None = None
    turn_revision: int | None = None
    created_at_s: float = Field(default_factory=perf_counter)


# ── STT → TranscriptionNotifier → LLM ────────────────────────────────


class PartialTranscription(PipelineMessage):
    """Live partial transcription (consumed by TranscriptionNotifier, not forwarded to LLM)."""

    tag: Literal["partial_transcription"] = "partial_transcription"
    text: str
    turn_id: str | None = None
    turn_revision: int | None = None


class Transcription(PipelineMessage):
    """Final transcription result."""

    tag: Literal["transcription"] = "transcription"
    text: str
    language_code: Optional[str] = None
    turn_id: str | None = None
    turn_revision: int | None = None
    speech_stopped_at_s: float | None = None


# ── LLM → LMOutputProcessor ──────────────────────────────────────────


class AssistantTextPart(BaseModel):
    """One ordered assistant text part."""

    type: Literal["text"] = "text"
    text: str


class AssistantToolCallPart(BaseModel):
    """One ordered assistant function-call part."""

    type: Literal["tool_call"] = "tool_call"
    tool: ResponseFunctionToolCall


AssistantOutputPart: TypeAlias = Annotated[
    AssistantTextPart | AssistantToolCallPart,
    Field(discriminator="type"),
]


class LLMResponseChunk(PipelineMessage):
    """One ordered group of assistant output parts.

    ``text`` and ``tools`` remain as compatibility views for callers that
    still construct the legacy shape. New code can populate ``parts`` to
    represent arbitrary text/tool interleaving without losing order.
    """

    tag: Literal["llm_response_chunk"] = "llm_response_chunk"
    parts: list[AssistantOutputPart] = Field(default_factory=list)
    text: str = ""
    language_code: Optional[str] = None
    tools: list[ResponseFunctionToolCall] = Field(default_factory=list)
    runtime_config: RuntimeConfig | None = None
    response: RealtimeResponseCreateParams | None = None
    turn_id: str | None = None
    turn_revision: int | None = None
    speech_stopped_at_s: float | None = None
    cancel_generation: int | None = None

    @model_validator(mode="after")
    def _normalize_ordered_parts(self) -> "LLMResponseChunk":
        if self.parts:
            self.text = "".join(part.text for part in self.parts if isinstance(part, AssistantTextPart))
            self.tools = [part.tool for part in self.parts if isinstance(part, AssistantToolCallPart)]
        else:
            if self.text:
                self.parts.append(AssistantTextPart(text=self.text))
            self.parts.extend(AssistantToolCallPart(tool=tool) for tool in self.tools)
        return self


class TokenUsage(PipelineMessage):
    """Token count report (side-channel, not forwarded to TTS)."""

    tag: Literal["token_usage"] = "token_usage"
    input_tokens: int
    output_tokens: int
    turn_id: str | None = None
    turn_revision: int | None = None


class EndOfResponse(PipelineMessage):
    """Sentinel marking the end of a response.

    ``error`` is set when generation could not start (e.g. an out-of-band
    response whose ``input`` failed validation); the output processor turns it
    into a ``response.done(status="failed")`` while still closing the response
    normally for pipeline cleanup.
    """

    tag: Literal["end_of_response"] = "end_of_response"
    turn_id: str | None = None
    turn_revision: int | None = None
    cancel_generation: int | None = None
    error: str | None = None


# ── LMOutputProcessor → TTS ──────────────────────────────────────────


class TTSInput(PipelineMessage):
    """Text to synthesize with per-response context."""

    tag: Literal["tts_input"] = "tts_input"
    text: str
    language_code: Optional[str] = None
    runtime_config: RuntimeConfig | None = None
    response: RealtimeResponseCreateParams | None = None
    turn_id: str | None = None
    turn_revision: int | None = None
    speech_stopped_at_s: float | None = None
    cancel_generation: int | None = None


class AudioOutput(PipelineMessage):
    """Audio queue item tagged with the response generation that produced it."""

    tag: Literal["audio_output"] = "audio_output"
    audio: bytes | np.ndarray
    cancel_generation: int | None = None


# ── Realtime service → LLM ────────────────────────────────────────────


class GenerateResponseRequest(PipelineMessage):
    """Triggers LLM generation for a realtime session.

    Carries everything the LM handler needs to produce a response so it
    never has to reach back into shared objects.  ``runtime_config``
    holds the per-connection session config *and* the conversation chat;
    ``response`` carries per-response overrides from ``response.create``.
    Downstream handlers resolve each attribute by preferring the
    per-response value over the session default.
    """

    tag: Literal["generate_response"] = "generate_response"
    runtime_config: RuntimeConfig
    response: RealtimeResponseCreateParams | None = None
    language_code: Optional[str] = None
    turn_id: str | None = None
    turn_revision: int | None = None
    speech_stopped_at_s: float | None = None


# ── Binary sentinels (audio/output queue) ─────────────────────────────

AUDIO_RESPONSE_DONE: Final[bytes] = b"__RESPONSE_DONE__"
PIPELINE_END: Final[bytes] = b"END"

PipelineEndSentinel: TypeAlias = Literal[b"END"]
AudioResponseDoneSentinel: TypeAlias = Literal[b"__RESPONSE_DONE__"]
SentinelMessage: TypeAlias = PipelineEndSentinel | AudioResponseDoneSentinel
