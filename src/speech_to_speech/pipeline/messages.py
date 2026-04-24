"""Single source of truth for inter-component pipeline messages.

Typed :class:`PipelineMessage` subclasses replace the ad-hoc tuples that
previously flowed between STT, LLM, LMOutputProcessor and TTS stages.
Binary sentinels carried on the audio/output queue are plain ``bytes``
constants.
"""

from __future__ import annotations

from typing import Final, Literal, Optional, TypeAlias

import numpy as np
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from pydantic import BaseModel, ConfigDict, Field

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


# ── STT → TranscriptionNotifier → LLM ────────────────────────────────


class PartialTranscription(PipelineMessage):
    """Live partial transcription (consumed by TranscriptionNotifier, not forwarded to LLM)."""

    tag: Literal["partial_transcription"] = "partial_transcription"
    text: str


class Transcription(PipelineMessage):
    """Final transcription result."""

    tag: Literal["transcription"] = "transcription"
    text: str
    language_code: Optional[str] = None


# ── LLM → LMOutputProcessor ──────────────────────────────────────────


class LLMResponseChunk(PipelineMessage):
    """One sentence/chunk of the LLM response."""

    tag: Literal["llm_response_chunk"] = "llm_response_chunk"
    text: str
    language_code: Optional[str] = None
    tools: list = Field(default_factory=list)
    runtime_config: RuntimeConfig | None = None
    response: RealtimeResponseCreateParams | None = None


class TokenUsage(PipelineMessage):
    """Token count report (side-channel, not forwarded to TTS)."""

    tag: Literal["token_usage"] = "token_usage"
    input_tokens: int
    output_tokens: int


class EndOfResponse(PipelineMessage):
    """Sentinel marking the end of a response."""

    tag: Literal["end_of_response"] = "end_of_response"


# ── LMOutputProcessor → TTS ──────────────────────────────────────────


class TTSInput(PipelineMessage):
    """Text to synthesize with per-response context."""

    tag: Literal["tts_input"] = "tts_input"
    text: str
    language_code: Optional[str] = None
    runtime_config: RuntimeConfig | None = None
    response: RealtimeResponseCreateParams | None = None


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


# ── Binary sentinels (audio/output queue) ─────────────────────────────

AUDIO_RESPONSE_DONE: Final[bytes] = b"__RESPONSE_DONE__"
PIPELINE_END: Final[bytes] = b"END"

PipelineEndSentinel: TypeAlias = Literal[b"END"]
AudioResponseDoneSentinel: TypeAlias = Literal[b"__RESPONSE_DONE__"]
SentinelMessage: TypeAlias = PipelineEndSentinel | AudioResponseDoneSentinel
