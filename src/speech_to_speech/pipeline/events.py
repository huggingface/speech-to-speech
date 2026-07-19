"""Typed events flowing on ``text_output_queue``.

These are internal pipeline events produced by VAD, TranscriptionNotifier, and
LMOutputProcessor, consumed by the realtime ``WebSocketRouter`` send-loop and
``RealtimeService.dispatch_pipeline_event``.  They replace the raw ``dict``
literals that were previously put on the queue.
"""

from __future__ import annotations

from typing import Literal, Optional

from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from pydantic import BaseModel, Field


class PipelineEvent(BaseModel):
    """Base for all text_output_queue events.

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


# ── LLM output events (LMOutputProcessor) ────────────────────────────


class AssistantTextEvent(PipelineEvent):
    type: Literal["assistant_text"] = "assistant_text"
    text: str
    tools: list[ResponseFunctionToolCall] = Field(default_factory=list)
    turn_id: str | None = None
    turn_revision: int | None = None
    # Response generation that produced this text, mirroring AudioOutput. Lets the
    # send loop discard stale assistant text by the same generation-aware rule as
    # audio, instead of blanket-dropping while cancel_scope.discarding is set.
    cancel_generation: int | None = None


class TokenUsageEvent(PipelineEvent):
    type: Literal["token_usage"] = "token_usage"
    input_tokens: int = 0
    output_tokens: int = 0
    turn_id: str | None = None
    turn_revision: int | None = None


# ── s2mlt translation events ─────────────────────────────────────────
#
# Events for the speech → multi-language-translation-text pipeline
# (``s2mlt.py``). One VAD turn == one translated segment; clients key on
# ``turn_id`` and treat a higher ``turn_revision`` (a reopened segment) or a
# newer event for the same key as an upsert of that segment.


class InputTranscriptionDeltaEvent(PipelineEvent):
    """Live transcription snapshot for a segment.

    ``text`` is the full transcription of the segment so far (a snapshot, not
    an append-only delta); the client replaces the segment's text with it.
    """

    type: Literal["input.transcription.delta"] = "input.transcription.delta"
    text: str
    turn_id: str | None = None
    turn_revision: int | None = None


class InputTranscriptionDoneEvent(PipelineEvent):
    """Final transcription for a segment (may be empty for discarded noise)."""

    type: Literal["input.transcription.done"] = "input.transcription.done"
    text: str
    language_code: Optional[str] = None
    turn_id: str | None = None
    turn_revision: int | None = None


class TranslationDeltaEvent(PipelineEvent):
    """Streaming snapshot of the structured translation for a segment.

    ``translations`` maps target language code → translation-so-far;
    ``corrected`` is the cleaned-up input transcript. Fields fill in as the
    model streams; each event supersedes the previous one for the segment.
    """

    type: Literal["translation.delta"] = "translation.delta"
    corrected: str = ""
    translations: dict[str, str] = Field(default_factory=dict)
    turn_id: str | None = None
    turn_revision: int | None = None


class TranslationDoneEvent(PipelineEvent):
    """Final structured translation for a segment.

    ``error`` is set (with empty payload fields) when generation failed or the
    model output could not be parsed.
    """

    type: Literal["translation.done"] = "translation.done"
    corrected: str = ""
    translations: dict[str, str] = Field(default_factory=dict)
    turn_id: str | None = None
    turn_revision: int | None = None
    error: str | None = None


class ResponseFailedEvent(PipelineEvent):
    """Signals that a response could not be generated (e.g. invalid out-of-band input).

    Dispatched to the service so it can close the in-progress response with
    ``status="failed"`` instead of the usual ``completed``.
    """

    type: Literal["response_failed"] = "response_failed"
    message: str = ""
    turn_id: str | None = None
    turn_revision: int | None = None
