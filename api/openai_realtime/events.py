"""Typed events flowing on ``text_output_queue``.

These are internal pipeline events produced by VAD, TranscriptionNotifier, and
LMOutputProcessor, consumed by the realtime ``WebSocketRouter`` send-loop and
``RealtimeService.dispatch_pipeline_event``.  They replace the raw ``dict``
literals that were previously put on the queue.
"""

from __future__ import annotations

from typing import Literal

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


class SpeechStoppedEvent(PipelineEvent):
    type: Literal["speech_stopped"] = "speech_stopped"
    duration_s: float = 0.0
    audio_end_ms: int = 0


# ── Transcription events (TranscriptionNotifier) ─────────────────────

class PartialTranscriptionEvent(PipelineEvent):
    type: Literal["partial_transcription"] = "partial_transcription"
    delta: str


class TranscriptionCompletedEvent(PipelineEvent):
    type: Literal["transcription_completed"] = "transcription_completed"
    transcript: str
    language_code: str | None = None


# ── LLM output events (LMOutputProcessor) ────────────────────────────

class AssistantTextEvent(PipelineEvent):
    type: Literal["assistant_text"] = "assistant_text"
    text: str
    tools: list[dict] = Field(default_factory=list)


class TokenUsageEvent(PipelineEvent):
    type: Literal["token_usage"] = "token_usage"
    input_tokens: int = 0
    output_tokens: int = 0
