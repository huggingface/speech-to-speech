"""Single source of truth for inter-component pipeline messages.

Typed :class:`PipelineMessage` subclasses replace the ad-hoc tuples that
previously flowed between STT, LLM, LMOutputProcessor and TTS stages.
Binary sentinels carried on the audio/output queue are plain ``bytes``
constants.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from threading import Event, Lock
from time import perf_counter
from typing import Annotated, Any, Final, Literal, Optional, TypeAlias
from uuid import uuid4

import numpy as np
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from pydantic import BaseModel, ConfigDict, Field, model_validator

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

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
    runtime_config: RuntimeConfig | None = None
    mode: Literal["progressive", "final"] | None = None
    turn_id: str | None = None
    turn_revision: int | None = None
    processing_delay_s: float = 0.0
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


class TranscriptionFailure(PipelineMessage):
    """A final STT operation failed without producing LLM input."""

    tag: Literal["transcription_failure"] = "transcription_failure"
    message: str
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


def _normalize_assistant_output_fields(
    parts: list[AssistantOutputPart],
    text: str,
    tools: list[ResponseFunctionToolCall],
    fields_set: set[str],
) -> tuple[str, list[ResponseFunctionToolCall]] | None:
    """Keep ordered parts and their legacy compatibility views consistent."""

    if parts or "parts" in fields_set:
        derived_text = "".join(part.text for part in parts if isinstance(part, AssistantTextPart))
        derived_tools = [part.tool for part in parts if isinstance(part, AssistantToolCallPart)]
        if "text" in fields_set and text != derived_text:
            raise ValueError("text must match the ordered parts")
        if "tools" in fields_set and tools != derived_tools:
            raise ValueError("tools must match the ordered parts")
        return derived_text, derived_tools

    if text:
        parts.append(AssistantTextPart(text=text))
    parts.extend(AssistantToolCallPart(tool=tool) for tool in tools)
    return None


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
    response_key: str | None = Field(default=None, exclude=True, repr=False)
    prefetch_transaction: Any = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def _normalize_ordered_parts(self) -> "LLMResponseChunk":
        legacy_views = _normalize_assistant_output_fields(self.parts, self.text, self.tools, self.model_fields_set)
        if legacy_views is not None:
            self.text, self.tools = legacy_views
        return self


class TokenUsage(PipelineMessage):
    """Token count report (side-channel, not forwarded to TTS)."""

    tag: Literal["token_usage"] = "token_usage"
    input_tokens: int
    output_tokens: int
    turn_id: str | None = None
    turn_revision: int | None = None
    cancel_generation: int | None = None
    response_key: str | None = Field(default=None, exclude=True, repr=False)


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
    response_key: str | None = Field(default=None, exclude=True, repr=False)
    error: str | None = None
    cleanup_only: bool = False


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
    response_key: str | None = Field(default=None, exclude=True, repr=False)
    prefetch_transaction: Any = Field(default=None, exclude=True, repr=False)


class AudioOutput(PipelineMessage):
    """Audio queue item tagged with the response generation that produced it."""

    tag: Literal["audio_output"] = "audio_output"
    audio: bytes | np.ndarray
    cancel_generation: int | None = None
    response_key: str | None = Field(default=None, exclude=True, repr=False)
    cleanup_only: bool = False


# ── Realtime service → LLM ────────────────────────────────────────────


class ResponsePrefetchTransaction:
    """Commit irreversible chat cleanup only after a prefetch is claimed.

    Generated assistant items are already provisional and can be rolled back
    by response key. Image stripping and history trimming are not reversible,
    so an unclaimed prefetch parks those operations here until the matching
    standard ``response.create`` arrives.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._cleanup: Callable[[], None] | None = None
        self._abort_callbacks: list[Callable[[], None]] = []
        self._claimed = False
        self._discarded = False
        self._resolved = Event()

    def register_abort(self, abort: Callable[[], None]) -> None:
        """Abort immediately if discarded, otherwise arm an active prefetch."""
        run_abort = False
        with self._lock:
            if self._discarded:
                run_abort = True
            elif not self._claimed:
                self._abort_callbacks.append(abort)
        if run_abort:
            self._run_abort(abort)

    @staticmethod
    def _run_abort(abort: Callable[[], None]) -> None:
        try:
            abort()
        except Exception:
            # Cancellation must still reach response rollback and tombstoning
            # when a provider or backend raises while releasing its stream.
            logger.exception("Failed to abort discarded response prefetch")

    @property
    def discarded(self) -> bool:
        """Whether this speculative request has been invalidated."""
        with self._lock:
            return self._discarded

    @property
    def claimed(self) -> bool:
        with self._lock:
            return self._claimed

    def wait_until_resolved(self, timeout: float) -> bool:
        """Wait until public claim or internal discard decides this prefetch."""
        return self._resolved.wait(timeout)

    def complete(self, cleanup: Callable[[], None]) -> None:
        """Store cleanup, or run it immediately when already claimed."""
        run_cleanup = False
        with self._lock:
            if self._discarded:
                return
            self._abort_callbacks = []
            if self._claimed:
                run_cleanup = True
            else:
                self._cleanup = cleanup
        if run_cleanup:
            cleanup()

    def claim(self) -> bool:
        """Commit cleanup or return false when speculative work already failed."""
        cleanup: Callable[[], None] | None
        with self._lock:
            if self._discarded:
                return False
            if self._claimed:
                return True
            self._claimed = True
            self._abort_callbacks = []
            cleanup = self._cleanup
            self._cleanup = None
        if cleanup is not None:
            try:
                cleanup()
            except Exception:
                with self._lock:
                    self._claimed = False
                    self._discarded = True
                self._resolved.set()
                raise
        self._resolved.set()
        return True

    def discard(self) -> None:
        """Prevent cleanup from mutating chat after speculation is invalidated."""
        abort_callbacks: list[Callable[[], None]]
        with self._lock:
            if self._discarded or self._claimed:
                return
            self._discarded = True
            self._cleanup = None
            abort_callbacks = self._abort_callbacks
            self._abort_callbacks = []
            self._resolved.set()
        for abort in abort_callbacks:
            self._run_abort(abort)


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
    response_key: str = Field(default_factory=lambda: uuid4().hex, exclude=True, repr=False)
    runtime_config: RuntimeConfig
    response: RealtimeResponseCreateParams | None = None
    audio: np.ndarray | None = None
    audio_sample_rate: int = 16000
    language_code: Optional[str] = None
    turn_id: str | None = None
    turn_revision: int | None = None
    speech_stopped_at_s: float | None = None
    prefetch_transaction: ResponsePrefetchTransaction | None = Field(default=None, exclude=True, repr=False)


# ── Binary sentinels (audio/output queue) ─────────────────────────────

AUDIO_RESPONSE_DONE: Final[bytes] = b"__RESPONSE_DONE__"
PIPELINE_END: Final[bytes] = b"END"

PipelineEndSentinel: TypeAlias = Literal[b"END"]
AudioResponseDoneSentinel: TypeAlias = Literal[b"__RESPONSE_DONE__"]
SentinelMessage: TypeAlias = PipelineEndSentinel | AudioResponseDoneSentinel
