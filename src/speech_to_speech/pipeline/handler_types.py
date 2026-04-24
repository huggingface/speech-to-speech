"""Convenience `TypeAlias` definitions for pipeline handler generics.

These aliases keep handler declarations readable (e.g. `BaseHandler[STTIn, STTOut]`)
without hiding what actually flows between stages (see `pipeline/messages.py` and
`pipeline/queue_types.py` for the full story, including control + sentinel items).
"""

from __future__ import annotations

# ruff: noqa: I001

from typing import TypeAlias

import numpy as np

from speech_to_speech.pipeline.messages import (
    EndOfResponse,
    GenerateResponseRequest,
    LLMResponseChunk,
    PartialTranscription,
    TTSInput,
    TokenUsage,
    Transcription,
    VADAudio,
)

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig

# ── VAD stage ─────────────────────────────────────────────────────────
VADIn: TypeAlias = bytes | tuple[bytes, RuntimeConfig]
VADOut: TypeAlias = VADAudio

# ── STT stage ─────────────────────────────────────────────────────────
STTIn: TypeAlias = VADAudio
STTOut: TypeAlias = PartialTranscription | Transcription

# ── LLM stage ─────────────────────────────────────────────────────────
LLMIn: TypeAlias = Transcription | GenerateResponseRequest
LLMOut: TypeAlias = LLMResponseChunk | TokenUsage | EndOfResponse

# ── TTS stage ─────────────────────────────────────────────────────────
TTSIn: TypeAlias = TTSInput | EndOfResponse
TTSOut: TypeAlias = bytes | np.ndarray
