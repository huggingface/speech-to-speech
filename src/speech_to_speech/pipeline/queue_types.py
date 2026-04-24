"""Typed aliases for items flowing through pipeline queues.

This module centralizes queue payload unions so the rest of the codebase can
reference a small set of stable types instead of repeating large unions.
"""

from __future__ import annotations

# ruff: noqa: I001

from typing import TypeAlias

import numpy as np

from speech_to_speech.pipeline.control import PipelineControlMessage
from speech_to_speech.pipeline.events import PipelineEvent
from speech_to_speech.pipeline.handler_types import LLMIn, LLMOut, STTOut, TTSIn, VADIn, VADOut

# Use plain ``bytes`` for sentinel values on queues (``PIPELINE_END``, ``AUDIO_RESPONSE_DONE``).
# ``Queue`` is invariant; ``Literal[b"END"]`` is not accepted where ``bytes`` is required.
PipelineInternalItem: TypeAlias = PipelineControlMessage | bytes

# Audio chunks coming from IO (socket/websocket/mic) into VAD.
AudioInItem: TypeAlias = VADIn | PipelineControlMessage

# Audio segments flowing from VAD to STT.
VADOutItem: TypeAlias = VADOut | PipelineInternalItem

# STT output (partial + final) flowing to TranscriptionNotifier.
STTOutItem: TypeAlias = STTOut | PipelineInternalItem

# Text prompts flowing into LLM.
TextPromptItem: TypeAlias = LLMIn | PipelineInternalItem

# LLM outputs flowing into LMOutputProcessor.
LMOutItem: TypeAlias = LLMOut | PipelineInternalItem

# Inputs flowing into TTS.
TTSInItem: TypeAlias = TTSIn | PipelineInternalItem

# Audio outputs flowing to speakers / client (includes sentinels as bytes).
AudioOutItem: TypeAlias = bytes | np.ndarray | PipelineControlMessage

# Side-channel text events sent to websocket/realtime clients.
TextEventItem: TypeAlias = PipelineEvent | PipelineInternalItem
