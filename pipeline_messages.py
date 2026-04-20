"""Single source of truth for inter-component queue message tags.

Tuple-based tags (first element of queue tuples on text/LLM/STT/TTS queues)
are represented by :class:`MessageTag`.  Binary sentinels carried on the
audio/output queue are plain ``bytes`` constants.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from api.openai_realtime.runtime_config import RuntimeConfig
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams


class MessageTag(str, Enum):
    """Strongly-typed tags used as the first element of queue tuples."""

    ADD_TO_CONTEXT = "__ADD_TO_CONTEXT__"
    FUNCTION_RESULT = "__FUNCTION_RESULT__"
    GENERATE_RESPONSE = "__GENERATE_RESPONSE__"
    TOKEN_USAGE = "__TOKEN_USAGE__"
    END_OF_RESPONSE = "__END_OF_RESPONSE__"
    PARTIAL = "__PARTIAL__"


class GenerateResponseRequest(BaseModel):
    """Payload for ``GENERATE_RESPONSE`` queue messages.

    Carries everything the LM handler needs to produce a response so it
    never has to reach back into shared objects.  ``runtime_config``
    holds the per-connection session config *and* the conversation chat;
    ``response`` carries per-response overrides from ``response.create``.
    Downstream handlers resolve each attribute by preferring the
    per-response value over the session default.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    runtime_config: RuntimeConfig
    response: RealtimeResponseCreateParams | None = None
    language_code: str | None = None


AUDIO_RESPONSE_DONE: bytes = b"__RESPONSE_DONE__"
PIPELINE_END: bytes = b"END"
