"""Single source of truth for inter-component queue message tags.

Tuple-based tags (first element of queue tuples on text/LLM/STT/TTS queues)
are represented by :class:`MessageTag`.  Binary sentinels carried on the
audio/output queue are plain ``bytes`` constants.
"""

from enum import Enum


class MessageTag(str, Enum):
    """Strongly-typed tags used as the first element of queue tuples."""

    ADD_TO_CONTEXT = "__ADD_TO_CONTEXT__"
    FUNCTION_RESULT = "__FUNCTION_RESULT__"
    GENERATE_RESPONSE = "__GENERATE_RESPONSE__"
    TOKEN_USAGE = "__TOKEN_USAGE__"
    END_OF_RESPONSE = "__END_OF_RESPONSE__"
    PARTIAL = "__PARTIAL__"


AUDIO_RESPONSE_DONE: bytes = b"__RESPONSE_DONE__"
PIPELINE_END: bytes = b"END"
