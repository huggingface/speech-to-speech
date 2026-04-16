"""Single source of truth for inter-component queue message tags.

Tuple-based tags (first element of queue tuples on text/LLM/STT/TTS queues)
are represented by :class:`MessageTag`.  Binary sentinels carried on the
audio/output queue are plain ``bytes`` constants.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict

from LLM.chat import Chat


class MessageTag(str, Enum):
    """Strongly-typed tags used as the first element of queue tuples."""

    ADD_TO_CONTEXT = "__ADD_TO_CONTEXT__"
    FUNCTION_RESULT = "__FUNCTION_RESULT__"
    GENERATE_RESPONSE = "__GENERATE_RESPONSE__"
    TOKEN_USAGE = "__TOKEN_USAGE__"
    END_OF_RESPONSE = "__END_OF_RESPONSE__"
    PARTIAL = "__PARTIAL__"


class GenerateRequest(BaseModel):
    """Payload for ``GENERATE_RESPONSE`` queue messages.

    Carries everything the LM handler needs to produce a response so it
    never has to reach back into shared objects.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    chat: Chat
    instructions: str | None = None
    tools: list | None = None
    tool_choice: str | None = None
    override_instructions: str | None = None
    language_code: str | None = None


AUDIO_RESPONSE_DONE: bytes = b"__RESPONSE_DONE__"
PIPELINE_END: bytes = b"END"
