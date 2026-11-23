"""Reasoning on the Realtime wire: the effort clients select and opt-in readable text.

Encrypted reasoning is never published.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from openai.types.realtime import (
    RealtimeSessionCreateRequest,
    ResponseCreateEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    SessionUpdateEvent,
)
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from openai.types.realtime.realtime_transcription_session_create_request import (
    RealtimeTranscriptionSessionCreateRequest,
)
from pydantic import BaseModel

# The SDK types ``reasoning.effort`` as minimal..xhigh. This server also takes
# ``none``, which the SDK's models refuse to parse and warn about when dumped.
# The models below leave ``reasoning`` untyped, so it stays the plain object the
# client sent, and are used wherever a session or a response config crosses the
# wire. ``runtime_config.validate_reasoning`` checks the value instead.


class SessionRequest(RealtimeSessionCreateRequest):
    reasoning: Any = None

    @classmethod
    def of(cls, session: RealtimeSessionCreateRequest) -> SessionRequest:
        """Return ``session`` as this class, keeping which fields the client set."""
        if isinstance(session, cls):
            return session
        return cls.model_construct(_fields_set=set(session.model_fields_set), **dict(session))


_Session = Union[SessionRequest, RealtimeTranscriptionSessionCreateRequest]


class SessionUpdate(SessionUpdateEvent):
    session: _Session


class SessionCreated(SessionCreatedEvent):
    session: _Session


class SessionUpdated(SessionUpdatedEvent):
    session: _Session


class ResponseParams(RealtimeResponseCreateParams):
    reasoning: Any = None


class ResponseCreate(ResponseCreateEvent):
    response: Optional[ResponseParams] = None


class SpeechToSpeechReasoningDeltaEvent(BaseModel):
    type: Literal["speech_to_speech.reasoning.delta"] = "speech_to_speech.reasoning.delta"
    event_id: str
    response_id: str
    delta: str
    kind: Literal["reasoning", "summary"] = "reasoning"
