"""Opt-in readable provider reasoning; encrypted reasoning is never published."""

from typing import Literal

from pydantic import BaseModel


class SpeechToSpeechReasoningDeltaEvent(BaseModel):
    type: Literal["speech_to_speech.reasoning.delta"] = "speech_to_speech.reasoning.delta"
    event_id: str
    response_id: str
    delta: str
    kind: Literal["reasoning", "summary"] = "reasoning"
