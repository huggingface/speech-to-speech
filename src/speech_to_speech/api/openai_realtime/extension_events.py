"""Project-specific Realtime server events.

These optional events improve the bundled demo without changing the OpenAI
Realtime compatibility contract. Unknown clients can ignore them; confirmed
turns and cancellation still use the standard
``input_audio_buffer.speech_started`` event.
"""

from typing import Literal

from openai.types.realtime import InputAudioBufferSpeechStartedEvent
from pydantic import BaseModel


class InputAudioBufferSpeechCandidateStartedEvent(BaseModel):
    type: Literal["input_audio_buffer.speech_candidate_started"] = "input_audio_buffer.speech_candidate_started"
    event_id: str
    candidate_id: str
    audio_start_ms: int
    interrupt_response: bool = True


class InputAudioBufferSpeechCandidateRejectedEvent(BaseModel):
    type: Literal["input_audio_buffer.speech_candidate_rejected"] = "input_audio_buffer.speech_candidate_rejected"
    event_id: str
    candidate_id: str


class ExtendedInputAudioBufferSpeechStartedEvent(InputAudioBufferSpeechStartedEvent):
    """Standard speech-start event with the project's interrupt decision."""

    interrupt_response: bool = True
