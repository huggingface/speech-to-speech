from typing import Union

from openai.types.realtime import (
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionFailedEvent,
    InputAudioBufferSpeechStoppedEvent,
)
from pydantic import BaseModel

InputTranscriptionTerminal = Union[
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionFailedEvent,
]


class InputItemState(BaseModel):
    """Active state for one client-visible input transcription item."""

    transcript_prefix: str = ""
    latest_transcript: str = ""
    audio_duration_s: float = 0.0


class PendingInputTerminal(BaseModel):
    """One input item's terminal lifecycle, held until its turn commits.

    A speculative pause is an internal stop candidate: speech can resume and
    the same logical turn stays open. Realtime has no way to retract a
    ``speech_stopped`` or an already completed transcription, so both are kept
    here until the turn is externally committed. A revision that a later
    revision supersedes discards its held events instead of publishing them.
    """

    turn_id: str | None = None
    turn_revision: int | None = None
    speech_stopped: InputAudioBufferSpeechStoppedEvent | None = None
    transcription: InputTranscriptionTerminal | None = None
    # Set once the pipeline has finished with this item, so publishing it also
    # releases its routing and transcript state.
    input_closed: bool = False
