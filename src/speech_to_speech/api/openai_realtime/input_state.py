from typing import Literal

from pydantic import BaseModel


class SpeechToSpeechInputAudioTranscriptionSnapshotEvent(BaseModel):
    """Cumulative, replaceable STT hypothesis snapshot emitted to opted-in clients."""

    type: Literal["speech_to_speech.input_audio_transcription.snapshot"] = (
        "speech_to_speech.input_audio_transcription.snapshot"
    )
    event_id: str
    item_id: str
    content_index: int = 0
    transcript: str


class InputItemState(BaseModel):
    """Active state for one client-visible input transcription item."""

    transcript_prefix: str = ""
    latest_transcript: str = ""
    audio_duration_s: float = 0.0
