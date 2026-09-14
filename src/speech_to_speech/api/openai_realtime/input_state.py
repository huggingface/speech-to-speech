from pydantic import BaseModel


class InputItemState(BaseModel):
    """Active state for one client-visible input transcription item."""

    transcript_prefix: str = ""
    latest_transcript: str = ""
    audio_duration_s: float = 0.0
