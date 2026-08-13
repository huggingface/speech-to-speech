from pydantic import BaseModel


class InputItemState(BaseModel):
    """Lifecycle state for one client-visible input transcription item."""

    transcript_prefix: str | None = ""
    audio_duration_s: float = 0.0
    completed: bool = False
