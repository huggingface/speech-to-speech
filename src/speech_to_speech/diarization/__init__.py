"""Streaming speaker activity, independent of transcription and turn detection."""

from .streaming import SpeakerSegment, StreamingDiarizer

__all__ = ["SpeakerSegment", "StreamingDiarizer"]
