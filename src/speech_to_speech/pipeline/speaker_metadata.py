"""Speaker activity attached to audio/transcripts, never treated as spoken text."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import Future
from threading import Event

from pydantic import BaseModel, Field


class SpeakerInterval(BaseModel):
    speaker: int = Field(ge=0)
    start: float = Field(ge=0)
    end: float = Field(ge=0)


class SpeakerAttribution(BaseModel):
    # Times are relative to the concatenated VAD audio sent to STT.
    intervals: list[SpeakerInterval] = Field(default_factory=list)
    complete: bool = True
    available: bool = True

    def durations_for_log(self) -> str:
        """Sum detected activity per speaker; simultaneous speakers count separately."""
        durations: dict[int, float] = {}
        for interval in self.intervals:
            durations[interval.speaker] = durations.get(interval.speaker, 0.0) + max(0.0, interval.end - interval.start)
        return ",".join(f"speaker_{speaker}={seconds:.3f}s" for speaker, seconds in sorted(durations.items())) or "none"

    def for_llm(self, transcript: str) -> str:
        if not transcript:
            return transcript
        speakers = sorted({interval.speaker for interval in self.intervals}) if self.available else []
        labels = ",".join(f"speaker_{speaker}" for speaker in speakers) or "unknown"
        complete = str(self.complete and self.available).lower()
        header = f"[speaker={labels}, complete={complete}]"
        # Each turn must retain its meaning when older chat history is evicted.
        explanation = (
            "Speaker IDs are anonymous and stable only within this session. "
            "These labels are metadata, not spoken words; do not read them aloud. "
            "Multiple labels indicate detected speaker activity; individual words are not attributed. "
            "Do not assume who said each part. 'unknown' means no reliable speaker label is available. "
            "complete=false means attribution is incomplete or unavailable; additional speakers may be missing.\n"
        )
        return f"{explanation}{header}\n{transcript}"


class SpeakerSession:
    """Cancellation shared by immutable queued work and pending STT metadata."""

    def __init__(self) -> None:
        self.invalid = Event()


class SpeakerAttributionFuture(Future[SpeakerAttribution]):
    """A final result plus a detached scored-prefix snapshot during tail flush."""

    def __init__(self) -> None:
        super().__init__()
        self.partial: SpeakerAttribution | None = None


class PendingSpeakerAttribution:
    """Per-revision promises resolved after STT; never wait on the response path."""

    def __init__(self, parts: Iterable[tuple[Future[SpeakerAttribution], SpeakerSession, float]]):
        self.parts = tuple(parts)

    def with_prefix(self, prefix: PendingSpeakerAttribution | None, offset: float) -> PendingSpeakerAttribution:
        if prefix is None:
            return self
        return PendingSpeakerAttribution(
            (*prefix.parts, *((future, session, start + offset) for future, session, start in self.parts))
        )

    def resolve(self) -> SpeakerAttribution:
        intervals: list[SpeakerInterval] = []
        complete = True
        available = True
        for future, session, offset in self.parts:
            if session.invalid.is_set():
                complete = available = False
                continue
            if not future.done():
                complete = False
                if not isinstance(future, SpeakerAttributionFuture) or future.partial is None:
                    continue
                result = future.partial
            else:
                result = future.result()
            complete = complete and result.complete
            available = available and result.available
            intervals.extend(
                interval.model_copy(update={"start": interval.start + offset, "end": interval.end + offset})
                for interval in result.intervals
            )
        return SpeakerAttribution(intervals=intervals, complete=complete, available=available)
