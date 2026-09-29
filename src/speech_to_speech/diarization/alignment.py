from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .streaming import SpeakerSegment


@dataclass(frozen=True)
class SpeakerWord:
    text: str
    start: float
    end: float
    speakers: tuple[int, ...]


def align_words(words: Iterable[dict], segments: Iterable[SpeakerSegment]) -> list[SpeakerWord]:
    """Attach every temporally overlapping speaker to timestamped ASR words.

    This is temporal association, not source separation. Multiple speakers means
    ambiguous attribution; an empty tuple means no speaker activity matched.
    Words without complete timestamps are rejected rather than silently lost.
    """
    ordered = sorted(segments, key=lambda segment: segment.start)
    pending = []
    cursor = 0
    previous_start = -float("inf")
    result = []
    for word in words:
        start, end = word["timestamp"]
        if start is None or end is None or not 0 <= start <= end:
            raise ValueError("Word alignment requires finite, complete, nonnegative timestamps")
        if not (float("-inf") < start < float("inf") and float("-inf") < end < float("inf")):
            raise ValueError("Word alignment requires finite timestamps")
        if start < previous_start:
            raise ValueError("Words must be ordered by start time")
        previous_start = start
        while cursor < len(ordered) and ordered[cursor].start < end:
            pending.append(ordered[cursor])
            cursor += 1
        pending = [segment for segment in pending if segment.end > start]
        speakers = tuple(sorted({segment.speaker for segment in pending if segment.start < end}))
        result.append(SpeakerWord(word["text"], float(start), float(end), speakers))
    return result
