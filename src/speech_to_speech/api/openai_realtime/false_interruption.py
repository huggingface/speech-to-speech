"""Server-side PCM audio replay for false-interruption recovery."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from speech_to_speech.pipeline.history import ResponseHistory
from speech_to_speech.pipeline.messages import AssistantTextPart, AssistantToolCallPart

PIPELINE_SAMPLE_RATE = 16_000
RESUME_REWIND_MAX_MS = 600
RESUME_FADE_IN_MS = 10
# Retain at most two minutes of 16 kHz PCM16 (~3.8 MB) per active reply.
MAX_RETAINED_AUDIO_SAMPLES = PIPELINE_SAMPLE_RATE * 120

_REWIND_WINDOW_MS = 20
_REWIND_STEP_MS = 10
_QUIET_RMS_THRESHOLD = 150.0
# Individual CJK characters or Unicode letter/digit words (internal apostrophes/hyphens kept).
_WORD_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[^\W_]+(?:['’-][^\W_]+)*")


def count_interruption_words(text: str) -> int:
    """Count spoken words across space-delimited and CJK scripts."""
    return len(_WORD_RE.findall(text)) if text else 0


def resume_rewind_samples(
    pcm: bytes | bytearray,
    cut_samples: int,
    *,
    sample_rate: int = PIPELINE_SAMPLE_RATE,
    max_rewind_ms: int = RESUME_REWIND_MAX_MS,
) -> int:
    """Rewind up to *max_rewind_ms* before *cut_samples* to a quiet word gap."""
    cut = min(max(0, cut_samples), len(pcm) // 2)
    max_rewind = max(0, int(max_rewind_ms * sample_rate / 1000))
    if cut <= 0 or max_rewind == 0:
        return cut
    if cut <= max_rewind:
        return 0
    search_start = cut - max_rewind
    win = max(1, int(_REWIND_WINDOW_MS * sample_rate / 1000))
    step = max(1, int(_REWIND_STEP_MS * sample_rate / 1000))
    region = np.frombuffer(pcm, dtype=np.int16, count=cut - search_start, offset=search_start * 2).astype(np.float32)
    if region.size < win:
        return search_start
    frames = np.lib.stride_tricks.sliding_window_view(region, win)[::step]
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    quiet = np.flatnonzero(rms <= _QUIET_RMS_THRESHOLD)
    idx = int(quiet[-1]) if quiet.size else int(np.argmin(rms))
    return search_start + idx * step + win // 2


def apply_fade_in(pcm: bytes, *, sample_rate: int = PIPELINE_SAMPLE_RATE, fade_ms: int = RESUME_FADE_IN_MS) -> bytes:
    """Apply a short linear fade-in to PCM16 mono audio to avoid splice clicks."""
    fade = min(len(pcm) // 2, max(0, int(fade_ms * sample_rate / 1000)))
    if fade <= 1:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).copy()
    ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
    samples[:fade] = np.round(samples[:fade].astype(np.float32) * ramp).astype(np.int16)
    return samples.tobytes()


def _unheard_text_suffix(text: str, start_in_seg: int, seg_duration: int) -> str:
    if not text or start_in_seg <= 0 or seg_duration <= 0:
        return text
    if start_in_seg >= seg_duration:
        return ""
    budget = len(text) * start_in_seg / seg_duration
    used = 0
    words = text.split()
    for i, word in enumerate(words):
        used += len(word) + (1 if i else 0)
        if used > budget:
            return " ".join(words[i:])
    return ""


@dataclass
class ReplaySegment:
    text: str
    start_sample: int
    end_sample: int


@dataclass
class AssistantAudioBuffer:
    """Retained 16 kHz PCM16 audio and aligned text for the latest assistant reply."""

    response_key: str | None
    root_item_id: str
    current_item_id: str
    wire_start_sample: int = 0
    resume_source: tuple[str, int] | None = None
    turn_id: str | None = None
    turn_revision: int | None = None
    cancel_generation: int | None = None
    segments: list[ReplaySegment] = field(default_factory=list)
    pcm: bytearray = field(default_factory=bytearray)
    pcm_base_sample: int = 0
    generation_done: bool = False
    had_tool_call: bool = False
    next_output_sequence: int = 0
    done_status: Literal["completed", "incomplete", "failed"] = "completed"
    done_reason: Literal["max_output_tokens", "content_filter"] | None = None
    final_history_proposal: ResponseHistory | None = None
    buffered_tool_events: list[Any] = field(default_factory=list)

    @property
    def total_samples(self) -> int:
        return self.pcm_base_sample + len(self.pcm) // 2

    def append_text(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if self.segments and self.segments[-1].start_sample == self.segments[-1].end_sample:
            self.segments[-1].text = f"{self.segments[-1].text} {text}".strip()
        else:
            pos = self.total_samples
            self.segments.append(ReplaySegment(text=text, start_sample=pos, end_sample=pos))

    def record_output_parts(self, parts: Sequence[Any]) -> bool:
        """Append text parts and return True if any tool-call part is present."""
        saw_tool = False
        for part in parts:
            if isinstance(part, AssistantToolCallPart):
                self.had_tool_call = saw_tool = True
            elif isinstance(part, AssistantTextPart):
                self.append_text(part.text)
        return saw_tool

    def append_pcm(self, chunk: bytes) -> None:
        if not chunk:
            return
        if not self.segments:
            pos = self.total_samples
            self.segments.append(ReplaySegment(text="", start_sample=pos, end_sample=pos))
        self.pcm.extend(chunk)
        self.segments[-1].end_sample = self.total_samples
        excess = len(self.pcm) // 2 - MAX_RETAINED_AUDIO_SAMPLES
        if excess > 0:
            del self.pcm[: excess * 2]
            self.pcm_base_sample += excess

    def replace_history_proposal(self, proposal: ResponseHistory | None) -> None:
        if proposal is None or proposal is self.final_history_proposal:
            return
        prev = self.final_history_proposal
        if prev is not None and (len(proposal.items), proposal.complete) < (len(prev.items), prev.complete):
            proposal.resolve()
            return
        if prev is not None:
            prev.resolve()
        self.final_history_proposal = proposal

    def resolve_history_proposal(self) -> None:
        if self.final_history_proposal is not None:
            self.final_history_proposal.resolve()
            self.final_history_proposal = None

    def _slice_pcm(self, start_sample: int, end_sample: int) -> bytes:
        lo = max(self.pcm_base_sample, start_sample)
        hi = max(lo, min(self.total_samples, end_sample))
        return b"" if hi <= lo else bytes(self.pcm[(lo - self.pcm_base_sample) * 2 : (hi - self.pcm_base_sample) * 2])

    def remaining_segments_from(self, resume_sample: int) -> list[tuple[str, bytes]]:
        """Return ordered ``(text, pcm)`` pairs from *resume_sample* onward."""
        start = max(self.pcm_base_sample, min(resume_sample, self.total_samples))
        output: list[tuple[str, bytes]] = []
        faded = False
        for seg in self.segments:
            seg_len = seg.end_sample - seg.start_sample
            if seg_len == 0:
                if seg.start_sample >= start and seg.text:
                    output.append((seg.text, b""))
                continue
            if seg.end_sample <= start:
                continue
            text = (
                _unheard_text_suffix(seg.text, start - seg.start_sample, seg_len)
                if seg.start_sample < start
                else seg.text
            )
            pcm = self._slice_pcm(max(start, seg.start_sample), seg.end_sample)
            if pcm and not faded and start > 0:
                pcm, faded = apply_fade_in(pcm), True
            if text or pcm:
                output.append((text, pcm))
        return output

    def words_in_range(self, start_sample: int, end_sample: int) -> int:
        """Count words from segments falling within ``[start_sample, end_sample)``."""
        start = max(self.pcm_base_sample, min(start_sample, self.total_samples))
        end = max(start, min(end_sample, self.total_samples))
        if end <= start:
            return 0
        words = 0
        for seg in self.segments:
            seg_len = seg.end_sample - seg.start_sample
            if seg_len <= 0 or seg.end_sample <= start or seg.start_sample >= end:
                continue
            lo = max(start, seg.start_sample)
            hi = min(end, seg.end_sample)
            from_lo = _unheard_text_suffix(seg.text, lo - seg.start_sample, seg_len).split()
            from_hi = (
                _unheard_text_suffix(seg.text, hi - seg.start_sample, seg_len).split() if hi < seg.end_sample else []
            )
            words += max(0, len(from_lo) - len(from_hi))
        return words

    def full_text(self) -> str:
        return " ".join(seg.text for seg in self.segments if seg.text)


@dataclass
class PendingFalseInterruption:
    """In-flight barge-in awaiting STT confirmation before discarding or replaying."""

    interrupt_turn_id: str | None
    audio_buffer: AssistantAudioBuffer
    was_generating: bool
    estimated_played_samples: int
    timeout_s: float
    deadline: float
    speech_stopped: bool = False
    client_played_samples: int | None = None
    short_transcript_pending: bool = False

    def cut_samples(self) -> int:
        chosen = self.client_played_samples if self.client_played_samples is not None else self.estimated_played_samples
        return min(max(self.audio_buffer.pcm_base_sample, chosen), self.audio_buffer.total_samples)
