"""Server-side audio replay for false-interruption recovery.

When ``min_interruption_words > 0``, the server retains the 16 kHz PCM16 audio
and aligned transcript segments of the current assistant reply. On barge-in,
the client still receives standard OpenAI Realtime cancellation events right
away so playback stops immediately. If the interrupting speech resolves to
fewer than ``min_interruption_words`` words (for example an empty transcript
from a cough or an STT failure) before ``false_interruption_timeout_ms``, the
server replays the unplayed audio — rewound up to 600 ms to a quiet word gap —
as a standard Realtime response and continues streaming any remaining live
LLM/TTS output for long turns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from speech_to_speech.pipeline.history import ResponseHistory

PIPELINE_SAMPLE_RATE = 16_000
RESUME_REWIND_MAX_MS = 600
RESUME_FADE_IN_MS = 10
# Retain at most two minutes of 16 kHz PCM16 (~3.8 MB) per active reply.
MAX_RETAINED_AUDIO_SAMPLES = PIPELINE_SAMPLE_RATE * 120

_REWIND_WINDOW_MS = 20
_REWIND_STEP_MS = 10
_QUIET_RMS_THRESHOLD = 150.0

# Count each CJK character individually (scripts without inter-word spaces) and
# Unicode letter/digit runs (preserving internal apostrophes/hyphens) for all
# other scripts. Punctuation-only transcripts count as 0 words.
_WORD_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[^\W_]+(?:['’-][^\W_]+)*")


def count_interruption_words(text: str) -> int:
    """Count spoken words in *text* across space-delimited and CJK scripts."""
    if not text:
        return 0
    return len(_WORD_RE.findall(text))


def resume_rewind_samples(
    pcm: bytes | bytearray,
    cut_samples: int,
    *,
    sample_rate: int = PIPELINE_SAMPLE_RATE,
    max_rewind_ms: int = RESUME_REWIND_MAX_MS,
) -> int:
    """Choose a clean replay start sample up to *max_rewind_ms* before *cut_samples*.

    If *cut_samples* falls within *max_rewind_ms* of the start of the buffer,
    rewind all the way to sample ``0`` so the opening word is not clipped.
    Otherwise, scan the preceding window for the latest quiet gap (or the
    lowest-energy dip if speech is continuous) so replay resumes on a word
    boundary instead of mid-syllable.
    """
    total_samples = len(pcm) // 2
    cut = min(max(0, cut_samples), total_samples)
    max_rewind = max(0, int(max_rewind_ms * sample_rate / 1000))
    if cut <= 0 or max_rewind == 0:
        return cut
    if cut <= max_rewind:
        return 0

    search_start = cut - max_rewind
    window_samples = max(1, int(_REWIND_WINDOW_MS * sample_rate / 1000))
    step_samples = max(1, int(_REWIND_STEP_MS * sample_rate / 1000))
    region = np.frombuffer(pcm, dtype=np.int16, count=cut - search_start, offset=search_start * 2).astype(np.float32)
    if region.size < window_samples:
        return search_start

    last_quiet_pos: int | None = None
    best_rms = float("inf")
    best_pos = search_start
    for offset in range(0, region.size - window_samples + 1, step_samples):
        frame = region[offset : offset + window_samples]
        rms = float(np.sqrt(np.mean(frame * frame)))
        midpoint = search_start + offset + window_samples // 2
        if rms <= _QUIET_RMS_THRESHOLD:
            last_quiet_pos = midpoint
        if rms < best_rms:
            best_rms = rms
            best_pos = midpoint

    return last_quiet_pos if last_quiet_pos is not None else best_pos


def apply_fade_in(
    pcm: bytes,
    *,
    sample_rate: int = PIPELINE_SAMPLE_RATE,
    fade_ms: int = RESUME_FADE_IN_MS,
) -> bytes:
    """Apply a short linear fade-in to PCM16 mono audio to avoid splice clicks."""
    fade_samples = min(len(pcm) // 2, max(0, int(fade_ms * sample_rate / 1000)))
    if fade_samples <= 1:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).copy()
    ramp = np.linspace(0.0, 1.0, fade_samples, endpoint=False, dtype=np.float32)
    samples[:fade_samples] = np.round(samples[:fade_samples].astype(np.float32) * ramp).astype(np.int16)
    return samples.tobytes()


def _unheard_text_suffix(text: str, start_in_seg: int, seg_duration: int) -> str:
    """Return the suffix of *text* starting proportionally at *start_in_seg*."""
    if not text or start_in_seg <= 0 or seg_duration <= 0:
        return text
    if start_in_seg >= seg_duration:
        return ""
    budget = len(text) * start_in_seg / seg_duration
    words = text.split()
    used = 0
    keep_from = 0
    for index, word in enumerate(words):
        next_used = used + len(word) + (1 if index > 0 else 0)
        if next_used > budget:
            keep_from = index
            break
        used = next_used
    else:
        return ""
    return " ".join(words[keep_from:])


@dataclass
class ReplaySegment:
    """One transcript chunk aligned to pipeline-rate PCM sample offsets."""

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

    @property
    def total_samples(self) -> int:
        return self.pcm_base_sample + len(self.pcm) // 2

    def append_text(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        pos = self.total_samples
        if self.segments and self.segments[-1].start_sample == self.segments[-1].end_sample:
            self.segments[-1].text = f"{self.segments[-1].text} {text}".strip()
        else:
            self.segments.append(ReplaySegment(text=text, start_sample=pos, end_sample=pos))

    def append_pcm(self, chunk: bytes) -> None:
        if not chunk:
            return
        if not self.segments:
            pos = self.total_samples
            self.segments.append(ReplaySegment(text="", start_sample=pos, end_sample=pos))
        self.pcm.extend(chunk)
        self.segments[-1].end_sample = self.total_samples
        # Bound memory on very long monologues while keeping the most recent tail.
        excess_samples = len(self.pcm) // 2 - MAX_RETAINED_AUDIO_SAMPLES
        if excess_samples > 0:
            del self.pcm[: excess_samples * 2]
            self.pcm_base_sample += excess_samples

    def replace_history_proposal(self, proposal: ResponseHistory | None) -> None:
        if proposal is None or proposal is self.final_history_proposal:
            return
        previous = self.final_history_proposal
        if previous is not None and (len(proposal.items), proposal.complete) < (
            len(previous.items),
            previous.complete,
        ):
            proposal.resolve()
            return
        if previous is not None:
            previous.resolve()
        self.final_history_proposal = proposal

    def resolve_history_proposal(self) -> None:
        if self.final_history_proposal is not None:
            self.final_history_proposal.resolve()
            self.final_history_proposal = None

    def _slice_pcm(self, start_sample: int, end_sample: int) -> bytes:
        lo = max(self.pcm_base_sample, start_sample)
        hi = max(lo, min(self.total_samples, end_sample))
        if hi <= lo:
            return b""
        return bytes(self.pcm[(lo - self.pcm_base_sample) * 2 : (hi - self.pcm_base_sample) * 2])

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
            if seg.start_sample < start:
                text = _unheard_text_suffix(seg.text, start - seg.start_sample, seg_len)
                pcm = self._slice_pcm(start, seg.end_sample)
            else:
                text = seg.text
                pcm = self._slice_pcm(seg.start_sample, seg.end_sample)
            if pcm and not faded and start > 0:
                pcm = apply_fade_in(pcm)
                faded = True
            if text or pcm:
                output.append((text, pcm))
        return output

    def full_text(self) -> str:
        return " ".join(seg.text for seg in self.segments if seg.text)

    def as_audio_segments(self) -> list[list[object]]:
        """Return segments in ``ResponseHandler._heard_transcript``'s shape."""
        return [
            [seg.text, seg.end_sample - seg.start_sample, seg.end_sample - seg.start_sample] for seg in self.segments
        ]


@dataclass
class PendingFalseInterruption:
    """In-flight barge-in awaiting STT confirmation before discarding or replaying."""

    interrupt_turn_id: str | None
    interrupt_turn_revision: int | None
    audio_buffer: AssistantAudioBuffer
    was_generating: bool
    estimated_played_samples: int
    deadline: float
    client_played_samples: int | None = None
    short_transcript_pending: bool = False

    def cut_samples(self) -> int:
        chosen = self.client_played_samples if self.client_played_samples is not None else self.estimated_played_samples
        return min(max(self.audio_buffer.pcm_base_sample, chosen), self.audio_buffer.total_samples)
