from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class SpeakerSegment:
    """One speaker's activity in seconds from the start of this session.

    Speaker indices follow first arrival, are local to a session, and may overlap.
    They identify model slots, not named people or separated audio sources.
    """

    speaker: int
    start: float
    end: float


class StreamingDiarizer:
    """Adapt arbitrary mono float32 audio blocks to Transformers streaming chunks.

    Feed each audio sample once within a continuous region. ``finish_utterance``
    flushes VAD-selected speech and restarts audio windows while preserving speaker
    identity across skipped silence. Each instance owns one session and must be
    called from a single worker. Only the
    overlap needed for the next chunk, the model cache, and open segments persist.
    ``push`` returns closed segments; ``active_speakers`` exposes ongoing speech.
    Call ``finish`` on a clean end, or ``reset`` to discard a disconnected session.
    """

    def __init__(self, processor: Any, model: Any, *, streaming_mode: str = "low_latency", threshold: float = 0.5):
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        required = ("set_streaming_mode", "audio_chunk_start", "num_samples_first_audio_chunk")
        if not all(hasattr(processor, name) for name in required):
            raise ValueError(
                "This processor does not support streaming diarization; install the supporting Transformers version."
            )
        processor.set_streaming_mode(streaming_mode)
        self.processor = processor
        self.model = model.eval()
        self.threshold = threshold
        self.sample_rate = processor.feature_extractor.sampling_rate
        self._hop = processor.feature_extractor.hop_length
        self.reset()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        device: str = "cpu",
        dtype: str = "float32",
        streaming_mode: str = "low_latency",
        threshold: float = 0.5,
    ) -> StreamingDiarizer:
        from transformers import AutoModelForAudioFrameClassification, AutoProcessor

        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16, or bfloat16")
        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        model, loading_info = AutoModelForAudioFrameClassification.from_pretrained(
            model_id, revision=revision, dtype=getattr(torch, dtype), output_loading_info=True
        )
        missing = loading_info.get("missing_keys", [])
        unexpected = loading_info.get("unexpected_keys", [])
        mismatched = loading_info.get("mismatched_keys", [])
        if missing or unexpected or mismatched:
            raise RuntimeError(
                "Diarization checkpoint is incompatible with the installed Transformers model "
                f"(revision={revision or 'main'}, missing={len(missing)}, unexpected={len(unexpected)}, "
                f"mismatched={len(mismatched)}). Select a compatible model revision."
            )
        model = model.to(device)
        return cls(processor, model, streaming_mode=streaming_mode, threshold=threshold)

    def reset(self) -> None:
        """Discard all session audio, speaker identities, and timestamp state."""
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start = 0
        self._samples = 0
        self._mel_cursor = 0
        self._frames = 0
        self._cache = None
        self._open: dict[int, float] = {}
        self._finished = False

    @property
    def active_speakers(self) -> tuple[int, ...]:
        return tuple(sorted(self._open))

    @property
    def active_segments(self) -> tuple[SpeakerSegment, ...]:
        """Detached snapshots of open activity, up to the last scored frame."""
        return tuple(SpeakerSegment(speaker, start, self.processed_seconds) for speaker, start in self._open.items())

    def warmup(self) -> None:
        """Initialize kernels before capture, discarding synthetic speaker state."""
        samples = self.processor.num_samples_first_audio_chunk + 4 * self.processor.num_mel_frames_per_step * self._hop
        try:
            for _ in range(2):
                self.reset()
                self.push(np.zeros(samples, dtype=np.float32), sample_rate=self.sample_rate)
                self.finish()
        finally:
            self.reset()

    @property
    def processed_seconds(self) -> float:
        return min(self._frames * self._hop, self._samples) / self.sample_rate

    @property
    def buffered_samples(self) -> int:
        return len(self._buffer)

    def push(self, audio: np.ndarray, *, sample_rate: int) -> list[SpeakerSegment]:
        if self._finished:
            raise RuntimeError("This session is finished; call reset before feeding more audio")
        if sample_rate != self.sample_rate:
            raise ValueError(f"Expected {self.sample_rate} Hz audio, got {sample_rate}; resample before push")
        audio = np.asarray(audio)
        if audio.ndim != 1 or not np.issubdtype(audio.dtype, np.floating):
            raise ValueError("Expected one-dimensional floating-point mono audio")
        if not np.isfinite(audio).all():
            raise ValueError("Audio must contain only finite samples")
        segments = []
        offset = 0
        while offset < len(audio):
            first = self._mel_cursor == 0
            required = (
                self.processor.num_samples_first_audio_chunk if first else self.processor.num_samples_per_audio_chunk
            )
            take = min(required - len(self._buffer), len(audio) - offset)
            self._buffer = np.concatenate((self._buffer, audio[offset : offset + take].astype(np.float32)))
            self._samples += take
            offset += take
            if len(self._buffer) < required:
                break
            segments.extend(self._infer(self._buffer, first=first, last=False))
            self._mel_cursor += self.processor.num_mel_frames_per_step
            next_start = self.processor.audio_chunk_start(self._mel_cursor)
            drop = next_start - self._buffer_start
            if not 0 < drop <= len(self._buffer):
                raise RuntimeError("Processor returned an invalid streaming audio boundary")
            self._buffer = self._buffer[drop:].copy()
            self._buffer_start = next_start
        return segments

    def finish_utterance(self) -> list[SpeakerSegment]:
        """Flush a VAD boundary, retaining identity but restarting audio windows.

        The next speech onset has its own centered spectrogram. Skipped silence
        must not be concatenated into an artificial waveform/STFT boundary.
        """
        segments = self.finish(preserve_speaker_cache=True)
        cache = self._cache
        self.reset()
        self._cache = cache
        return segments

    def finish(self, *, preserve_speaker_cache: bool = False) -> list[SpeakerSegment]:
        """Score the remaining look-ahead and close speech at the real audio end.

        Pad only the final analysis window with silence, then trim to the real
        duration. This also handles a recording shorter than the first chunk.
        Repeated calls are harmless; ``reset`` starts the next session.
        """
        if self._finished:
            return []
        segments = []
        remaining = ceil(self._samples / self._hop) - self._frames
        if remaining > 0:
            first = self._mel_cursor == 0
            needed = (
                remaining * self._hop if first else self.processor.feature_extractor.n_fft + (remaining - 1) * self._hop
            )
            audio = np.pad(self._buffer, (0, max(0, needed - len(self._buffer))))
            segments.extend(self._infer(audio, first=first, last=True, max_frames=remaining))
        end = self._samples / self.sample_rate
        segments.extend(SpeakerSegment(speaker, start, end) for speaker, start in self._open.items() if end > start)
        self._open.clear()
        self._buffer = np.empty(0, dtype=np.float32)
        if not preserve_speaker_cache:
            self._cache = None
        self._finished = True
        return sorted(segments, key=lambda segment: (segment.end, segment.start, segment.speaker))

    @torch.inference_mode()
    def _infer(
        self, audio: np.ndarray, *, first: bool, last: bool, max_frames: int | None = None
    ) -> list[SpeakerSegment]:
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            is_streaming=True,
            is_first_audio_chunk=first,
            is_last_audio_chunk=last,
        ).to(self.model.device, dtype=self.model.dtype)
        # A short first-and-last chunk has no cache yet. Explicit zero look-ahead
        # keeps the model in streaming mode even in that case.
        if last:
            inputs["num_lookahead_frames"] = 0
        outputs = self.model(**inputs, speaker_cache=self._cache)
        active = (outputs.logits[0, :max_frames].sigmoid() > self.threshold).cpu().numpy()
        if max_frames is not None and len(active) != max_frames:
            raise RuntimeError("Processor/model did not score all final audio frames")
        self._cache = outputs.speaker_cache
        segments = []
        for index, frame in enumerate(active):
            timestamp = min((self._frames + index) * self._hop, self._samples) / self.sample_rate
            for speaker, speaking in enumerate(frame):
                if speaking:
                    self._open.setdefault(speaker, timestamp)
                elif speaker in self._open:
                    start = self._open.pop(speaker)
                    if timestamp > start:
                        segments.append(SpeakerSegment(speaker, start, timestamp))
        self._frames += len(active)
        return segments
