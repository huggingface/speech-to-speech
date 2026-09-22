"""FireRed streaming VAD backend using FireRedStreamVad start and end events."""

from __future__ import annotations

from collections import deque
from typing import Protocol

import numpy as np
import torch

_FIRERED_WINDOW_SAMPLES = 400
_FIRERED_HOP_SAMPLES = 160


class FireRedFrameResult(Protocol):
    smoothed_prob: float
    is_speech_start: bool
    is_speech_end: bool
    is_speech: bool


class FireRedStreamer(Protocol):
    def reset(self) -> None: ...

    def detect_chunk(self, audio_chunk: np.ndarray) -> list[FireRedFrameResult]: ...


class FireRedVadIterator:
    """PCM buffer and prefix pad for VADHandler, driven by FireRed start/end events."""

    def __init__(
        self,
        streamer: FireRedStreamer,
        *,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 30,
    ) -> None:
        if sampling_rate != 16000:
            raise ValueError("FireRedVadIterator only supports a sampling rate of 16000")

        self.streamer = streamer
        self.sampling_rate = sampling_rate
        self.buffer: list[torch.Tensor] = []
        self.prefix_buffer: list[torch.Tensor] = []
        self.active_speech_samples = 0
        self.last_utterance_active_speech_samples = 0
        self._candidate_speech_samples = 0
        self._pre_speech_buffer: deque[torch.Tensor] = deque()
        self._pre_speech_samples = 0
        self._tail = np.zeros(0, dtype=np.float32)
        self.speech_pad_samples = int(sampling_rate * speech_pad_ms / 1000)
        self._threshold = threshold
        self._min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self._apply_threshold(threshold)
        self._apply_min_silence_samples(self._min_silence_samples)
        self.reset_states()

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value
        self._apply_threshold(value)

    @property
    def min_silence_samples(self) -> float:
        return self._min_silence_samples

    @min_silence_samples.setter
    def min_silence_samples(self, value: float) -> None:
        self._min_silence_samples = value
        self._apply_min_silence_samples(value)

    def _apply_threshold(self, value: float) -> None:
        config = getattr(self.streamer, "config", None)
        if config is not None and hasattr(config, "speech_threshold"):
            config.speech_threshold = value
        postprocessor = getattr(self.streamer, "postprocessor", None)
        if postprocessor is not None and hasattr(postprocessor, "speech_threshold"):
            postprocessor.speech_threshold = value

    def _apply_min_silence_samples(self, samples: float) -> None:
        min_silence_frame = max(1, round(samples / self.sampling_rate * 100))
        config = getattr(self.streamer, "config", None)
        if config is not None and hasattr(config, "min_silence_frame"):
            config.min_silence_frame = min_silence_frame
        postprocessor = getattr(self.streamer, "postprocessor", None)
        if postprocessor is not None and hasattr(postprocessor, "min_silence_frame"):
            postprocessor.min_silence_frame = min_silence_frame

    def reset_states(self) -> None:
        self._tail = np.zeros(0, dtype=np.float32)
        self.triggered = False
        self.buffer = []
        self.prefix_buffer = []
        self.active_speech_samples = 0
        self.last_utterance_active_speech_samples = 0
        self._candidate_speech_samples = 0
        self._pre_speech_buffer.clear()
        self._pre_speech_samples = 0
        self.streamer.reset()

    def _num_samples(self, chunk: torch.Tensor) -> int:
        return len(chunk[0]) if chunk.dim() == 2 else len(chunk)

    def _trim_pre_speech_buffer(self, keep_samples: int) -> None:
        while self._pre_speech_buffer and self._pre_speech_samples > keep_samples:
            first = self._pre_speech_buffer[0]
            first_samples = self._num_samples(first)
            excess = self._pre_speech_samples - keep_samples

            if excess >= first_samples:
                self._pre_speech_buffer.popleft()
                self._pre_speech_samples -= first_samples
                continue

            if first.dim() == 2:
                self._pre_speech_buffer[0] = first[:, excess:]
            else:
                self._pre_speech_buffer[0] = first[excess:]
            self._pre_speech_samples -= excess

    def _remember_pre_speech(self, chunk: torch.Tensor) -> None:
        self._pre_speech_buffer.append(chunk)
        self._pre_speech_samples += self._num_samples(chunk)
        # Keep unconfirmed speech and the analysis tail even when padding is disabled.
        self._trim_pre_speech_buffer(max(self.speech_pad_samples, self._candidate_speech_samples + len(self._tail)))

    @property
    def pre_speech_samples(self) -> int:
        """Audio before the current chunk needed by the streaming STT buffer."""
        if self.triggered:
            return sum(self._num_samples(chunk) for chunk in self.prefix_buffer)
        return self._pre_speech_samples

    def speech_buffer(self) -> list[torch.Tensor]:
        if not self.prefix_buffer:
            return list(self.buffer)
        return [*self.prefix_buffer, *self.buffer]

    def _detect_frames(self, x: torch.Tensor) -> list[FireRedFrameResult]:
        samples = x.detach().cpu().contiguous().view(-1).numpy().astype(np.float32, copy=False)
        # FireRed fbank trains on int16 PCM. Incoming audio is Silero-scale [-1, 1].
        audio = np.concatenate((self._tail, samples * 32768.0))
        if len(audio) < _FIRERED_WINDOW_SAMPLES:
            self._tail = audio
            return []
        # detect_chunk restarts fbank, so feed complete 400-sample windows at a 160-sample hop.
        n_frames = (len(audio) - _FIRERED_WINDOW_SAMPLES) // _FIRERED_HOP_SAMPLES + 1
        chunk_end = _FIRERED_WINDOW_SAMPLES + (n_frames - 1) * _FIRERED_HOP_SAMPLES
        results = self.streamer.detect_chunk(audio[:chunk_end])
        self._tail = audio[n_frames * _FIRERED_HOP_SAMPLES :]
        return results

    def _append_chunk(self, x: torch.Tensor) -> None:
        self.buffer.append(x)

    def _frame_is_speech(self, frame: FireRedFrameResult) -> bool:
        is_speech = getattr(frame, "is_speech", None)
        if is_speech is not None:
            return bool(is_speech)
        return frame.smoothed_prob >= self.threshold

    def _end_utterance(self) -> list[torch.Tensor]:
        self.triggered = False
        spoken_utterance = self.speech_buffer()
        self.last_utterance_active_speech_samples = self.active_speech_samples
        self.active_speech_samples = 0
        self.buffer = []
        self.prefix_buffer = []
        return spoken_utterance

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> list[torch.Tensor] | None:
        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except Exception:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        frames = self._detect_frames(x)
        chunk_in_buffer = False
        ended_utterance: list[torch.Tensor] | None = None

        for frame_index, frame in enumerate(frames):
            if not self.triggered:
                if self._frame_is_speech(frame):
                    self._candidate_speech_samples += _FIRERED_HOP_SAMPLES
                else:
                    self._candidate_speech_samples = 0
            if frame.is_speech_start and not self.triggered:
                self.triggered = True
                # Locate the first candidate hop relative to this input chunk. The
                # remaining hops and analysis tail have already consumed audio too.
                candidate_prefix_samples = (
                    self._candidate_speech_samples
                    + (len(frames) - frame_index - 1) * _FIRERED_HOP_SAMPLES
                    + len(self._tail)
                    - self._num_samples(x)
                )
                self._trim_pre_speech_buffer(max(self.speech_pad_samples, candidate_prefix_samples))
                self.prefix_buffer = list(self._pre_speech_buffer)
                self._pre_speech_buffer.clear()
                self._pre_speech_samples = 0
                self.buffer.append(x)
                chunk_in_buffer = True
                self.active_speech_samples = self._candidate_speech_samples
                self._candidate_speech_samples = 0
            elif self.triggered and self._frame_is_speech(frame):
                self.active_speech_samples += _FIRERED_HOP_SAMPLES
            if frame.is_speech_end and self.triggered:
                if not chunk_in_buffer:
                    self.buffer.append(x)
                ended_utterance = self._end_utterance()
                chunk_in_buffer = False
                # Keep looping so a 20 s split can keep the new start; this chunk may sit in both buffers.

        if ended_utterance is not None:
            return ended_utterance

        if self.triggered:
            if not chunk_in_buffer:
                self._append_chunk(x)
            return None

        self._remember_pre_speech(x)
        return None


def load_firered_streamer(
    model_dir: str,
    *,
    use_gpu: bool = False,
    speech_threshold: float = 0.5,
    min_silence_duration_ms: int = 300,
    speech_pad_ms: int = 30,
) -> FireRedStreamer:
    try:
        from fireredvad import FireRedStreamVad, FireRedStreamVadConfig
    except ImportError as exc:
        raise ImportError(
            "FireRedVAD is not installed. Install the optional extra with "
            '`pip install "speech-to-speech[fireredvad]"` or `pip install fireredvad`.'
        ) from exc
    config = FireRedStreamVadConfig(
        use_gpu=use_gpu,
        speech_threshold=speech_threshold,
        min_silence_frame=max(1, round(min_silence_duration_ms / 10)),
        pad_start_frame=max(0, round(speech_pad_ms / 10)),
    )
    return FireRedStreamVad.from_pretrained(model_dir, config)
