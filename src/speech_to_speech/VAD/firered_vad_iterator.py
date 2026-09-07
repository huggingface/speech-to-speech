"""FireRed streaming VAD as a drop-in speech-probability source for VADIterator."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from speech_to_speech.VAD.vad_iterator import VADIterator

_FIRERED_WINDOW_SAMPLES = 400
_FIRERED_HOP_SAMPLES = 160


class FireRedFrameResult(Protocol):
    smoothed_prob: float


class FireRedStreamer(Protocol):
    def reset(self) -> None: ...

    def detect_chunk(self, audio_chunk: np.ndarray) -> list[FireRedFrameResult]: ...


class FireRedProbModel:
    """Adapt FireRedStreamVad.detect_chunk to the Silero model() call shape."""

    def __init__(self, streamer: FireRedStreamer) -> None:
        self.streamer = streamer
        self._tail = np.zeros(0, dtype=np.float32)
        self._last_prob = 0.0

    def reset_states(self) -> None:
        self._tail = np.zeros(0, dtype=np.float32)
        self._last_prob = 0.0
        self.streamer.reset()

    def __call__(self, x: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        samples = x.detach().cpu().contiguous().view(-1).numpy().astype(np.float32, copy=False)
        # FireRed fbank trains on int16 PCM. VADIterator feeds Silero-scale [-1, 1].
        audio = np.concatenate((self._tail, samples * 32768.0))
        if len(audio) < _FIRERED_WINDOW_SAMPLES:
            self._tail = audio
            return torch.tensor(self._last_prob, dtype=torch.float32)
        # detect_chunk restarts fbank, so feed complete 400-sample windows at a 160-sample hop.
        n_frames = (len(audio) - _FIRERED_WINDOW_SAMPLES) // _FIRERED_HOP_SAMPLES + 1
        chunk_end = _FIRERED_WINDOW_SAMPLES + (n_frames - 1) * _FIRERED_HOP_SAMPLES
        results = self.streamer.detect_chunk(audio[:chunk_end])
        self._tail = audio[n_frames * _FIRERED_HOP_SAMPLES :]
        if not results:
            return torch.tensor(0.0, dtype=torch.float32)
        self._last_prob = float(results[-1].smoothed_prob)
        return torch.tensor(self._last_prob, dtype=torch.float32)


class FireRedVadIterator(VADIterator):
    """Same trigger/silence/pad behaviour as Silero, with FireRed speech scores."""

    def __init__(
        self,
        streamer: FireRedStreamer,
        *,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 30,
    ) -> None:
        super().__init__(
            FireRedProbModel(streamer),
            threshold=threshold,
            sampling_rate=sampling_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )


def load_firered_streamer(model_dir: str, *, use_gpu: bool = False) -> FireRedStreamer:
    try:
        from fireredvad import FireRedStreamVad, FireRedStreamVadConfig
    except ImportError as exc:
        raise ImportError(
            "FireRedVAD is not installed. Install the optional extra with "
            '`pip install "speech-to-speech[fireredvad]"` or `pip install fireredvad`.'
        ) from exc
    config = FireRedStreamVadConfig(use_gpu=use_gpu)
    return FireRedStreamVad.from_pretrained(model_dir, config)
