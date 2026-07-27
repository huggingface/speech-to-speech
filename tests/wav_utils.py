"""Shared helper for tests that need small in-memory WAV clips."""

import io

import numpy as np
import soundfile as sf


def wav_bytes(seconds: float = 4.0, sr: int = 16000, channels: int = 1, freq: float = 440.0) -> bytes:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    data = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if channels > 1:
        data = np.stack([data] * channels, axis=1)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
