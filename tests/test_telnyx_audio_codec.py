"""Tests for the Telnyx audio format helpers.

These tests exercise the WAV container and MP3 decode paths without any
network access. The MP3 test is skipped when pydub or ffmpeg are missing.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest

from speech_to_speech.utils.telnyx_ws import mp3_base64_to_pcm_int16, pcm_int16_to_wav_bytes


def test_pcm_to_wav_to_pcm_roundtrip():
    """Random int16 PCM survives a WAV round-trip with matching samples."""
    rng = np.random.default_rng(seed=42)
    pcm = rng.integers(-32768, 32767, size=16000, dtype=np.int16)

    wav_bytes = pcm_int16_to_wav_bytes(pcm, sample_rate=16000)
    assert isinstance(wav_bytes, bytes)
    assert len(wav_bytes) > 0
    # WAV header is 44 bytes for mono PCM int16
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"

    import wave

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        recovered = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    np.testing.assert_array_equal(recovered, pcm)


def test_pcm_to_wav_respects_sample_rate():
    """The WAV header carries the requested sample rate."""
    pcm = np.zeros(8000, dtype=np.int16)
    wav_bytes = pcm_int16_to_wav_bytes(pcm, sample_rate=8000)

    import wave

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getframerate() == 8000
        assert wf.getnframes() == 8000


def test_pcm_to_wav_accepts_float_input():
    """Float PCM is cast to int16 before wrapping."""
    pcm = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    wav_bytes = pcm_int16_to_wav_bytes(pcm, sample_rate=16000)

    import wave

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        recovered = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    assert recovered.dtype == np.int16
    assert len(recovered) == 5


def test_mp3_base64_to_pcm_int16():
    """A sine wave encoded as MP3 round-trips through base64 + pydub decode."""
    pytest.importorskip("pydub")

    # ffmpeg is required by pydub for MP3 decode
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    from pydub import AudioSegment

    sample_rate = 16000
    duration_s = 0.5
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    # 440 Hz sine, scaled to int16 range
    sine = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

    audio = AudioSegment(
        data=sine.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )
    buf = io.BytesIO()
    audio.export(buf, format="mp3")
    mp3_bytes = buf.getvalue()
    b64 = base64.b64encode(mp3_bytes).decode("ascii")

    recovered = mp3_base64_to_pcm_int16(b64, target_sample_rate=sample_rate)

    assert recovered.dtype == np.int16
    # MP3 is lossy, so we can't expect exact equality, but the length should
    # be close to the original (within a few frames of codec delay).
    assert abs(len(recovered) - len(sine)) < 4096
    # And the recovered signal should have non-trivial energy.
    assert np.max(np.abs(recovered)) > 1000
