"""Shared fixtures for the Telnyx backend tests.

Builds real MP3 bitstreams with ffmpeg so the tests exercise the same decode
path as production. No pydub, no network.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")


def sine_pcm(duration_s: float = 1.0, sample_rate: int = 16000, freq: int = 440) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)


def pcm_to_mp3(pcm: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Encode mono int16 PCM to an MP3 bitstream."""
    require_ffmpeg()
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-i",
            "pipe:0",
            "-f",
            "mp3",
            "pipe:1",
        ],
        input=pcm.tobytes(),
        capture_output=True,
        check=True,
    )
    return proc.stdout


def mp3_to_pcm(mp3: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Decode a complete MP3 bitstream to mono int16 PCM in one shot."""
    require_ffmpeg()
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "mp3",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "pipe:1",
        ],
        input=mp3,
        capture_output=True,
        check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.int16)


def slice_stream(data: bytes, size: int = 480) -> list[bytes]:
    """Split a byte stream into fixed-size slices.

    Telnyx sends the MP3 response this way: arbitrary slices of one continuous
    bitstream, not a sequence of standalone MP3 files.
    """
    return [data[i : i + size] for i in range(0, len(data), size)]
