"""Tests for the Telnyx audio format helpers.

Exercises the WAV container and the streaming MP3 decoder without any network
access. The MP3 tests are skipped when ffmpeg is missing.
"""

from __future__ import annotations

import io
import time
import wave

import numpy as np

from speech_to_speech.utils.telnyx_ws import Mp3StreamDecoder, pcm_int16_to_wav_bytes
from tests.telnyx_helpers import mp3_to_pcm, pcm_to_mp3, require_ffmpeg, sine_pcm, slice_stream


def test_pcm_to_wav_to_pcm_roundtrip():
    """Random int16 PCM survives a WAV round-trip with matching samples."""
    rng = np.random.default_rng(seed=42)
    pcm = rng.integers(-32768, 32767, size=16000, dtype=np.int16)

    wav_bytes = pcm_int16_to_wav_bytes(pcm, sample_rate=16000)
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        recovered = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    np.testing.assert_array_equal(recovered, pcm)


def test_pcm_to_wav_respects_sample_rate():
    """The WAV header carries the requested sample rate."""
    wav_bytes = pcm_int16_to_wav_bytes(np.zeros(8000, dtype=np.int16), sample_rate=8000)

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getframerate() == 8000
        assert wf.getnframes() == 8000


def test_pcm_to_wav_accepts_float_input():
    """Float PCM is cast to int16 before wrapping."""
    pcm = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    wav_bytes = pcm_int16_to_wav_bytes(pcm, sample_rate=16000)

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        recovered = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    assert recovered.dtype == np.int16
    assert len(recovered) == 5


def test_mp3_stream_decoder_matches_one_shot_decode():
    """Feeding a bitstream in slices produces the same PCM as decoding it whole."""
    require_ffmpeg()
    mp3 = pcm_to_mp3(sine_pcm(duration_s=1.0))
    expected = mp3_to_pcm(mp3)

    decoder = Mp3StreamDecoder(sample_rate=16000)
    try:
        parts = [decoder.feed(chunk) for chunk in slice_stream(mp3, size=480)]
        parts.append(decoder.flush())
        streamed = np.concatenate(parts)
    finally:
        decoder.close()

    np.testing.assert_array_equal(streamed, expected)


def test_mp3_stream_decoder_emits_before_flush():
    """Audio comes out while the stream is still arriving, not only at the end.

    ``feed`` never blocks, so it returns whatever ffmpeg has produced by the
    time it is called. The small sleep stands in for the network gaps between
    Telnyx frames; without any gap the reader thread may not have run yet.
    """
    require_ffmpeg()
    mp3 = pcm_to_mp3(sine_pcm(duration_s=3.0))

    decoder = Mp3StreamDecoder(sample_rate=16000)
    try:
        emitted_early = 0
        for chunk in slice_stream(mp3, size=480):
            emitted_early += len(decoder.feed(chunk))
            time.sleep(0.001)
        tail = len(decoder.flush())
    finally:
        decoder.close()

    # The bulk of a 3s clip must arrive during streaming, not in the tail.
    assert emitted_early > tail


def test_mp3_stream_decoder_output_is_audible():
    """The decoded signal carries the energy of the source tone."""
    require_ffmpeg()
    mp3 = pcm_to_mp3(sine_pcm(duration_s=0.5))

    decoder = Mp3StreamDecoder(sample_rate=16000)
    try:
        parts = [decoder.feed(chunk) for chunk in slice_stream(mp3, size=480)]
        parts.append(decoder.flush())
        pcm = np.concatenate(parts)
    finally:
        decoder.close()

    assert pcm.dtype == np.int16
    assert abs(len(pcm) - 8000) < 4096  # codec delay
    assert np.max(np.abs(pcm)) > 1000


def test_mp3_stream_decoder_handles_unaligned_slices():
    """Slices that cut across MP3 frame boundaries still reconstruct exactly.

    Telnyx slices the bitstream at arbitrary offsets, so the decoder has to
    carry partial frames across `feed` calls.
    """
    require_ffmpeg()
    mp3 = pcm_to_mp3(sine_pcm(duration_s=1.0))
    expected = mp3_to_pcm(mp3)

    decoder = Mp3StreamDecoder(sample_rate=16000)
    try:
        # 7 bytes at a time guarantees every slice is frame-misaligned.
        parts = [decoder.feed(chunk) for chunk in slice_stream(mp3, size=7)]
        parts.append(decoder.flush())
        streamed = np.concatenate(parts)
    finally:
        decoder.close()

    np.testing.assert_array_equal(streamed, expected)
