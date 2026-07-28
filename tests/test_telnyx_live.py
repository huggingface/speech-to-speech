"""Live API tests for the Telnyx STT and TTS backends.

These tests hit the real Telnyx WebSocket APIs and are skipped unless
``TELNYX_API_KEY`` is set in the environment. Keep them minimal — they
exist to catch credential, URL, and protocol regressions, not to validate
audio quality.
"""

from __future__ import annotations

import base64
import io
import os

import numpy as np
import pytest

from speech_to_speech.utils.telnyx_ws import (
    TelnyxSTTClient,
    TelnyxTTSClient,
    pcm_int16_to_wav_bytes,
)


pytestmark = pytest.mark.skipif(
    not os.environ.get("TELNYX_API_KEY"),
    reason="requires TELNYX_API_KEY env var",
)


def test_live_stt_connection():
    """STT WebSocket connects and accepts a short audio frame."""
    client = TelnyxSTTClient(
        api_key=os.environ["TELNYX_API_KEY"],
        engine="Telnyx",
        language="en",
        input_format="wav",
        partial_results=False,
    )
    client.connect()
    try:
        # 0.5s of silence at 16kHz
        pcm = np.zeros(8000, dtype=np.int16)
        client.send_audio(pcm, sample_rate=16000)

        # Drain a few events; we don't assert on content (silence may yield
        # nothing or an empty transcript).
        for _ in range(5):
            event = client.recv_transcript()
            if event is None:
                break
    finally:
        client.close()


def test_live_tts_connection():
    """TTS WebSocket connects and returns at least one audio frame."""
    client = TelnyxTTSClient(
        api_key=os.environ["TELNYX_API_KEY"],
        voice="Telnyx.NaturalHD.astra",
    )
    client.connect()
    try:
        client.send_init()
        client.send_text("Hello.")
        client.send_stop()

        # Drain frames until the server closes the stream
        frames_received = 0
        for _ in range(20):
            frame = client.recv_audio()
            if frame is None:
                break
            mp3_bytes, is_final = frame
            assert isinstance(mp3_bytes, bytes)
            assert len(mp3_bytes) > 0
            frames_received += 1
            if is_final:
                break

        assert frames_received >= 1
    finally:
        client.close()
