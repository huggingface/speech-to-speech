"""Live API tests for the Telnyx STT and TTS backends.

These hit the real Telnyx WebSocket APIs and are skipped unless
``TELNYX_API_KEY`` is set. They assert on decoded audio and transcript text,
not just on connectivity, so a protocol change fails the test rather than
passing silently.

The STT WebSocket endpoint needs STT enabled on the account; a key without
that entitlement gets a 403 on the handshake and the STT test skips.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("TELNYX_API_KEY"),
    reason="requires TELNYX_API_KEY env var",
)

pytest.importorskip("websocket")

from speech_to_speech.utils.telnyx_ws import Mp3StreamDecoder, TelnyxSTTClient, TelnyxTTSClient  # noqa: E402
from tests.telnyx_helpers import require_ffmpeg  # noqa: E402

SENTENCE = "The quick brown fox jumps over the lazy dog."


def _synthesize(text: str, sample_rate: int = 16000) -> np.ndarray:
    client = TelnyxTTSClient(api_key=os.environ["TELNYX_API_KEY"], voice="Telnyx.NaturalHD.astra")
    decoder = Mp3StreamDecoder(sample_rate=sample_rate)
    parts = []
    try:
        client.connect()
        client.send_init()
        client.send_text(text)
        client.send_stop()
        while True:
            frame = client.recv_audio()
            if frame is None:
                break
            mp3_bytes, is_final = frame
            if mp3_bytes:
                parts.append(decoder.feed(mp3_bytes))
            if is_final:
                break
        parts.append(decoder.flush())
    finally:
        decoder.close()
        client.close()
    return np.concatenate(parts) if parts else np.array([], dtype=np.int16)


def test_live_tts_returns_audible_speech():
    """TTS returns decodable audio of a plausible length and amplitude."""
    require_ffmpeg()
    # A unique prefix avoids Telnyx's synthesis cache, exercising the streaming path.
    pcm = _synthesize(f"Test {random.randint(100000, 999999)}. {SENTENCE}")

    assert pcm.dtype == np.int16
    duration_s = len(pcm) / 16000
    assert 1.0 < duration_s < 20.0, f"implausible duration: {duration_s:.2f}s"
    assert np.max(np.abs(pcm)) > 1000, "decoded audio is silent"


def test_live_tts_respects_sample_rate():
    """Requesting 8 kHz yields roughly half the samples of 16 kHz."""
    require_ffmpeg()
    text = f"Test {random.randint(100000, 999999)}. {SENTENCE}"
    at_16k = len(_synthesize(text, sample_rate=16000))
    at_8k = len(_synthesize(text, sample_rate=8000))

    assert at_16k > 0 and at_8k > 0
    assert 1.6 < at_16k / at_8k < 2.4


def test_live_stt_transcribes_synthesized_speech():
    """STT returns a final transcript matching the synthesized sentence."""
    require_ffmpeg()
    import websocket

    pcm = _synthesize(SENTENCE)
    assert len(pcm) > 0, "TTS produced no audio to transcribe"

    client = TelnyxSTTClient(
        api_key=os.environ["TELNYX_API_KEY"],
        engine="Telnyx",
        language="en",
        partial_results=False,
    )
    try:
        client.connect()
    except websocket.WebSocketBadStatusException as e:
        if e.status_code in (401, 403):
            pytest.skip(f"account lacks STT WebSocket access (HTTP {e.status_code})")
        raise

    final_text = ""
    try:
        client.send_audio(pcm, sample_rate=16000)
        while True:
            event = client.recv_transcript()
            if event is None:
                break
            if event.get("is_final"):
                final_text = event.get("transcript") or ""
                break
    finally:
        client.close()

    assert final_text.strip(), "no final transcript returned"
    words = set(final_text.lower().replace(".", "").split())
    assert {"quick", "brown", "fox"} <= words, f"unexpected transcript: {final_text!r}"
