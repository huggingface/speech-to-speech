"""Tests for the Telnyx TTS handler protocol.

These tests mock the WebSocket client to verify the handler builds the right
frames, decodes MP3 audio, and chunks it to the configured blocksize.
"""

from __future__ import annotations

import base64
import io
import json
from threading import Event
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.TTS.telnyx_tts_handler import TelnyxTTSHandler


def _make_handler(**overrides):
    """Build a TelnyxTTSHandler with a mocked WebSocket client."""
    kwargs = dict(
        should_listen=Event(),
        api_key="test-key",
        voice="Telnyx.NaturalHD.astra",
        sample_rate=16000,
        blocksize=512,
        gen_kwargs={},
        cancel_scope=None,
        speculative_turns=None,
    )
    kwargs.update(overrides)
    handler = object.__new__(TelnyxTTSHandler)
    handler.setup(**kwargs)
    return handler


def _make_mp3_b64(duration_s: float = 0.1, sample_rate: int = 16000) -> str:
    """Build a base64-encoded MP3 frame of a sine wave."""
    pytest.importorskip("pydub")
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    from pydub import AudioSegment

    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    sine = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    audio = AudioSegment(
        data=sine.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )
    buf = io.BytesIO()
    audio.export(buf, format="mp3")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_tts_handler_yields_audio_chunks():
    """Handler yields int16 numpy arrays of blocksize length."""
    handler = _make_handler(blocksize=512)

    mp3_b64 = _make_mp3_b64(duration_s=0.2)

    sent_frames: list[str] = []
    recv_frames = [
        json.dumps({"audio": mp3_b64, "isFinal": False}),
        json.dumps({"audio": mp3_b64, "isFinal": True}),
    ]

    fake_ws = MagicMock()
    fake_ws.send = lambda payload, opcode=None: sent_frames.append(payload)
    fake_ws.recv = lambda: recv_frames.pop(0) if recv_frames else ""
    fake_ws.close = MagicMock()

    fake_websocket_module = MagicMock()
    fake_websocket_module.WebSocket = MagicMock(return_value=fake_ws)

    with patch.dict("sys.modules", {"websocket": fake_websocket_module}):
        tts_input = TTSInput(text="Hello world.", turn_id="turn_1", turn_revision=1)
        results = list(handler.process(tts_input))

    # Init + content + stop frames
    assert len(sent_frames) == 3
    assert json.loads(sent_frames[0]) == {"text": " "}
    assert json.loads(sent_frames[1]) == {"text": "Hello world."}
    assert json.loads(sent_frames[2]) == {"text": ""}

    # At least one audio chunk yielded
    assert len(results) >= 1
    for chunk in results:
        assert isinstance(chunk, np.ndarray)
        assert chunk.dtype == np.int16
        assert len(chunk) == 512

    fake_ws.close.assert_called_once()


def test_tts_handler_end_of_response_yields_done():
    """EndOfResponse yields AUDIO_RESPONSE_DONE."""
    handler = _make_handler()

    fake_ws = MagicMock()
    fake_ws.send = MagicMock()
    fake_ws.recv = MagicMock()
    fake_ws.close = MagicMock()

    fake_websocket_module = MagicMock()
    fake_websocket_module.WebSocket = MagicMock(return_value=fake_ws)

    with patch.dict("sys.modules", {"websocket": fake_websocket_module}):
        results = list(handler.process(EndOfResponse(turn_id="turn_1", turn_revision=1)))

    assert results == [AUDIO_RESPONSE_DONE]
    # No WS interaction for EndOfResponse
    fake_ws.send.assert_not_called()


def test_tts_handler_missing_api_key_raises(monkeypatch):
    """Setup raises ValueError when no API key is provided."""
    monkeypatch.delenv("TELNYX_API_KEY", raising=False)

    handler = object.__new__(TelnyxTTSHandler)
    with pytest.raises(ValueError, match="Telnyx TTS requires an API key"):
        handler.setup(should_listen=Event(), api_key="")


def test_tts_handler_reads_api_key_from_env(monkeypatch):
    """Setup falls back to TELNYX_API_KEY env var when api_key is empty."""
    monkeypatch.setenv("TELNYX_API_KEY", "env-key")

    handler = object.__new__(TelnyxTTSHandler)
    handler.setup(should_listen=Event(), api_key="")
    assert handler.api_key == "env-key"


def test_tts_handler_breaks_on_cancel():
    """Handler stops yielding when cancel_scope marks the generation stale."""
    from speech_to_speech.pipeline.cancel_scope import CancelScope

    scope = CancelScope()
    handler = _make_handler(cancel_scope=scope, blocksize=512)

    mp3_b64 = _make_mp3_b64(duration_s=0.1)

    # Many frames so the cancel check fires
    recv_frames = [json.dumps({"audio": mp3_b64, "isFinal": False}) for _ in range(20)]

    fake_ws = MagicMock()
    fake_ws.send = MagicMock()
    fake_ws.recv = lambda: recv_frames.pop(0) if recv_frames else ""
    fake_ws.close = MagicMock()

    fake_websocket_module = MagicMock()
    fake_websocket_module.WebSocket = MagicMock(return_value=fake_ws)

    # Cancel before processing
    scope.cancel()

    with patch.dict("sys.modules", {"websocket": fake_websocket_module}):
        tts_input = TTSInput(text="Hello.", turn_id="turn_1", turn_revision=1)
        results = list(handler.process(tts_input))

    # Should bail out early; may yield zero or one chunk depending on timing
    assert len(results) <= 1
    fake_ws.close.assert_called_once()
