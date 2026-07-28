"""Tests for the Telnyx STT handler protocol.

These tests mock the WebSocket client to verify the handler builds the right
frames, parses transcript events correctly, and surfaces errors gracefully.
"""

from __future__ import annotations

import json
import os
from queue import Queue
from threading import Event
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from speech_to_speech.pipeline.messages import Transcription, VADAudio
from speech_to_speech.STT.telnyx_stt_handler import TelnyxSTTHandler


def _make_handler(**overrides):
    """Build a TelnyxSTTHandler with a mocked WebSocket client."""
    kwargs = dict(
        should_listen=Event(),
        api_key="test-key",
        engine="Telnyx",
        language="en",
        model="",
        input_format="wav",
        partial_results=True,
        gen_kwargs={},
        cancel_scope=None,
        speculative_turns=None,
        enable_live_transcription=False,
        live_transcription_update_interval=0.25,
    )
    kwargs.update(overrides)
    handler = object.__new__(TelnyxSTTHandler)
    handler.setup(**kwargs)
    return handler


def test_stt_handler_yields_final_transcript():
    """Handler yields one Transcription with the final text from the WS stream."""
    handler = _make_handler()

    sent_frames: list[bytes] = []
    recv_frames = [
        json.dumps({"type": "transcript", "transcript": "hello ", "is_final": False, "confidence": 0.5}),
        json.dumps({"type": "transcript", "transcript": "hello world", "is_final": True, "confidence": 0.95}),
    ]

    fake_ws = MagicMock()
    fake_ws.send = lambda payload, opcode=None: sent_frames.append(payload)
    fake_ws.recv = lambda: recv_frames.pop(0) if recv_frames else ""
    fake_ws.close = MagicMock()

    fake_websocket_module = MagicMock()
    fake_websocket_module.WebSocket = MagicMock(return_value=fake_ws)

    with patch.dict("sys.modules", {"websocket": fake_websocket_module}):
        vad = VADAudio(
            audio=np.zeros(16000, dtype=np.int16),
            mode="final",
            turn_id="turn_1",
            turn_revision=1,
            created_at_s=123.0,
        )
        results = list(handler.process(vad))

    assert len(results) == 1
    assert isinstance(results[0], Transcription)
    assert results[0].text == "hello world"
    assert results[0].language_code == "en"
    assert results[0].turn_id == "turn_1"
    assert results[0].turn_revision == 1
    assert results[0].speech_stopped_at_s == 123.0

    # Exactly one binary audio frame was sent
    assert len(sent_frames) == 1
    assert sent_frames[0][:4] == b"RIFF"  # WAV container
    fake_ws.close.assert_called_once()


def test_stt_handler_yields_empty_transcript_on_error():
    """Handler yields an empty Transcription when the server reports an error."""
    handler = _make_handler()

    recv_frames = [
        json.dumps({"type": "error", "error": "boom"}),
    ]

    fake_ws = MagicMock()
    fake_ws.send = MagicMock()
    fake_ws.recv = lambda: recv_frames.pop(0) if recv_frames else ""
    fake_ws.close = MagicMock()

    fake_websocket_module = MagicMock()
    fake_websocket_module.WebSocket = MagicMock(return_value=fake_ws)

    with patch.dict("sys.modules", {"websocket": fake_websocket_module}):
        vad = VADAudio(
            audio=np.zeros(16000, dtype=np.int16),
            mode="final",
            turn_id="turn_1",
            turn_revision=1,
        )
        results = list(handler.process(vad))

    assert len(results) == 1
    assert isinstance(results[0], Transcription)
    assert results[0].text == ""


def test_stt_handler_missing_api_key_raises(monkeypatch):
    """Setup raises ValueError when no API key is provided."""
    monkeypatch.delenv("TELNYX_API_KEY", raising=False)

    handler = object.__new__(TelnyxSTTHandler)
    with pytest.raises(ValueError, match="Telnyx STT requires an API key"):
        handler.setup(api_key="")


def test_stt_handler_reads_api_key_from_env(monkeypatch):
    """Setup falls back to TELNYX_API_KEY env var when api_key is empty."""
    monkeypatch.setenv("TELNYX_API_KEY", "env-key")

    handler = object.__new__(TelnyxSTTHandler)
    handler.setup(api_key="")
    assert handler.api_key == "env-key"


def test_stt_handler_emits_partial_when_live_transcription_enabled():
    """Partial transcripts are yielded as PartialTranscription when enabled."""
    handler = _make_handler(enable_live_transcription=True, live_transcription_update_interval=0.0)

    recv_frames = [
        json.dumps({"type": "transcript", "transcript": "hel", "is_final": False}),
        json.dumps({"type": "transcript", "transcript": "hello", "is_final": False}),
        json.dumps({"type": "transcript", "transcript": "hello world", "is_final": True}),
    ]

    fake_ws = MagicMock()
    fake_ws.send = MagicMock()
    fake_ws.recv = lambda: recv_frames.pop(0) if recv_frames else ""
    fake_ws.close = MagicMock()

    fake_websocket_module = MagicMock()
    fake_websocket_module.WebSocket = MagicMock(return_value=fake_ws)

    with patch.dict("sys.modules", {"websocket": fake_websocket_module}):
        vad = VADAudio(
            audio=np.zeros(16000, dtype=np.int16),
            mode="final",
            turn_id="turn_1",
            turn_revision=1,
        )
        results = list(handler.process(vad))

    # At least one partial + one final
    assert len(results) >= 2
    assert any(r.text == "hello world" for r in results)
