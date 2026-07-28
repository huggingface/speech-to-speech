"""Tests for the Telnyx STT handler protocol.

The fake WebSocket returns frames in the shape Telnyx documents for the
transcription endpoint: ``{"transcript", "is_final", "confidence"}``, with
errors carried in an ``error`` key. There is no ``type`` discriminator.
"""

from __future__ import annotations

import io
import json
import wave
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

pytest.importorskip("websocket")

from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio  # noqa: E402
from speech_to_speech.STT.telnyx_stt_handler import TelnyxSTTHandler  # noqa: E402
from speech_to_speech.utils.telnyx_ws import TelnyxProtocolError, TelnyxSTTClient  # noqa: E402


def _make_handler(**overrides):
    kwargs = dict(
        api_key="test-key",
        engine="Telnyx",
        language="en",
        model="",
        partial_results=True,
        gen_kwargs={},
        enable_live_transcription=False,
        live_transcription_update_interval=0.25,
    )
    kwargs.update(overrides)
    handler = object.__new__(TelnyxSTTHandler)
    handler.setup(**kwargs)
    return handler


def _fake_ws(recv_frames, sent=None):
    """A WebSocket stub that replays `recv_frames` then reports end-of-stream."""
    queue = list(recv_frames)
    ws = MagicMock()
    ws.send = (lambda payload, opcode=None: sent.append(payload)) if sent is not None else MagicMock()
    ws.recv = lambda: queue.pop(0) if queue else ""
    return ws


def _vad(**overrides):
    kwargs = dict(
        audio=np.zeros(16000, dtype=np.int16),
        mode="final",
        turn_id="turn_1",
        turn_revision=1,
        created_at_s=123.0,
    )
    kwargs.update(overrides)
    return VADAudio(**kwargs)


def test_stt_handler_yields_final_transcript():
    """The handler yields one Transcription carrying the final text."""
    handler = _make_handler()
    sent: list[bytes] = []
    ws = _fake_ws(
        [
            json.dumps({"transcript": "hello ", "is_final": False, "confidence": 0.5}),
            json.dumps({"transcript": "hello world", "is_final": True, "confidence": 0.95}),
        ],
        sent,
    )

    with patch("websocket.WebSocket", return_value=ws):
        results = list(handler.process(_vad()))

    assert len(results) == 1
    assert isinstance(results[0], Transcription)
    assert results[0].text == "hello world"
    assert results[0].language_code == "en"
    assert results[0].turn_id == "turn_1"
    assert results[0].turn_revision == 1
    assert results[0].speech_stopped_at_s == 123.0
    ws.close.assert_called_once()


def test_stt_handler_sends_chunked_wav():
    """Audio goes out as a valid WAV split into 2 KB binary frames."""
    handler = _make_handler()
    sent: list[bytes] = []
    ws = _fake_ws([json.dumps({"transcript": "ok", "is_final": True})], sent)

    with patch("websocket.WebSocket", return_value=ws):
        list(handler.process(_vad()))

    assert len(sent) > 1, "audio should be chunked, not sent as one frame"
    assert all(len(frame) <= 2048 for frame in sent)
    assert sent[0][:4] == b"RIFF"

    with wave.open(io.BytesIO(b"".join(sent)), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 16000


def test_stt_handler_yields_empty_transcript_on_error_frame():
    """An `error` frame is logged and yields an empty Transcription."""
    handler = _make_handler()
    ws = _fake_ws([json.dumps({"error": "boom"})])

    with patch("websocket.WebSocket", return_value=ws):
        results = list(handler.process(_vad()))

    assert len(results) == 1
    assert results[0].text == ""


def test_stt_handler_survives_stream_close_without_final():
    """A stream that closes before is_final still yields a Transcription."""
    handler = _make_handler()
    ws = _fake_ws([json.dumps({"transcript": "partial only", "is_final": False})])

    with patch("websocket.WebSocket", return_value=ws):
        results = list(handler.process(_vad()))

    assert len(results) == 1
    assert results[0].text == ""


def test_stt_handler_emits_partials_when_live_transcription_enabled():
    """Interim transcripts surface as PartialTranscription."""
    handler = _make_handler(enable_live_transcription=True, live_transcription_update_interval=0.0)
    ws = _fake_ws(
        [
            json.dumps({"transcript": "hel", "is_final": False}),
            json.dumps({"transcript": "hello", "is_final": False}),
            json.dumps({"transcript": "hello world", "is_final": True}),
        ]
    )

    with patch("websocket.WebSocket", return_value=ws):
        results = list(handler.process(_vad()))

    partials = [r for r in results if isinstance(r, PartialTranscription)]
    finals = [r for r in results if isinstance(r, Transcription)]
    assert [p.text for p in partials] == ["hel", "hello"]
    assert len(finals) == 1
    assert finals[0].text == "hello world"


def test_stt_handler_suppresses_partials_when_disabled():
    """Without live transcription only the final Transcription is emitted."""
    handler = _make_handler(enable_live_transcription=False)
    ws = _fake_ws(
        [
            json.dumps({"transcript": "hel", "is_final": False}),
            json.dumps({"transcript": "hello world", "is_final": True}),
        ]
    )

    with patch("websocket.WebSocket", return_value=ws):
        results = list(handler.process(_vad()))

    assert len(results) == 1
    assert isinstance(results[0], Transcription)


def test_stt_handler_missing_api_key_raises(monkeypatch):
    """Setup raises ValueError when no API key is provided."""
    monkeypatch.delenv("TELNYX_API_KEY", raising=False)

    handler = object.__new__(TelnyxSTTHandler)
    with pytest.raises(ValueError, match="Telnyx STT requires an API key"):
        handler.setup(api_key="")


def test_stt_handler_reads_api_key_from_env(monkeypatch):
    """Setup falls back to TELNYX_API_KEY when api_key is empty."""
    monkeypatch.setenv("TELNYX_API_KEY", "env-key")

    handler = object.__new__(TelnyxSTTHandler)
    handler.setup(api_key="")
    assert handler.api_key == "env-key"


def test_stt_client_url_encodes_params():
    """Query params are URL-encoded and the model is optional."""
    query = parse_qs(urlparse(TelnyxSTTClient(api_key="k", engine="Deepgram", model="nova-3").url()).query)
    assert query["transcription_engine"] == ["Deepgram"]
    assert query["input_format"] == ["wav"]
    assert query["partial_results"] == ["true"]
    assert query["model"] == ["nova-3"]

    assert "model" not in parse_qs(urlparse(TelnyxSTTClient(api_key="k").url()).query)

    encoded = TelnyxSTTClient(api_key="k", engine="Engine With Spaces&x").url()
    assert " " not in encoded.split("?", 1)[1]
    assert parse_qs(urlparse(encoded).query)["transcription_engine"] == ["Engine With Spaces&x"]


def test_stt_client_raises_on_error_frame():
    """The client surfaces error frames instead of treating them as EOF."""
    client = TelnyxSTTClient(api_key="k")
    client._ws = _fake_ws([json.dumps({"error": "bad engine"})])

    with pytest.raises(TelnyxProtocolError, match="bad engine"):
        client.recv_transcript()
