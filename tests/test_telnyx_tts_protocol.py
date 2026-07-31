"""Tests for the Telnyx TTS handler protocol.

The fake WebSocket replays frames the way Telnyx actually sends them: one
continuous MP3 bitstream sliced across many small frames, terminated by an
``isFinal`` frame with an empty audio payload.
"""

from __future__ import annotations

import base64
import json
from threading import Event
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

pytest.importorskip("websocket")

from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput  # noqa: E402
from speech_to_speech.TTS.telnyx_tts_handler import TelnyxTTSHandler  # noqa: E402
from speech_to_speech.utils.telnyx_ws import TelnyxProtocolError, TelnyxTTSClient  # noqa: E402
from tests.telnyx_helpers import mp3_to_pcm, pcm_to_mp3, require_ffmpeg, sine_pcm, slice_stream  # noqa: E402


def _make_handler(**overrides):
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


def _audio_frames(mp3: bytes, slice_size: int = 480) -> list[str]:
    """Build the frame sequence Telnyx sends for one synthesis request."""
    frames = [
        json.dumps({"audio": base64.b64encode(s).decode("ascii"), "isFinal": False, "text": None})
        for s in slice_stream(mp3, size=slice_size)
    ]
    frames.append(json.dumps({"audio": "", "isFinal": True, "text": None}))
    return frames


def _fake_ws(recv_frames, sent=None):
    queue = list(recv_frames)
    ws = MagicMock()
    ws.send = (lambda payload, opcode=None: sent.append(payload)) if sent is not None else MagicMock()
    ws.recv = lambda: queue.pop(0) if queue else ""
    return ws


def test_tts_handler_reconstructs_the_full_stream():
    """Sliced frames decode to the same audio as the original bitstream."""
    require_ffmpeg()
    mp3 = pcm_to_mp3(sine_pcm(duration_s=1.0))
    expected = mp3_to_pcm(mp3)

    handler = _make_handler(blocksize=512)
    ws = _fake_ws(_audio_frames(mp3))

    with patch("websocket.WebSocket", return_value=ws):
        chunks = list(handler.process(TTSInput(text="Hello world.", turn_id="turn_1", turn_revision=1)))

    assert chunks, "handler produced no audio"
    for chunk in chunks:
        assert isinstance(chunk, np.ndarray)
        assert chunk.dtype == np.int16
        assert len(chunk) == 512

    produced = np.concatenate(chunks)
    # The last block is zero-padded up to blocksize.
    assert len(produced) - len(expected) < 512
    np.testing.assert_array_equal(produced[: len(expected)], expected)
    ws.close.assert_called_once()


def test_tts_handler_sends_init_content_stop():
    """The client emits the documented three-frame sequence."""
    require_ffmpeg()
    sent: list[str] = []
    ws = _fake_ws(_audio_frames(pcm_to_mp3(sine_pcm(duration_s=0.2))), sent)
    handler = _make_handler()

    with patch("websocket.WebSocket", return_value=ws):
        list(handler.process(TTSInput(text="Hello world.", turn_id="turn_1", turn_revision=1)))

    assert [json.loads(f) for f in sent] == [{"text": " "}, {"text": "Hello world."}, {"text": ""}]


def test_tts_handler_stops_at_is_final():
    """Frames after isFinal are never read."""
    require_ffmpeg()
    mp3 = pcm_to_mp3(sine_pcm(duration_s=0.2))
    frames = _audio_frames(mp3)
    frames.append(json.dumps({"audio": base64.b64encode(mp3).decode("ascii"), "isFinal": False}))

    remaining = list(frames)
    ws = MagicMock()
    ws.recv = lambda: remaining.pop(0) if remaining else ""
    handler = _make_handler()

    with patch("websocket.WebSocket", return_value=ws):
        list(handler.process(TTSInput(text="Hi.", turn_id="turn_1", turn_revision=1)))

    assert len(remaining) == 1, "handler kept reading past the final frame"


def test_tts_handler_yields_nothing_on_error_frame():
    """An error frame aborts synthesis instead of being read as end-of-stream."""
    ws = _fake_ws([json.dumps({"error": "voice not found"})])
    handler = _make_handler()

    with patch("websocket.WebSocket", return_value=ws):
        chunks = list(handler.process(TTSInput(text="Hi.", turn_id="turn_1", turn_revision=1)))

    assert chunks == []
    ws.close.assert_called_once()


def test_tts_handler_end_of_response_yields_done():
    """EndOfResponse yields AUDIO_RESPONSE_DONE and touches no socket."""
    ws = _fake_ws([])
    handler = _make_handler()

    with patch("websocket.WebSocket", return_value=ws):
        results = list(handler.process(EndOfResponse(turn_id="turn_1", turn_revision=1)))

    assert results == [AUDIO_RESPONSE_DONE]
    ws.send.assert_not_called()


def test_tts_handler_stops_mid_stream_on_cancel():
    """Cancelling during playback abandons the rest of the stream."""
    require_ffmpeg()
    from speech_to_speech.pipeline.cancel_scope import CancelScope

    scope = CancelScope()
    handler = _make_handler(cancel_scope=scope, blocksize=512)

    frames = _audio_frames(pcm_to_mp3(sine_pcm(duration_s=3.0)))
    remaining = list(frames)

    def recv():
        # Interrupt part-way through, the way a barge-in would.
        if len(remaining) == len(frames) // 2:
            scope.cancel()
        return remaining.pop(0) if remaining else ""

    ws = MagicMock()
    ws.recv = recv

    with patch("websocket.WebSocket", return_value=ws):
        chunks = list(handler.process(TTSInput(text="Hello.", turn_id="turn_1", turn_revision=1)))

    assert remaining, "handler drained the stream instead of bailing out on cancel"
    # Whatever was already decoded may be yielded; the tail must not be.
    assert len(np.concatenate(chunks)) < 3 * 16000 if chunks else True
    ws.close.assert_called_once()


def test_tts_handler_missing_api_key_raises(monkeypatch):
    """Setup raises ValueError when no API key is provided."""
    monkeypatch.delenv("TELNYX_API_KEY", raising=False)

    handler = object.__new__(TelnyxTTSHandler)
    with pytest.raises(ValueError, match="Telnyx TTS requires an API key"):
        handler.setup(should_listen=Event(), api_key="")


def test_tts_handler_reads_api_key_from_env(monkeypatch):
    """Setup falls back to TELNYX_API_KEY when api_key is empty."""
    monkeypatch.setenv("TELNYX_API_KEY", "env-key")

    handler = object.__new__(TelnyxTTSHandler)
    handler.setup(should_listen=Event(), api_key="")
    assert handler.api_key == "env-key"


def test_tts_client_url_encodes_voice():
    """Voice identifiers are URL-encoded."""
    url = TelnyxTTSClient(api_key="k", voice="ElevenLabs.Voice Name&x").url()
    assert " " not in url.split("?", 1)[1]
    assert parse_qs(urlparse(url).query)["voice"] == ["ElevenLabs.Voice Name&x"]


def test_tts_client_reports_final_frame():
    """The terminal frame decodes to empty audio with is_final set."""
    client = TelnyxTTSClient(api_key="k", voice="v")
    client._ws = _fake_ws([json.dumps({"audio": "", "isFinal": True, "text": None})])

    assert client.recv_audio() == (b"", True)


def test_tts_client_raises_on_error_frame():
    """The client surfaces error frames instead of treating them as EOF."""
    client = TelnyxTTSClient(api_key="k", voice="v")
    client._ws = _fake_ws([json.dumps({"error": "voice not found"})])

    with pytest.raises(TelnyxProtocolError, match="voice not found"):
        client.recv_audio()
