"""
Unit tests for MiniMaxTTSHandler.

Tests are structured to avoid live API calls (unit tests) while also
providing an integration test that calls the real API when MINIMAX_API_KEY
is available.
"""

import json
import os
import sys
from pathlib import Path
from queue import Queue
from threading import Event
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TTS.minimax_tts_handler import MINIMAX_TTS_VOICES, MiniMaxTTSHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(api_key="test-key", **kwargs):
    """Create a MiniMaxTTSHandler without calling setup (unit-test helper)."""
    handler = object.__new__(MiniMaxTTSHandler)
    handler.queue_in = Queue()
    handler.queue_out = Queue()
    handler.stop_event = Event()
    handler.should_listen = Event()
    handler.api_key = api_key
    handler.base_url = kwargs.get("base_url", "https://api.minimax.io")
    handler.model = kwargs.get("model", "speech-2.8-hd")
    handler.voice = kwargs.get("voice", "English_Graceful_Lady")
    handler.speed = kwargs.get("speed", 1.0)
    handler.vol = kwargs.get("vol", 1.0)
    handler.pitch = kwargs.get("pitch", 0)
    handler.blocksize = kwargs.get("blocksize", 512)
    return handler


def _sse_response(chunks_hex, *, include_done=True):
    """
    Build a fake SSE byte stream from a list of hex-encoded PCM chunks.

    Each entry in chunks_hex becomes a status=1 event. An optional
    [DONE] line is appended at the end.
    """
    lines = []
    for hex_audio in chunks_hex:
        event = {
            "data": {"status": 1, "audio": hex_audio},
            "base_resp": {"status_code": 0, "status_msg": ""},
        }
        lines.append(f"data: {json.dumps(event)}\n")
    if include_done:
        lines.append("data: [DONE]\n")
    return "".join(lines).encode()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestMiniMaxTTSVoices:
    def test_voices_list_is_non_empty(self):
        assert len(MINIMAX_TTS_VOICES) > 0

    def test_voices_list_contains_expected_entries(self):
        assert "English_Graceful_Lady" in MINIMAX_TTS_VOICES
        assert "English_Insightful_Speaker" in MINIMAX_TTS_VOICES


class TestMiniMaxTTSHandlerSetup:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        handler = object.__new__(MiniMaxTTSHandler)
        with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
            handler.setup(Event(), api_key=None)

    def test_accepts_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "env-key")
        handler = object.__new__(MiniMaxTTSHandler)
        handler.setup(Event(), api_key=None)
        assert handler.api_key == "env-key"

    def test_explicit_api_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "env-key")
        handler = object.__new__(MiniMaxTTSHandler)
        handler.setup(Event(), api_key="explicit-key")
        assert handler.api_key == "explicit-key"

    def test_default_model_is_hd(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "key")
        handler = object.__new__(MiniMaxTTSHandler)
        handler.setup(Event())
        assert handler.model == "speech-2.8-hd"

    def test_default_voice(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "key")
        handler = object.__new__(MiniMaxTTSHandler)
        handler.setup(Event())
        assert handler.voice == "English_Graceful_Lady"


class TestProcessEndOfResponse:
    def test_end_of_response_yields_done_sentinel(self):
        handler = _make_handler()
        results = list(handler.process(("__END_OF_RESPONSE__", None)))
        assert results == [b"__RESPONSE_DONE__"]

    def test_end_of_response_string_tuple(self):
        handler = _make_handler()
        results = list(handler.process(("__END_OF_RESPONSE__",)))
        assert results == [b"__RESPONSE_DONE__"]

    def test_empty_text_yields_nothing(self):
        handler = _make_handler()
        # Patch _stream_pcm_chunks to avoid real API calls
        with patch.object(handler, "_stream_pcm_chunks", return_value=iter([])):
            results = list(handler.process("   "))
        assert results == []


class TestStreamPcmChunks:
    """Test _stream_pcm_chunks SSE parsing logic via mock HTTP responses."""

    def _mock_urlopen(self, body_bytes):
        """Return a context-manager mock whose read() yields body_bytes in 4096-byte pieces."""
        response = MagicMock()
        data = [body_bytes[i : i + 4096] for i in range(0, len(body_bytes), 4096)]
        data.append(b"")  # sentinel for EOF
        response.read.side_effect = data
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=response)
        cm.__exit__ = MagicMock(return_value=False)
        return cm

    def test_single_chunk_decoded_correctly(self):
        handler = _make_handler()

        # Create 512 samples of PCM silence (int16 zeros)
        pcm = np.zeros(512, dtype=np.int16)
        hex_audio = pcm.tobytes().hex()

        body = _sse_response([hex_audio])
        cm = self._mock_urlopen(body)

        with patch("urllib.request.urlopen", return_value=cm):
            chunks = list(handler._stream_pcm_chunks("Hello"))

        assert len(chunks) == 1
        np.testing.assert_array_equal(chunks[0], pcm)

    def test_multiple_chunks_decoded_correctly(self):
        handler = _make_handler()

        n_samples = 256
        pcm1 = np.ones(n_samples, dtype=np.int16) * 100
        pcm2 = np.ones(n_samples, dtype=np.int16) * 200

        body = _sse_response([pcm1.tobytes().hex(), pcm2.tobytes().hex()])
        cm = self._mock_urlopen(body)

        with patch("urllib.request.urlopen", return_value=cm):
            chunks = list(handler._stream_pcm_chunks("Hello world"))

        assert len(chunks) == 2
        np.testing.assert_array_equal(chunks[0], pcm1)
        np.testing.assert_array_equal(chunks[1], pcm2)

    def test_api_error_stops_iteration(self):
        handler = _make_handler()

        event = {
            "data": {"status": 1, "audio": ""},
            "base_resp": {"status_code": 1004, "status_msg": "Auth failed"},
        }
        body = f"data: {json.dumps(event)}\n".encode()
        cm = self._mock_urlopen(body)

        with patch("urllib.request.urlopen", return_value=cm):
            chunks = list(handler._stream_pcm_chunks("Hello"))

        # Should produce no audio chunks (error logged, iteration stopped)
        assert chunks == []

    def test_malformed_json_is_skipped(self):
        handler = _make_handler()

        pcm = np.zeros(128, dtype=np.int16)
        body = (
            b"data: {invalid json}\n"
            + f"data: {json.dumps({'data': {'status': 1, 'audio': pcm.tobytes().hex()}, 'base_resp': {'status_code': 0}})}\n".encode()
        )
        cm = self._mock_urlopen(body)

        with patch("urllib.request.urlopen", return_value=cm):
            chunks = list(handler._stream_pcm_chunks("Hello"))

        assert len(chunks) == 1


class TestProcessBlocksizeAlignment:
    """Verify that process() yields exactly blocksize-sized chunks."""

    def _mock_stream(self, samples):
        """Yield a numpy array of the given sample count."""

        def _gen(_text):
            yield np.ones(samples, dtype=np.int16) * 42

        return _gen

    @pytest.mark.parametrize("sample_count", [100, 512, 1024, 1500])
    def test_output_chunks_are_blocksize(self, sample_count):
        handler = _make_handler(blocksize=512)

        with patch.object(
            handler, "_stream_pcm_chunks", side_effect=self._mock_stream(sample_count)
        ):
            results = list(handler.process("Test text"))

        # Filter out non-ndarray outputs
        arrays = [r for r in results if isinstance(r, np.ndarray)]
        for arr in arrays:
            assert len(arr) == 512, f"Expected 512 samples, got {len(arr)}"


# ---------------------------------------------------------------------------
# Integration test (skipped when MINIMAX_API_KEY is absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set — skipping live API integration test",
)
class TestMiniMaxTTSIntegration:
    def test_synthesize_short_sentence(self):
        """Call the real MiniMax API and verify we get back int16 audio."""
        handler = object.__new__(MiniMaxTTSHandler)
        handler.setup(
            Event(),
            model="speech-2.8-turbo",  # faster for CI
            voice="English_Graceful_Lady",
            blocksize=512,
        )

        results = list(handler.process("Hello, this is a MiniMax TTS integration test."))

        audio_chunks = [r for r in results if isinstance(r, np.ndarray)]
        assert len(audio_chunks) > 0, "Expected at least one audio chunk"

        for chunk in audio_chunks:
            assert chunk.dtype == np.int16
            assert len(chunk) == 512

    def test_turbo_model_works(self):
        handler = object.__new__(MiniMaxTTSHandler)
        handler.setup(Event(), model="speech-2.8-turbo", blocksize=512)
        results = list(handler.process("Testing turbo model."))
        audio_chunks = [r for r in results if isinstance(r, np.ndarray)]
        assert len(audio_chunks) > 0

    def test_end_of_response_sentinel_not_broken_by_api(self):
        handler = object.__new__(MiniMaxTTSHandler)
        handler.setup(Event(), blocksize=512)
        results = list(handler.process(("__END_OF_RESPONSE__", None)))
        assert results == [b"__RESPONSE_DONE__"]
