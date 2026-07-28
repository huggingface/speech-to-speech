"""Shared helpers for the Telnyx managed STT and TTS backends.

Two responsibilities:

1. Audio format conversion between the pipeline's PCM int16 representation and
   the formats Telnyx's WebSocket APIs accept (WAV for STT input, base64 MP3
   for TTS output).
2. Thin sync WebSocket clients for the two Telnyx endpoints. The pipeline
   runs handlers in plain threads, so we use ``websocket-client`` (sync) rather
   than the async ``websockets`` library used elsewhere in the codebase.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import wave
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Audio format helpers ────────────────────────────────────────────────


def pcm_int16_to_wav_bytes(pcm: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Wrap a mono int16 PCM numpy array in a WAV container.

    Uses the stdlib ``wave`` module so we don't pull in a codec dependency
    just to ship audio to Telnyx STT.
    """
    if pcm.dtype != np.int16:
        pcm = pcm.astype(np.int16)
    pcm = np.ascontiguousarray(pcm)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def mp3_base64_to_pcm_int16(b64: str, target_sample_rate: int = 16000) -> np.ndarray:
    """Decode a base64-encoded MP3 frame to a mono int16 PCM numpy array.

    Requires ``pydub`` and an ``ffmpeg`` system binary. Resamples to the
    target sample rate if the source rate differs.
    """
    from pydub import AudioSegment  # local import: optional dep

    raw = base64.b64decode(b64)
    audio = AudioSegment.from_file(io.BytesIO(raw), format="mp3")

    if audio.frame_rate != target_sample_rate:
        audio = audio.set_frame_rate(target_sample_rate)
    if audio.channels != 1:
        audio = audio.set_channels(1)
    if audio.sample_width != 2:
        audio = audio.set_sample_width(2)

    samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
    return np.ascontiguousarray(samples)


# ── STT WebSocket client ────────────────────────────────────────────────


STT_WS_URL = "wss://api.telnyx.com/v2/speech-to-text/transcription"


class TelnyxSTTClient:
    """Sync WebSocket client for Telnyx's managed STT endpoint.

    One client per utterance. The pipeline opens a fresh connection for each
    VAD segment, sends the audio as a single binary frame, and reads JSON
    transcript events until the server signals end-of-stream.
    """

    def __init__(
        self,
        api_key: str,
        engine: str = "Telnyx",
        language: str = "en",
        input_format: str = "wav",
        partial_results: bool = True,
        model: str = "",
    ) -> None:
        self.api_key = api_key
        self.engine = engine
        self.language = language
        self.input_format = input_format
        self.partial_results = partial_results
        self.model = model
        self._ws: Any = None

    def connect(self) -> None:
        import websocket  # local import: optional dep

        params: list[str] = [
            f"transcription_engine={self.engine}",
            f"input_format={self.input_format}",
            f"language={self.language}",
            f"partial_results={'true' if self.partial_results else 'false'}",
        ]
        if self.model:
            params.append(f"model={self.model}")
        url = f"{STT_WS_URL}?{'&'.join(params)}"

        self._ws = websocket.WebSocket()
        self._ws.connect(
            url,
            header=[f"Authorization: Bearer {self.api_key}"],
            timeout=30,
        )

    def send_audio(self, pcm: np.ndarray, sample_rate: int = 16000) -> None:
        """Wrap PCM in the configured container and send as a binary frame."""
        import websocket  # local import: optional dep

        if self._ws is None:
            raise RuntimeError("TelnyxSTTClient.connect() must be called before send_audio()")
        if self.input_format == "wav":
            payload = pcm_int16_to_wav_bytes(pcm, sample_rate=sample_rate)
        elif self.input_format == "mp3":
            # Lazy import: pydub is only needed for MP3 input, which is rare.
            from pydub import AudioSegment

            audio = AudioSegment(
                data=pcm.astype(np.int16).tobytes(),
                sample_width=2,
                frame_rate=sample_rate,
                channels=1,
            )
            buf = io.BytesIO()
            audio.export(buf, format="mp3")
            payload = buf.getvalue()
        else:
            raise ValueError(f"Unsupported input_format: {self.input_format!r}")
        self._ws.send(payload, opcode=websocket.ABNF.OPCODE_BINARY)

    def recv_transcript(self) -> dict[str, Any] | None:
        """Receive one JSON event from the server.

        Returns ``None`` when the server closes the stream.
        """
        if self._ws is None:
            raise RuntimeError("TelnyxSTTClient.connect() must be called before recv_transcript()")
        try:
            raw = self._ws.recv()
        except Exception:
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Telnyx STT: ignoring non-JSON frame: %r", raw[:200])
            return None

    def close(self) -> None:
        if self._ws is None:
            return
        try:
            self._ws.close()
        except Exception:
            pass
        self._ws = None


# ── TTS WebSocket client ────────────────────────────────────────────────


TTS_WS_URL = "wss://api.telnyx.com/v2/text-to-speech/speech"


class TelnyxTTSClient:
    """Sync WebSocket client for Telnyx's managed TTS endpoint.

    One client per LLM response. Sends an init frame, one or more content
    frames, then a stop frame. Reads JSON audio frames back until the server
    signals ``isFinal``.
    """

    def __init__(self, api_key: str, voice: str) -> None:
        self.api_key = api_key
        self.voice = voice
        self._ws: Any = None

    def connect(self) -> None:
        import websocket  # local import: optional dep

        url = f"{TTS_WS_URL}?voice={self.voice}"
        self._ws = websocket.WebSocket()
        self._ws.connect(
            url,
            header=[f"Authorization: Bearer {self.api_key}"],
            timeout=30,
        )

    def _send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("TelnyxTTSClient.connect() must be called before sending frames")
        self._ws.send(json.dumps(payload))

    def send_init(self) -> None:
        """Send the init frame that primes the TTS session."""
        self._send_json({"text": " "})

    def send_text(self, text: str) -> None:
        """Send a content frame. Can be called multiple times for streaming."""
        self._send_json({"text": text})

    def send_stop(self) -> None:
        """Send the stop frame that closes the synthesis stream."""
        self._send_json({"text": ""})

    def recv_audio(self) -> tuple[bytes, bool] | None:
        """Receive one audio frame.

        Returns ``(raw_mp3_bytes, is_final)`` or ``None`` when the server
        closes the stream.
        """
        if self._ws is None:
            raise RuntimeError("TelnyxTTSClient.connect() must be called before recv_audio()")
        try:
            raw = self._ws.recv()
        except Exception:
            return None
        if not raw:
            return None
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Telnyx TTS: ignoring non-JSON frame: %r", raw[:200])
            return None
        audio_b64 = event.get("audio")
        if not audio_b64:
            return None
        return base64.b64decode(audio_b64), bool(event.get("isFinal", False))

    def close(self) -> None:
        if self._ws is None:
            return
        try:
            self._ws.close()
        except Exception:
            pass
        self._ws = None
