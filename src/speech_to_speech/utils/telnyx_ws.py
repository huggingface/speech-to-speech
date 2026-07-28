"""Shared helpers for the Telnyx managed STT and TTS backends.

Two responsibilities:

1. Audio format conversion between the pipeline's PCM int16 representation and
   the formats Telnyx's WebSocket APIs use (WAV for STT input, MP3 for TTS
   output).
2. Thin sync WebSocket clients for the two Telnyx endpoints. The pipeline runs
   handlers in plain threads, so we use ``websocket-client`` (sync) rather than
   the async ``websockets`` library used elsewhere in the codebase.

Telnyx TTS returns the response as one continuous MP3 bitstream split across
many small WebSocket frames. Individual frames are *not* independently
decodable, so :class:`Mp3StreamDecoder` pipes the stream through a long-lived
ffmpeg process and emits PCM as it becomes available.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import subprocess
import wave
from queue import Empty, Queue
from threading import Thread
from typing import Any
from urllib.parse import urlencode

import numpy as np

logger = logging.getLogger(__name__)


class TelnyxProtocolError(RuntimeError):
    """Raised when a Telnyx WebSocket endpoint reports an error frame."""


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


_EMPTY_PCM = np.array([], dtype=np.int16)


class Mp3StreamDecoder:
    """Incrementally decode a continuous MP3 byte stream to mono int16 PCM.

    Feed MP3 bytes as they arrive; each call returns whatever PCM ffmpeg has
    produced so far. Call :meth:`flush` once the stream is complete to drain
    the decoder's tail, then :meth:`close`.
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        # -probesize/-analyzeduration keep ffmpeg from buffering input while it
        # probes the stream, which would add hundreds of ms to first audio.
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-probesize",
                "32",
                "-analyzeduration",
                "0",
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
                "-flush_packets",
                "1",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._pending: Queue[bytes | None] = Queue()
        self._remainder = b""
        self._drained = False
        self._reader = Thread(target=self._drain_stdout, daemon=True)
        self._reader.start()

    def _drain_stdout(self) -> None:
        stdout = self._proc.stdout
        # Popen(stdout=PIPE) with default buffering hands back a BufferedReader,
        # which is what gives us read1().
        assert isinstance(stdout, io.BufferedReader)
        try:
            while True:
                # read1() returns as soon as any data is available, unlike
                # read(n) which blocks until n bytes or EOF.
                buf = stdout.read1(8192)
                if not buf:
                    return
                self._pending.put(buf)
        except (OSError, ValueError):
            return
        finally:
            self._pending.put(None)

    def _collect(self, block: bool) -> np.ndarray:
        parts = [self._remainder] if self._remainder else []
        self._remainder = b""
        while True:
            try:
                buf = self._pending.get(block=block, timeout=10.0)
            except Empty:
                break
            if buf is None:
                self._drained = True
                break
            parts.append(buf)
            block = False  # the first read may wait; drain the rest immediately
        raw = b"".join(parts)
        # int16 samples are 2 bytes wide; carry an odd trailing byte forward.
        if len(raw) % 2:
            self._remainder = raw[-1:]
            raw = raw[:-1]
        if not raw:
            return _EMPTY_PCM
        return np.frombuffer(raw, dtype=np.int16)

    def feed(self, mp3: bytes) -> np.ndarray:
        """Push MP3 bytes in and return any PCM available right now."""
        stdin = self._proc.stdin
        assert stdin is not None
        stdin.write(mp3)
        stdin.flush()
        return self._collect(block=False)

    def flush(self) -> np.ndarray:
        """Close the input side and return the decoder's remaining PCM."""
        stdin = self._proc.stdin
        if stdin is not None and not stdin.closed:
            stdin.close()
        parts = []
        while not self._drained:
            parts.append(self._collect(block=True))
        return np.concatenate(parts) if parts else _EMPTY_PCM

    def close(self) -> None:
        for stream in (self._proc.stdin, self._proc.stdout):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()


# ── STT WebSocket client ────────────────────────────────────────────────


STT_WS_URL = "wss://api.telnyx.com/v2/speech-to-text/transcription"

# Telnyx recommends 2 KB audio frames.
STT_CHUNK_BYTES = 2048


class TelnyxSTTClient:
    """Sync WebSocket client for Telnyx's managed STT endpoint.

    One client per utterance. The pipeline opens a fresh connection for each
    VAD segment, sends the audio as WAV in 2 KB binary frames, and reads JSON
    transcript events until the server returns the final result.
    """

    def __init__(
        self,
        api_key: str,
        engine: str = "Telnyx",
        language: str = "en",
        partial_results: bool = True,
        model: str = "",
        connect_timeout: float = 10.0,
        recv_timeout: float = 10.0,
    ) -> None:
        self.api_key = api_key
        self.engine = engine
        self.language = language
        self.partial_results = partial_results
        self.model = model
        self.connect_timeout = connect_timeout
        self.recv_timeout = recv_timeout
        self._ws: Any = None

    def url(self) -> str:
        params = {
            "transcription_engine": self.engine,
            "input_format": "wav",
            "language": self.language,
            "partial_results": "true" if self.partial_results else "false",
        }
        if self.model:
            params["model"] = self.model
        return f"{STT_WS_URL}?{urlencode(params)}"

    def connect(self) -> None:
        import websocket  # local import: optional dep

        self._ws = websocket.WebSocket()
        self._ws.connect(
            self.url(),
            header=[f"Authorization: Bearer {self.api_key}"],
            timeout=self.connect_timeout,
        )
        self._ws.settimeout(self.recv_timeout)

    def send_audio(self, pcm: np.ndarray, sample_rate: int = 16000) -> None:
        """Wrap PCM in a WAV container and send it as 2 KB binary frames."""
        import websocket  # local import: optional dep

        if self._ws is None:
            raise RuntimeError("TelnyxSTTClient.connect() must be called before send_audio()")
        payload = pcm_int16_to_wav_bytes(pcm, sample_rate=sample_rate)
        for offset in range(0, len(payload), STT_CHUNK_BYTES):
            self._ws.send(
                payload[offset : offset + STT_CHUNK_BYTES],
                opcode=websocket.ABNF.OPCODE_BINARY,
            )

    def recv_transcript(self) -> dict[str, Any] | None:
        """Receive one JSON event from the server.

        Returns ``None`` when the server closes the stream or the read times
        out, and ``{}`` for a frame carrying nothing we can use. Raises
        :class:`TelnyxProtocolError` on an error frame.
        """
        import websocket  # local import: optional dep

        if self._ws is None:
            raise RuntimeError("TelnyxSTTClient.connect() must be called before recv_transcript()")
        try:
            raw = self._ws.recv()
        except (
            websocket.WebSocketConnectionClosedException,
            websocket.WebSocketTimeoutException,
        ):
            return None
        if not raw:
            return None
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Telnyx STT: ignoring non-JSON frame: %r", raw[:200])
            return {}
        if event.get("error"):
            raise TelnyxProtocolError(str(event["error"]))
        return event

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

    One client per synthesis request. Sends an init frame, one or more content
    frames, then a stop frame. Reads JSON frames back; each carries a slice of
    a single continuous MP3 bitstream. The final frame has ``isFinal: true``
    and an empty ``audio`` payload.
    """

    def __init__(
        self,
        api_key: str,
        voice: str,
        connect_timeout: float = 10.0,
        recv_timeout: float = 10.0,
    ) -> None:
        self.api_key = api_key
        self.voice = voice
        self.connect_timeout = connect_timeout
        self.recv_timeout = recv_timeout
        self._ws: Any = None

    def url(self) -> str:
        return f"{TTS_WS_URL}?{urlencode({'voice': self.voice})}"

    def connect(self) -> None:
        import websocket  # local import: optional dep

        self._ws = websocket.WebSocket()
        self._ws.connect(
            self.url(),
            header=[f"Authorization: Bearer {self.api_key}"],
            timeout=self.connect_timeout,
        )
        self._ws.settimeout(self.recv_timeout)

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

        Returns ``(mp3_bytes, is_final)`` or ``None`` when the server closes
        the stream. ``mp3_bytes`` is a slice of a continuous MP3 bitstream and
        is empty on the final frame. Raises :class:`TelnyxProtocolError` on an
        error frame.
        """
        import websocket  # local import: optional dep

        if self._ws is None:
            raise RuntimeError("TelnyxTTSClient.connect() must be called before recv_audio()")
        try:
            raw = self._ws.recv()
        except (
            websocket.WebSocketConnectionClosedException,
            websocket.WebSocketTimeoutException,
        ):
            return None
        if not raw:
            return None
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Telnyx TTS: ignoring non-JSON frame: %r", raw[:200])
            return b"", False
        if event.get("error"):
            raise TelnyxProtocolError(str(event["error"]))
        audio_b64 = event.get("audio") or ""
        return base64.b64decode(audio_b64), bool(event.get("isFinal", False))

    def close(self) -> None:
        if self._ws is None:
            return
        try:
            self._ws.close()
        except Exception:
            pass
        self._ws = None
