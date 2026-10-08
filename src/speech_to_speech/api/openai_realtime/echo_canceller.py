"""Acoustic echo cancellation for the packaged microphone/speaker client."""

from __future__ import annotations

from threading import Lock

import numpy as np

_SUPPORTED_RATES = (8000, 16000, 32000, 48000)


class EchoCanceller:
    """Pair speaker references with microphone audio in whole 10 ms frames.

    Only the microphone callback calls the native processor. The speaker
    callback feeds a bounded reference queue protected by the same lock.
    Missing references become silence; incomplete capture frames carry over.
    """

    def __init__(self, send_rate: int, recv_rate: int) -> None:
        for rate in (send_rate, recv_rate):
            if rate not in _SUPPORTED_RATES:
                raise ValueError(f"Echo cancellation supports {_SUPPORTED_RATES} Hz, not {rate} Hz")
        if send_rate != recv_rate:
            raise ValueError("Echo cancellation requires matching --send-rate and --recv-rate")
        try:
            from pywebrtc_audio import AudioProcessor
        except ImportError as exc:
            raise ImportError(
                "Echo cancellation requires the 'aec' extra: pip install 'speech-to-speech[aec]'"
            ) from exc

        self._apm = AudioProcessor(sample_rate=send_rate, echo_cancellation=True)
        self._frame_bytes = send_rate // 100 * 2
        self._render_limit = self._frame_bytes * 50  # At most 500 ms if callbacks drift or capture stalls.
        self._capture = bytearray()
        self._render = bytearray()
        self._lock = Lock()
        self._output_delay_s = 0.0

    def render(self, audio: bytes, output_delay_s: float) -> None:
        """Queue audio about to play, with its time until it reaches the speaker."""
        with self._lock:
            self._output_delay_s = max(0.0, output_delay_s)
            self._render.extend(audio)
            while len(self._render) > self._render_limit:
                del self._render[: self._frame_bytes]

    def capture(self, audio: bytes, input_delay_s: float) -> bytes:
        """Return microphone audio with echo removed, in whole 10 ms frames."""
        with self._lock:
            self._apm.stream_delay_ms = int((self._output_delay_s + max(0.0, input_delay_s)) * 1000)
            self._capture.extend(audio)
            cleaned = bytearray()
            while len(self._capture) >= self._frame_bytes:
                near = bytes(self._capture[: self._frame_bytes])
                del self._capture[: self._frame_bytes]
                available = min(len(self._render), self._frame_bytes)
                far = bytes(self._render[:available])
                del self._render[:available]
                far += bytes(self._frame_bytes - available)
                cleaned.extend(
                    self._apm.process(np.frombuffer(near, dtype=np.int16), np.frombuffer(far, dtype=np.int16)).tobytes()
                )
            return bytes(cleaned)
