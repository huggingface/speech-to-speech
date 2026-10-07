"""Acoustic echo cancellation for the packaged microphone/speaker client."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_SUPPORTED_RATES = (8000, 16000, 32000, 48000)


class EchoCanceller:
    """Remove the client's own playback from its microphone audio.

    Wraps the WebRTC audio processing module that LiveKit ships for Python and
    follows the wiring of LiveKit's agents console: the speaker callback passes
    every sample it plays as the reverse stream, and the microphone callback
    cleans each block before it is sent. The module works on 10 ms frames, so
    samples that do not fill a frame carry over to the next callback.
    """

    def __init__(self, send_rate: int, recv_rate: int) -> None:
        for rate in (send_rate, recv_rate):
            if rate not in _SUPPORTED_RATES:
                raise ValueError(f"Echo cancellation supports {_SUPPORTED_RATES} Hz, not {rate} Hz")
        try:
            from livekit import rtc
        except ImportError as exc:
            raise ImportError(
                "Echo cancellation requires the 'aec' extra: pip install 'speech-to-speech[aec]'"
            ) from exc

        self._rtc: Any = rtc
        self._apm = rtc.AudioProcessingModule(echo_cancellation=True)
        self._send_rate = send_rate
        self._recv_rate = recv_rate
        self._capture_frame_bytes = send_rate // 100 * 2
        self._render_frame_bytes = recv_rate // 100 * 2
        self._capture = bytearray()
        self._render = bytearray()
        self._output_delay_s = 0.0

    def render(self, audio: bytes, output_delay_s: float) -> None:
        """Feed audio that is about to play, with its time until it reaches the speaker."""

        self._output_delay_s = max(0.0, output_delay_s)
        self._render.extend(audio)
        frame_bytes = self._render_frame_bytes
        while len(self._render) >= frame_bytes:
            frame = self._frame(self._render[:frame_bytes], self._recv_rate)
            del self._render[:frame_bytes]
            self._apm.process_reverse_stream(frame)

    def capture(self, audio: bytes, input_delay_s: float) -> bytes:
        """Return microphone audio with the echo removed, in whole 10 ms frames."""

        delay_ms = int((self._output_delay_s + max(0.0, input_delay_s)) * 1000)
        try:
            self._apm.set_stream_delay_ms(delay_ms)
        except RuntimeError:
            # The delay is a hint; the canceller also estimates it.
            logger.debug("Echo canceller rejected stream delay %d ms", delay_ms)
        self._capture.extend(audio)
        frame_bytes = self._capture_frame_bytes
        cleaned = bytearray()
        while len(self._capture) >= frame_bytes:
            frame = self._frame(self._capture[:frame_bytes], self._send_rate)
            del self._capture[:frame_bytes]
            self._apm.process_stream(frame)
            cleaned.extend(bytes(frame.data))
        return bytes(cleaned)

    def _frame(self, data: bytearray, rate: int) -> Any:
        return self._rtc.AudioFrame(bytes(data), rate, 1, rate // 100)
