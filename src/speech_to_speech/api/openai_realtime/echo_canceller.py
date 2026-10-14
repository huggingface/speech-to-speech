"""Acoustic echo cancellation for the packaged microphone/speaker client."""

from __future__ import annotations

import logging
from collections import deque
from threading import Lock

import numpy as np

logger = logging.getLogger(__name__)

_SUPPORTED_RATES = (8000, 16000, 32000, 48000)
_FRAME_DURATION_S = 0.01


class EchoCanceller:
    """Match speaker references to microphone sample times, then remove echo.

    Playback and capture times must use the same clock. References stay queued
    until their playback time reaches capture, regardless of device buffering.
    Incomplete 10 ms capture frames carry over to the next callback. A native
    failure logs once and passes raw microphone audio for the rest of the session.
    """

    def __init__(self, send_rate: int, recv_rate: int) -> None:
        for rate in (send_rate, recv_rate):
            if rate not in _SUPPORTED_RATES:
                raise ValueError(f"Echo cancellation supports {_SUPPORTED_RATES} Hz, not {rate} Hz")
        if send_rate != recv_rate:
            raise ValueError("Echo cancellation requires matching --send-rate and --recv-rate")
        from pywebrtc_audio import AudioProcessor

        self._apm = AudioProcessor(sample_rate=send_rate, echo_cancellation=True)
        self._rate = send_rate
        self._frame_bytes = send_rate // 100 * 2
        self._capture = bytearray()
        self._capture_time = 0.0
        self._render: deque[tuple[float, np.ndarray]] = deque()
        self._next_render_time: float | None = None
        self._next_capture_time: float | None = None
        self._lock = Lock()
        self._failed = False

    def render(self, audio: bytes, playback_time_s: float) -> None:
        """Queue speaker samples with the playback time of their first sample."""
        with self._lock:
            if self._failed:
                return
            # Driver timestamps are estimates. Preserve continuous sample spacing
            # unless the estimate changes by at least one DSP frame.
            if self._next_render_time is not None:
                if abs(playback_time_s - self._next_render_time) < _FRAME_DURATION_S:
                    playback_time_s = self._next_render_time
            samples = np.frombuffer(audio, dtype=np.int16)
            if len(samples):
                self._render.append((playback_time_s, samples))
            self._next_render_time = playback_time_s + len(samples) / self._rate

    def capture(self, audio: bytes, capture_time_s: float, *, lost_input: bool = False) -> bytes:
        """Clean microphone samples, whose first sample was captured at the given time."""
        with self._lock:
            if self._failed:
                return audio
            try:
                return self._process(audio, capture_time_s, lost_input=lost_input)
            except Exception:
                logger.exception("Echo cancellation failed; sending raw microphone audio for the rest of the session")
                self._failed = True
                self._capture.clear()
                self._render.clear()
                return audio

    def _process(self, audio: bytes, capture_time_s: float, *, lost_input: bool) -> bytes:
        gap = self._next_capture_time is not None and abs(capture_time_s - self._next_capture_time) >= _FRAME_DURATION_S
        if lost_input or gap:
            self._capture.clear()
        elif self._next_capture_time is not None:
            capture_time_s = self._next_capture_time
        self._next_capture_time = capture_time_s + len(audio) / (self._rate * 2)
        if not self._capture:
            self._capture_time = capture_time_s
        self._capture.extend(audio)
        size = len(self._capture) // self._frame_bytes * self._frame_bytes
        if not size:
            return b""

        count = size // 2
        start = self._capture_time
        end = start + count / self._rate
        reference = np.zeros(count, dtype=np.int16)
        while self._render and self._render[0][0] + len(self._render[0][1]) / self._rate <= start:
            self._render.popleft()
        for playback_time, samples in self._render:
            if playback_time >= end:
                break
            offset = round((playback_time - start) * self._rate)
            destination = max(offset, 0)
            source = max(-offset, 0)
            available = min(count - destination, len(samples) - source)
            if available > 0:
                reference[destination : destination + available] = samples[source : source + available]

        # References already share capture's physical time axis; AEC3 estimates
        # the remaining acoustic delay. Process whole frames in one native call
        # so another Python thread cannot take the GIL between every 10 ms frame.
        self._apm.stream_delay_ms = 0
        near = np.frombuffer(bytes(self._capture[:size]), dtype=np.int16)
        cleaned = self._apm.process(near, reference).tobytes()
        del self._capture[:size]
        self._capture_time = end
        return cleaned
