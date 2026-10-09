import numpy as np
from scipy.signal import firwin, lfilter


class StreamingPcm16Resampler:
    """Stateful PCM16 resampler whose output is independent of chunk boundaries."""

    def __init__(self, from_rate: int, to_rate: int) -> None:
        if from_rate <= 0 or to_rate <= 0:
            raise ValueError("sample rates must be positive")
        self.from_rate = from_rate
        self.to_rate = to_rate
        gcd = int(np.gcd(from_rate, to_rate))
        self._up = to_rate // gcd
        self._down = from_rate // gcd
        self._input_samples = 0
        self._upsampled_samples = 0
        self._output_samples = 0

        if self._up == self._down:
            self._taps = np.ones(1, dtype=np.float64)
            self._filter_state = np.empty(0, dtype=np.float64)
            self._delay = 0
            return

        max_rate = max(self._up, self._down)
        half_length = 10 * max_rate
        self._taps = (
            firwin(
                2 * half_length + 1,
                cutoff=1.0 / max_rate,
                window=("kaiser", 5.0),
            )
            * self._up
        )
        self._filter_state = np.zeros(self._taps.size - 1, dtype=np.float64)
        self._delay = (self._taps.size - 1) // 2

    def push(self, audio_int16: bytes) -> bytes:
        """Resample one even-length PCM16 fragment while retaining FIR state."""
        if len(audio_int16) % 2:
            raise ValueError("PCM16 audio must contain complete samples")
        if not audio_int16:
            return b""
        if self._up == self._down:
            return audio_int16

        incoming = np.frombuffer(audio_int16, dtype="<i2").astype(np.float64)
        self._input_samples += incoming.size
        upsampled = np.zeros(incoming.size * self._up, dtype=np.float64)
        upsampled[:: self._up] = incoming
        return self._filter(upsampled)

    def flush(self) -> bytes:
        """Finish the stream with zero padding, preserving its intended duration."""
        if self._up == self._down or not self._input_samples:
            return b""
        # Supply the lookahead withheld by delay compensation. _filter caps
        # output at the duration of real input, excluding this padding.
        return self._filter(np.zeros(self._delay + self._down, dtype=np.float64))

    @property
    def pending_output_samples(self) -> int:
        """Output samples withheld until the input stream is finished."""
        target = (self._input_samples * self.to_rate + self.from_rate - 1) // self.from_rate
        return max(0, target - self._output_samples)

    def _filter(self, upsampled: np.ndarray) -> bytes:
        filtered, self._filter_state = lfilter(
            self._taps,
            np.ones(1, dtype=np.float64),
            upsampled,
            zi=self._filter_state,
        )

        global_start = self._upsampled_samples
        self._upsampled_samples += upsampled.size
        minimum_index = max(global_start, self._delay)
        phase = (minimum_index - self._delay) % self._down
        first_index = minimum_index if phase == 0 else minimum_index + self._down - phase
        local_start = first_index - global_start
        output = filtered[local_start :: self._down] if local_start < filtered.size else np.empty(0, dtype=np.float64)

        target_total = (self._input_samples * self.to_rate + self.from_rate - 1) // self.from_rate
        remaining = max(0, target_total - self._output_samples)
        if output.size > remaining:
            output = output[:remaining]
        self._output_samples += output.size
        return np.clip(np.round(output), -32768, 32767).astype("<i2").tobytes()
