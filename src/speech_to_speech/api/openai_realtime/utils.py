import numpy as np
from scipy.signal import firwin, upfirdn


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
        self._output_samples = 0
        self._history = np.empty(0, dtype=np.float64)

        if self._up == self._down:
            self._taps = np.ones(1, dtype=np.float64)
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
        self._delay = (self._taps.size - 1) // 2

    def push(self, audio_int16: bytes) -> bytes:
        """Resample one even-length PCM16 fragment while retaining FIR state."""
        if len(audio_int16) % 2:
            raise ValueError("PCM16 audio must contain complete samples")
        if not audio_int16:
            return b""
        if self._up == self._down:
            self._input_samples += np.frombuffer(audio_int16, dtype="<i2").size
            self._output_samples = self._input_samples
            return audio_int16

        incoming = np.frombuffer(audio_int16, dtype="<i2").astype(np.float64)
        return self._process(incoming, final=False)

    def flush(self) -> bytes:
        """Finish the stream with zero padding, preserving its intended duration."""
        if self._up == self._down or not self._input_samples:
            return b""
        return self._process(np.empty(0, dtype=np.float64), final=True)

    @property
    def pending_output_samples(self) -> int:
        """Output samples withheld until the input stream is finished."""
        target = (self._input_samples * self.to_rate + self.from_rate - 1) // self.from_rate
        return max(0, target - self._output_samples)

    def _process(self, incoming: np.ndarray, *, final: bool) -> bytes:
        """Filter through the retained history and emit the newly available samples.

        The filter is a plain FIR, so it is applied as a polyphase resample of
        the retained input instead of zero-stuffing the chunk up to the common
        rate and running a stateful IIR-shaped ``lfilter`` over that grid: for
        44.1 kHz -> 16 kHz the zero-stuffed grid is 160x the input and the
        filter has 8821 taps, which cost ~9 s of CPU per second of audio. The
        polyphase form produces the same samples for a bounded amount of
        history. `tests/openai_realtime/test_input_resampler.py` pins the
        equivalence against the previous implementation.
        """
        # Offset of the retained history in the global upsampled grid.
        origin = (self._input_samples - self._history.size) * self._up
        self._input_samples += incoming.size
        source = np.concatenate((self._history, incoming))

        if final:
            target_total = (self._input_samples * self._up + self._down - 1) // self._down
        else:
            # Emit only samples whose centred FIR has every input it needs.
            target_total = max(0, (self._input_samples * self._up - self._delay + self._down - 1) // self._down)
        count = target_total - self._output_samples

        if count > 0 and source.size:
            # Align the local polyphase grid with delay + k * down globally.
            prepad = (origin - self._delay) % self._down
            taps = np.pad(self._taps, (prepad, 0)) if prepad else self._taps
            filtered = upfirdn(taps, source, up=self._up, down=self._down)
            start = (self._delay + self._output_samples * self._down - origin + prepad) // self._down
            output = filtered[start : start + count]
        else:
            output = np.empty(0, dtype=np.float64)

        history_size = (self._taps.size - 1 + self._up - 1) // self._up + 1
        self._history = np.empty(0, dtype=np.float64) if final else source[-history_size:].copy()
        self._output_samples += output.size
        return np.clip(np.round(output), -32768, 32767).astype("<i2").tobytes()
