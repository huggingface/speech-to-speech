"""The Realtime PCM16 resampler must stay sample-exact and cheap at 44.1 kHz.

The polyphase implementation added here replaced a zero-stuffing + ``lfilter``
version. For 44.1 kHz -> 16 kHz that version expanded the input 160x before
filtering with 8821 taps, which cost about 9 s of CPU per second of audio, so
an `input_audio_buffer.append` at 44.1 kHz could not keep up with real time.
These tests pin the output (duration, chunk invariance, anti-aliasing, and
byte-for-byte agreement with the previous algorithm) and keep a coarse upper
bound on the cost.
"""

import time

import numpy as np
import pytest
from scipy.signal import firwin, lfilter, resample_poly

from speech_to_speech.api.openai_realtime.utils import StreamingPcm16Resampler


class _PreviousResampler:
    """The implementation this change replaced, kept as a numerical reference.

    Zero-stuffs each chunk up to the common rate, filters it with a stateful
    ``lfilter``, then decimates on the delay-aligned grid. Exact, but its cost
    grows with ``up * taps`` instead of with the amount of audio.
    """

    def __init__(self, from_rate: int, to_rate: int) -> None:
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
        self._taps = firwin(2 * 10 * max_rate + 1, cutoff=1.0 / max_rate, window=("kaiser", 5.0)) * self._up
        self._filter_state = np.zeros(self._taps.size - 1, dtype=np.float64)
        self._delay = (self._taps.size - 1) // 2

    def push(self, audio_int16: bytes) -> bytes:
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
        if self._up == self._down or not self._input_samples:
            return b""
        return self._filter(np.zeros(self._delay + self._down, dtype=np.float64))

    def _filter(self, upsampled: np.ndarray) -> bytes:
        filtered, self._filter_state = lfilter(self._taps, np.ones(1), upsampled, zi=self._filter_state)
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


def _run(resampler, samples: np.ndarray, bounds):
    parts = [resampler.push(samples[start:end].tobytes()) for start, end in zip(bounds, bounds[1:])]
    parts.append(resampler.flush())
    return b"".join(parts)


def _int16(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<i2").astype(np.int32)


RATES = [(44100, 16000), (22050, 16000), (48000, 16000), (32000, 16000), (24000, 16000), (16000, 44100)]


def test_equal_rate_passthrough_has_no_pending_output():
    resampler = StreamingPcm16Resampler(16000, 16000)
    chunk = np.arange(160, dtype="<i2").tobytes()
    for incoming in (chunk, b"", chunk):
        assert resampler.push(incoming) == incoming
        assert resampler.pending_output_samples == 0
    assert resampler.flush() == b""
    assert resampler.pending_output_samples == 0


@pytest.mark.parametrize(("from_rate", "to_rate"), RATES)
def test_matches_the_previous_implementation(from_rate, to_rate):
    """Same samples as before the change, including where each push cuts the output."""
    rng = np.random.default_rng(20261104)
    samples = rng.integers(-32768, 32768, size=1200).astype("<i2")
    bounds = [0, 137, 400, 401, 1200]

    expected = _PreviousResampler(from_rate, to_rate)
    actual = StreamingPcm16Resampler(from_rate, to_rate)
    expected_parts = [expected.push(samples[start:end].tobytes()) for start, end in zip(bounds, bounds[1:])]
    actual_parts = [actual.push(samples[start:end].tobytes()) for start, end in zip(bounds, bounds[1:])]
    expected_parts.append(expected.flush())
    actual_parts.append(actual.flush())

    assert [len(part) for part in actual_parts] == [len(part) for part in expected_parts]
    assert b"".join(actual_parts) == b"".join(expected_parts)


@pytest.mark.parametrize(("from_rate", "to_rate"), RATES)
def test_output_is_independent_of_chunk_boundaries(from_rate, to_rate):
    rng = np.random.default_rng(7)
    samples = rng.integers(-32768, 32768, size=600).astype("<i2")
    whole = _run(StreamingPcm16Resampler(from_rate, to_rate), samples, [0, 600])
    for bounds in ([0, 1, 2, 600], [0, 7, 111, 599, 600], [0, 480, 600]):
        assert _run(StreamingPcm16Resampler(from_rate, to_rate), samples, bounds) == whole


@pytest.mark.parametrize(("from_rate", "to_rate"), RATES)
def test_length_follows_the_input_duration(from_rate, to_rate):
    seconds = 0.25
    samples = np.zeros(int(from_rate * seconds), dtype="<i2")
    output = _int16(_run(StreamingPcm16Resampler(from_rate, to_rate), samples, [0, samples.size]))
    assert output.size == int(round(samples.size * to_rate / from_rate))


def test_matches_scipy_reference_at_44100():
    """A whole-signal reference, independent of both implementations."""
    rng = np.random.default_rng(44100)
    samples = rng.integers(-32768, 32768, size=4410).astype("<i2")
    actual = _int16(_run(StreamingPcm16Resampler(44100, 16000), samples, [0, 1000, 4410]))
    reference = np.clip(np.round(resample_poly(samples.astype(np.float64), 16000, 44100)), -32768, 32767)
    assert actual.size == reference.size
    np.testing.assert_allclose(actual, reference.astype(np.int32), rtol=0, atol=1)


def test_stopband_is_attenuated_at_44100():
    """Downsampling must not alias content above the 8 kHz output Nyquist."""
    from_rate = 44100
    tone = 11000  # above the 8 kHz target Nyquist
    samples = np.round(np.sin(2 * np.pi * tone * np.arange(from_rate // 4) / from_rate) * 20000).astype("<i2")
    output = _int16(_run(StreamingPcm16Resampler(from_rate, 16000), samples, [0, samples.size]))[200:-200]
    rms = np.sqrt(np.mean(output.astype(np.float64) ** 2))
    assert rms < 500  # ~0.4% of the 20000 amplitude input


def test_retained_history_is_bounded():
    resampler = StreamingPcm16Resampler(44100, 16000)
    for _ in range(200):
        resampler.push(np.zeros(882, dtype="<i2").tobytes())
    assert resampler._history.size < 200


def test_one_second_of_44100_audio_is_cheap():
    """Regression guard for #764-class cost: it used to need ~9 s of CPU.

    The bound is deliberately loose (2 s for 1 s of audio) so a slow CI worker
    cannot fail it, while the previous implementation failed it by 4x.
    """
    resampler = StreamingPcm16Resampler(44100, 16000)
    chunk = np.zeros(441, dtype="<i2").tobytes()
    started = time.perf_counter()
    for _ in range(100):
        resampler.push(chunk)
    resampler.flush()
    assert time.perf_counter() - started < 2.0
