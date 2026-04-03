import numpy as np
from scipy.signal import resample_poly

def resample(audio_int16: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample int16 PCM audio between sample rates using polyphase filtering."""
    if from_rate == to_rate:
        return audio_int16
    samples = np.frombuffer(audio_int16, dtype=np.int16).astype(np.float32) / 32768.0
    gcd = np.gcd(to_rate, from_rate)
    resampled = resample_poly(samples, up=to_rate // gcd, down=from_rate // gcd)
    return np.clip(resampled * 32768, -32768, 32767).astype(np.int16).tobytes()