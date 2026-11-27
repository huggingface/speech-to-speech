"""Deterministic background noise for the benchmark question audio."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from speech_to_speech.evals.big_bench_audio.dataset import load_pcm16_mono


@dataclass(frozen=True)
class BackgroundNoise:
    pcm: bytes
    file_sha256: str
    snr_db: float
    seed: int

    @classmethod
    def from_file(cls, path: Path, *, snr_db: float, seed: int) -> "BackgroundNoise":
        if not -60 <= snr_db <= 60:
            raise ValueError("Noise SNR must be between -60 and 60 dB")
        contents = path.read_bytes()
        pcm = load_pcm16_mono(BytesIO(contents))
        if not any(pcm):
            raise ValueError("Background audio has no nonzero PCM samples")
        return cls(pcm, sha256(contents).hexdigest(), snr_db, seed)

    def metadata(self) -> dict[str, str | float | int]:
        return {
            "kind": "background_noise",
            "version": 1,
            "file_sha256": self.file_sha256,
            "snr_db": self.snr_db,
            "seed": self.seed,
        }

    def mix(self, pcm: bytes, item_id: int) -> bytes:
        import numpy as np

        speech = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
        speech_norm = np.linalg.norm(speech)
        if speech_norm == 0:
            return pcm
        noise = np.frombuffer(self.pcm, dtype="<i2")
        digest = sha256(f"{self.seed}:{item_id}".encode()).digest()
        offset = int.from_bytes(digest[:8], "big") % len(noise)
        window = noise[(offset + np.arange(len(speech))) % len(noise)].astype(np.float64)
        noise_norm = np.linalg.norm(window)
        if noise_norm == 0:
            raise ValueError("Selected background audio window is silent")

        mixed = speech + window * (speech_norm / noise_norm * 10 ** (-self.snr_db / 20))
        peak = np.max(np.abs(mixed))
        if peak > 32767:
            # Attenuate both components equally to preserve their ratio.
            mixed *= 32767 / peak
        return np.rint(mixed).astype("<i2").tobytes()
