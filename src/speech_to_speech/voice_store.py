"""Global library of cloned voices for voice-cloning TTS backends.

A voice is a (reference audio, reference transcript) pair — Qwen3-TTS
conditions on the pair at every generation call, so there is no bake step
and the voice id is purely a registry key. The store persists one folder
per voice under a local root directory:

    <root>/<voice_id>/ref.wav     normalized 24 kHz PCM16 mono reference
    <root>/<voice_id>/voice.json  id, display name, transcript, created-at

The voice id is a truncated content hash of the uploaded audio bytes, so
identical uploads converge on one entry (retries and duplicates are free)
and re-uploading the same audio with corrected metadata overwrites the
metadata in place.

The store is owned by the server layer and injected into both the realtime
router (HTTP routes) and the TTS voice-resolution path; it is deliberately
TTS-agnostic.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_STORE_DIR = Path.home() / ".cache" / "speech-to-speech" / "voices"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MIN_CLIP_SECONDS = 3.0
MAX_CLIP_SECONDS = 60.0
STORE_SAMPLE_RATE = 24000
MAX_NAME_CHARS = 80
MAX_REF_TEXT_CHARS = 2000
VOICE_ID_HEX_CHARS = 16
REF_AUDIO_FILENAME = "ref.wav"
METADATA_FILENAME = "voice.json"


class VoiceRecord(BaseModel):
    voice_id: str
    name: str
    ref_text: str
    created_at: str


class VoiceValidationError(Exception):
    """A client-side problem with an uploaded voice (bad audio, bad metadata)."""

    def __init__(self, message: str, *, code: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _validate_metadata(name: str, ref_text: str) -> tuple[str, str]:
    name = (name or "").strip()
    if not name or len(name) > MAX_NAME_CHARS:
        raise VoiceValidationError(
            f"Voice name is required and must be at most {MAX_NAME_CHARS} characters.",
            code="invalid_name",
        )
    ref_text = (ref_text or "").strip()
    if not ref_text or len(ref_text) > MAX_REF_TEXT_CHARS:
        raise VoiceValidationError(
            f"Reference transcript is required and must be at most {MAX_REF_TEXT_CHARS} characters.",
            code="invalid_transcript",
        )
    return name, ref_text


def _decode_and_normalize(audio_bytes: bytes) -> tuple[np.ndarray, float]:
    """Decode a WAV upload and return (mono 24 kHz float32 waveform, duration seconds)."""
    import soundfile as sf

    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise VoiceValidationError(
            f"Reference audio must be at most {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            code="audio_too_large",
            status_code=413,
        )

    try:
        with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
            container = f.format
            sample_rate = f.samplerate
            waveform = f.read(always_2d=False, dtype="float32")
    except Exception as e:
        raise VoiceValidationError(
            f"Reference audio is not readable WAV audio: {e}",
            code="audio_unreadable",
            status_code=415,
        ) from e

    if container != "WAV":
        raise VoiceValidationError(
            f"Reference audio must be a WAV file (got {container}).",
            code="audio_unreadable",
            status_code=415,
        )

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    duration = waveform.shape[0] / float(sample_rate) if sample_rate else 0.0
    if duration < MIN_CLIP_SECONDS:
        raise VoiceValidationError(
            f"Reference audio is too short: {duration:.1f}s (minimum {MIN_CLIP_SECONDS:.0f}s).",
            code="audio_too_short",
        )
    if duration > MAX_CLIP_SECONDS:
        raise VoiceValidationError(
            f"Reference audio is too long: {duration:.1f}s (maximum {MAX_CLIP_SECONDS:.0f}s).",
            code="audio_too_long",
        )

    if sample_rate != STORE_SAMPLE_RATE:
        from scipy.signal import resample_poly

        gcd = np.gcd(int(sample_rate), STORE_SAMPLE_RATE)
        waveform = resample_poly(waveform, up=STORE_SAMPLE_RATE // gcd, down=int(sample_rate) // gcd)

    return waveform, duration


class VoiceStore:
    """Folder-per-voice library rooted at a local directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, VoiceRecord] = {}
        self._scan()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_voice(self, audio_bytes: bytes, *, ref_text: str, name: str) -> VoiceRecord:
        """Validate, normalize, and persist a voice; returns its registry record.

        Raises VoiceValidationError on any client-side problem. Identical audio
        converges on one voice id; re-uploads overwrite name and transcript.
        """
        name, ref_text = _validate_metadata(name, ref_text)
        waveform, _duration = _decode_and_normalize(audio_bytes)

        voice_id = hashlib.sha256(audio_bytes).hexdigest()[:VOICE_ID_HEX_CHARS]
        existing = self._records.get(voice_id)
        record = VoiceRecord(
            voice_id=voice_id,
            name=name,
            ref_text=ref_text,
            created_at=existing.created_at if existing else datetime.now(timezone.utc).isoformat(),
        )

        voice_dir = self.root / voice_id
        try:
            voice_dir.mkdir(parents=True, exist_ok=True)
            if existing is None:
                import soundfile as sf

                sf.write(
                    str(voice_dir / REF_AUDIO_FILENAME),
                    waveform,
                    STORE_SAMPLE_RATE,
                    format="WAV",
                    subtype="PCM_16",
                )
            (voice_dir / METADATA_FILENAME).write_text(json.dumps(record.model_dump(), indent=2))
        except Exception:
            shutil.rmtree(voice_dir, ignore_errors=True)
            self._records.pop(voice_id, None)
            raise

        self._records[voice_id] = record
        logger.info("Voice %s (%r) %s in store %s", voice_id, name, "updated" if existing else "created", self.root)
        return record

    def list_voices(self) -> list[VoiceRecord]:
        return sorted(self._records.values(), key=lambda r: (r.created_at, r.voice_id))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _scan(self) -> None:
        """Rebuild the in-memory registry from the folders on disk."""
        records: dict[str, VoiceRecord] = {}
        for meta_path in self.root.glob(f"*/{METADATA_FILENAME}"):
            voice_id = meta_path.parent.name
            record = self._read_record(meta_path, voice_id)
            if record is not None:
                records[voice_id] = record
        self._records = records

    def _read_record(self, meta_path: Path, voice_id: str) -> Optional[VoiceRecord]:
        try:
            record = VoiceRecord.model_validate_json(meta_path.read_text())
        except Exception as e:
            logger.warning("Skipping unreadable voice metadata %s: %s", meta_path, e)
            return None
        if record.voice_id != voice_id:
            logger.warning(
                "Skipping voice folder %s: metadata id %s does not match folder name",
                meta_path.parent,
                record.voice_id,
            )
            return None
        if not (meta_path.parent / REF_AUDIO_FILENAME).exists():
            logger.warning("Skipping voice folder %s: missing %s", meta_path.parent, REF_AUDIO_FILENAME)
            return None
        return record
