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

When an HF Hub dataset repo is configured, the hub is the source of truth
and the local folder is a mirror the server reads from:

- Startup pulls every hub voice into the local folder (hub wins).
- Uploads write locally, then push to the repo synchronously *before* the
  request succeeds; a persistently failing push rolls the local copy back so
  no instance ever holds a voice the rest of the fleet cannot see. Voice
  folders are disjoint paths, so concurrent pushes from different instances
  can only hit optimistic-lock races, absorbed by a small retry loop.
- Reads stay fresh via a revision check: one cheap repo-info call per
  listing; when the revision moved, only new voice folders are downloaded
  and voices whose folders vanished from the repo are evicted from the
  registry (operator deletion propagates with no DELETE API). Evicted
  reference files are left on disk so a live session that already resolved
  the voice finishes on it.
- A lookup miss triggers one re-sync before the voice is rejected, so a
  voice created through another instance resolves immediately.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
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
HUB_PUSH_ATTEMPTS = 3
HUB_PUSH_RETRY_SLEEP_S = 0.2


class VoiceRecord(BaseModel):
    voice_id: str
    name: str
    ref_text: str
    created_at: str


class ResolvedVoice(BaseModel):
    """What a TTS handler needs to speak in a cloned voice."""

    voice_id: str
    ref_audio: str
    ref_text: str


class VoiceValidationError(Exception):
    """A client-side problem with an uploaded voice (bad audio, bad metadata)."""

    def __init__(self, message: str, *, code: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class VoiceSyncError(Exception):
    """The hub repo could not be updated; the upload was rolled back."""


class HubVoiceRepo:
    """Thin huggingface_hub adapter — one hub call per method.

    Everything above this surface (pull, push-with-retry-and-rollback,
    revision short-circuit, mirror deletion, miss re-sync) lives in
    VoiceStore and is tested against a fake of this interface.
    """

    def __init__(self, repo_id: str) -> None:
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self._api = HfApi()
        self._api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    def revision(self) -> str:
        sha = self._api.repo_info(self.repo_id, repo_type="dataset").sha
        if sha is None:
            raise VoiceSyncError(f"Hub repo {self.repo_id} reported no revision")
        return sha

    def list_voice_ids(self, revision: Optional[str] = None) -> set[str]:
        files = self._api.list_repo_files(self.repo_id, repo_type="dataset", revision=revision)
        return {f.split("/", 1)[0] for f in files if f.endswith(f"/{METADATA_FILENAME}")}

    def download_voice(self, voice_id: str, root: Path, revision: Optional[str] = None) -> None:
        from huggingface_hub import hf_hub_download

        for filename in (METADATA_FILENAME, REF_AUDIO_FILENAME):
            hf_hub_download(
                self.repo_id,
                f"{voice_id}/{filename}",
                repo_type="dataset",
                revision=revision,
                local_dir=root,
            )

    def upload_voice(self, root: Path, voice_id: str) -> None:
        self._api.upload_folder(
            repo_id=self.repo_id,
            repo_type="dataset",
            folder_path=root / voice_id,
            path_in_repo=voice_id,
            commit_message=f"Add voice {voice_id}",
        )


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
    """Folder-per-voice library rooted at a local directory.

    With ``hub`` set (any object implementing the HubVoiceRepo surface), the
    repo is the fleet-wide source of truth — see the module docstring for the
    sync semantics.
    """

    def __init__(self, root: Path | str, hub: Optional[HubVoiceRepo] = None) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self._hub = hub
        self._lock = RLock()
        self._synced_revision: Optional[str] = None
        self._records: dict[str, VoiceRecord] = {}
        self._scan()
        if self._hub is not None:
            # Startup pull must succeed: serving with a silently unsynced
            # library would defeat the fleet-consistency contract.
            self._mirror_sync()

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

        with self._lock:
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
                if self._hub is not None:
                    self._push_with_retry(voice_id)
            except Exception as e:
                self._rollback(voice_id, existing)
                if isinstance(e, VoiceSyncError):
                    raise
                raise

            self._records[voice_id] = record
            logger.info(
                "Voice %s (%r) %s in store %s", voice_id, name, "updated" if existing else "created", self.root
            )
            return record

    def list_voices(self) -> list[VoiceRecord]:
        with self._lock:
            if self._hub is not None:
                self._refresh_from_hub()
            return sorted(self._records.values(), key=lambda r: (r.created_at, r.voice_id))

    def resolve(self, voice_id: str) -> Optional[ResolvedVoice]:
        """Resolve a voice id to its (reference audio path, transcript) pair.

        A miss triggers one hub re-sync before giving up, so a voice created
        through another instance moments ago still resolves.
        """
        with self._lock:
            record = self._records.get(voice_id)
            if record is None and self._hub is not None:
                self._refresh_from_hub()
                record = self._records.get(voice_id)
            if record is None:
                return None
            return ResolvedVoice(
                voice_id=record.voice_id,
                ref_audio=str(self.root / record.voice_id / REF_AUDIO_FILENAME),
                ref_text=record.ref_text,
            )

    # ------------------------------------------------------------------
    # Hub synchronization
    # ------------------------------------------------------------------

    def _push_with_retry(self, voice_id: str) -> None:
        """Push one voice folder to the hub, absorbing optimistic-lock races.

        Voice folders are disjoint paths, so concurrent commits from other
        instances can only conflict on the repo head — safe to retry. A push
        that keeps failing raises VoiceSyncError; the caller rolls the local
        copy back so the fleet never diverges.
        """
        assert self._hub is not None
        last_error: Exception | None = None
        for attempt in range(1, HUB_PUSH_ATTEMPTS + 1):
            try:
                self._hub.upload_voice(self.root, voice_id)
                return
            except Exception as e:  # noqa: BLE001 — any hub failure is retryable here
                last_error = e
                logger.warning(
                    "Voice %s hub push attempt %d/%d failed: %s", voice_id, attempt, HUB_PUSH_ATTEMPTS, e
                )
                if attempt < HUB_PUSH_ATTEMPTS:
                    time.sleep(HUB_PUSH_RETRY_SLEEP_S)
        raise VoiceSyncError(
            f"Could not persist voice {voice_id} to the hub repo after {HUB_PUSH_ATTEMPTS} attempts: {last_error}"
        ) from last_error

    def _rollback(self, voice_id: str, previous: Optional[VoiceRecord]) -> None:
        """Undo a failed add: restore the prior metadata or remove the folder."""
        voice_dir = self.root / voice_id
        if previous is not None:
            try:
                (voice_dir / METADATA_FILENAME).write_text(json.dumps(previous.model_dump(), indent=2))
            except OSError:
                logger.exception("Failed to restore metadata for voice %s during rollback", voice_id)
            self._records[voice_id] = previous
        else:
            shutil.rmtree(voice_dir, ignore_errors=True)
            self._records.pop(voice_id, None)

    def _refresh_from_hub(self) -> None:
        """Best-effort revision check; a brief hub outage serves the local mirror."""
        try:
            self._mirror_sync()
        except Exception as e:  # noqa: BLE001 — reads must not fail on hub hiccups
            logger.warning("Voice store hub sync failed; serving local mirror: %s", e)

    def _mirror_sync(self) -> None:
        """Bring the local mirror in line with the hub repo (hub wins).

        One repo-info call; if the revision moved, download voice folders we
        do not have and evict registry entries whose folders vanished from the
        repo. Evicted reference files stay on disk so live sessions keep
        working; the local dir is a mirror, not the source of truth.
        """
        assert self._hub is not None
        revision = self._hub.revision()
        if revision == self._synced_revision:
            return

        hub_ids = self._hub.list_voice_ids(revision)
        complete = True
        for voice_id in hub_ids:
            if self._voice_files_present(voice_id):
                continue
            try:
                self._hub.download_voice(voice_id, self.root, revision)
            except Exception as e:  # noqa: BLE001 — sync the rest, retry on next revision check
                complete = False
                logger.warning("Failed to download voice %s from hub: %s", voice_id, e)

        self._scan()
        self._records = {vid: rec for vid, rec in self._records.items() if vid in hub_ids}
        if complete:
            self._synced_revision = revision

    def _voice_files_present(self, voice_id: str) -> bool:
        voice_dir = self.root / voice_id
        return (voice_dir / METADATA_FILENAME).exists() and (voice_dir / REF_AUDIO_FILENAME).exists()

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
