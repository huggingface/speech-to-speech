from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from speech_to_speech.setup.models import CredentialRef, ManagedService, SetupProfile

CURRENT_SCHEMA_VERSION = 1


def default_profile_path() -> Path:
    return Path.home() / "Library" / "Application Support" / "speech-to-speech" / "profiles" / "default.json"


def save_profile(profile: SetupProfile, path: Path | None = None) -> Path:
    destination = path or default_profile_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(profile), indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        os.close(descriptor) if _descriptor_is_open(descriptor) else None
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _descriptor_is_open(descriptor: int) -> bool:
    try:
        os.fstat(descriptor)
    except OSError:
        return False
    return True


def load_profile(path: Path | None = None) -> SetupProfile:
    source = path or default_profile_path()
    data: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    version = data.get("schema_version")
    if version != CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported setup profile schema version {version}; expected {CURRENT_SCHEMA_VERSION}. "
            "Run 'speech-to-speech setup' to recreate it."
        )
    return SetupProfile(
        schema_version=version,
        name=data.get("name", "default"),
        pipeline=data.get("pipeline", {}),
        credentials={name: CredentialRef(**value) for name, value in data.get("credentials", {}).items()},
        managed_services=[
            ManagedService(
                kind=value["kind"],
                model=value["model"],
                runtime=value["runtime"],
                args=tuple(value.get("args", ())),
            )
            for value in data.get("managed_services", [])
        ],
    )
