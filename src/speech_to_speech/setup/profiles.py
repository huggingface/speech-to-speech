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
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", prefix=f".{destination.name}.", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    return destination


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
        pipeline=data.get("pipeline", {}),
        credentials={name: CredentialRef(**value) for name, value in data.get("credentials", {}).items()},
        managed_services=[
            ManagedService(
                model=value["model"],
                model_path=value.get("model_path"),
            )
            for value in data.get("managed_services", [])
        ],
    )
