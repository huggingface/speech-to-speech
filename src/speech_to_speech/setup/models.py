from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CredentialRef:
    service: str
    account: str


@dataclass(frozen=True)
class ManagedService:
    kind: str
    model: str
    runtime: str
    args: tuple[str, ...] = ()
    model_path: str | None = None


@dataclass
class SetupProfile:
    name: str = "default"
    pipeline: dict[str, Any] = field(default_factory=dict)
    credentials: dict[str, CredentialRef] = field(default_factory=dict)
    managed_services: list[ManagedService] = field(default_factory=list)
    schema_version: int = 1
