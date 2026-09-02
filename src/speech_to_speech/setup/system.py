from __future__ import annotations

import platform
import shutil
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GIB = 1024**3


@dataclass(frozen=True)
class SystemSnapshot:
    memory_bytes: int
    free_bytes: int
    audio_input_count: int
    audio_output_count: int
    supported: bool
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelChoice:
    label: str
    model_id: str
    runtime: str
    estimated_bytes: int
    variant: str | None = None
    allow_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class CachedModel:
    model_id: str
    size_bytes: int


@dataclass(frozen=True)
class DiskEstimate:
    missing_bytes: int
    reserve_bytes: int
    free_bytes: int
    can_install: bool


def _read_sysctl(name: str) -> int:
    result = subprocess.run(["/usr/sbin/sysctl", "-n", name], capture_output=True, text=True, check=True)
    return int(result.stdout.strip())


def _query_audio() -> Sequence[dict[str, Any]]:
    import sounddevice

    return sounddevice.query_devices()  # type: ignore[no-any-return]


def inspect_system(
    *,
    system: str | None = None,
    machine: str | None = None,
    sysctl: Callable[[str], int] = _read_sysctl,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    audio_query: Callable[[], Sequence[dict[str, Any]]] = _query_audio,
    install_path: Path | None = None,
) -> SystemSnapshot:
    detected_system = system or platform.system()
    detected_machine = machine or platform.machine()
    diagnostics: list[str] = []
    try:
        translated = bool(sysctl("sysctl.proc_translated"))
    except (OSError, ValueError, subprocess.SubprocessError):
        translated = False
    try:
        memory_bytes = sysctl("hw.memsize")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        memory_bytes = 0
        diagnostics.append(f"Could not determine unified memory: {error}")
    try:
        free_bytes = int(disk_usage(install_path or Path.home()).free)
    except OSError as error:
        free_bytes = 0
        diagnostics.append(f"Could not determine available disk space: {error}")
    try:
        devices = audio_query()
        audio_input_count = sum(int(device.get("max_input_channels", 0)) > 0 for device in devices)
        audio_output_count = sum(int(device.get("max_output_channels", 0)) > 0 for device in devices)
    except Exception as error:
        audio_input_count = audio_output_count = 0
        diagnostics.append(f"Could not inspect audio devices: {error}")

    native_arm = detected_system == "Darwin" and detected_machine == "arm64" and not translated
    if translated:
        diagnostics.append("Python is running under Rosetta; install and run a native arm64 environment.")
    elif detected_system != "Darwin" or detected_machine != "arm64":
        diagnostics.append("The guided local installer requires Apple Silicon macOS (Darwin arm64).")

    return SystemSnapshot(
        memory_bytes=memory_bytes,
        free_bytes=free_bytes,
        audio_input_count=audio_input_count,
        audio_output_count=audio_output_count,
        supported=native_arm,
        diagnostics=tuple(diagnostics),
    )


def scan_model_caches(
    custom_directories: Iterable[Path] = (),
    *,
    scan_hf: Callable[[], Any] | None = None,
    managed_directory: Path | None = None,
) -> tuple[list[CachedModel], list[str]]:
    diagnostics: list[str] = []
    found: dict[str, CachedModel] = {}
    try:
        if scan_hf is None:
            from huggingface_hub import scan_cache_dir

            cache = scan_cache_dir()
        else:
            cache = scan_hf()
        for repo in cache.repos:
            model = CachedModel(repo.repo_id, int(repo.size_on_disk))
            _remember_largest(found, model)
    except Exception as error:
        diagnostics.append(f"Could not inspect the Hugging Face cache: {error}")

    for custom in custom_directories:
        try:
            if not custom.exists():
                diagnostics.append(f"Custom model directory does not exist: {custom}")
                continue
            for model_dir in custom.glob("models--*--*"):
                model_id = model_dir.name.removeprefix("models--").replace("--", "/")
                size = sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file())
                model = CachedModel(model_id, size)
                _remember_largest(found, model)
        except OSError as error:
            diagnostics.append(f"Could not inspect custom model directory {custom}: {error}")
    if managed_directory and managed_directory.exists():
        try:
            for model_dir in managed_directory.iterdir():
                if not model_dir.is_dir() or not (model_dir / ".complete").is_file():
                    continue
                model_id = model_dir.name.replace("--", "/")
                size = sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file())
                model = CachedModel(model_id, size)
                _remember_largest(found, model)
        except OSError as error:
            diagnostics.append(f"Could not inspect managed model directory {managed_directory}: {error}")
    return list(found.values()), diagnostics


def _remember_largest(found: dict[str, CachedModel], model: CachedModel) -> None:
    if model.size_bytes > (found[model.model_id].size_bytes if model.model_id in found else 0):
        found[model.model_id] = model


def estimate_required_space(
    choices: Iterable[ModelChoice],
    cached_models: Iterable[CachedModel],
    *,
    free_bytes: int,
    force: bool = False,
) -> DiskEstimate:
    cached_by_id: dict[str, int] = {}
    for model in cached_models:
        cached_by_id[model.model_id] = max(cached_by_id.get(model.model_id, 0), model.size_bytes)
    missing = sum(max(0, choice.estimated_bytes - cached_by_id.get(choice.model_id, 0)) for choice in choices)
    reserve = max(2 * GIB, (missing * 20 + 99) // 100)
    return DiskEstimate(missing, reserve, free_bytes, force or free_bytes >= missing + reserve)
