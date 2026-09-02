from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Sequence
from pathlib import Path

from rich.console import Console

from speech_to_speech.setup.endpoints import discover_endpoints
from speech_to_speech.setup.keychain import MacOSKeychain
from speech_to_speech.setup.models import CredentialRef, SetupProfile
from speech_to_speech.setup.profiles import load_profile
from speech_to_speech.setup.system import SystemSnapshot, inspect_system


def redact(message: str) -> str:
    message = re.sub(r"(?i)(authorization\s*:\s*)(?:bearer\s+)?\S+", r"\1[REDACTED]", message)
    return re.sub(r"(?i)(api[_ -]?key|secret|token)(\s*[=:]\s*)\S+", r"\1\2[REDACTED]", message)


def default_log_path() -> Path:
    return Path.home() / "Library" / "Logs" / "speech-to-speech" / "doctor.log"


def _discovered_endpoint_urls() -> set[str]:
    return {candidate.base_url for candidate in discover_endpoints()}


def _default_emit(message: str) -> None:
    safe = redact(message)
    Console().print(safe)
    path = default_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(re.sub(r"\[[^]]+\]", "", safe) + "\n")


def run_doctor(
    argv: Sequence[str] | None = None,
    *,
    profile_loader: Callable[[], SetupProfile] = load_profile,
    inspect: Callable[[], SystemSnapshot] = inspect_system,
    credential_getter: Callable[[CredentialRef], str] | None = None,
    endpoint_urls: Callable[[], set[str]] = _discovered_endpoint_urls,
    runtime_present: Callable[[], bool] | None = None,
    emit: Callable[[str], None] = _default_emit,
) -> int:
    parser = argparse.ArgumentParser(prog="speech-to-speech doctor", description="Check the saved local setup.")
    parser.parse_args(argv)
    problems = 0
    try:
        profile = profile_loader()
        emit("[green]Profile found.[/green]")
    except Exception:
        emit("[red]Default profile is missing or invalid. Run 'speech-to-speech setup'.[/red]")
        return 1
    snapshot = inspect()
    if not snapshot.supported:
        problems += 1
        emit("[red]A native Apple Silicon Python environment is required.[/red]")
    if not snapshot.audio_input_count or not snapshot.audio_output_count:
        problems += 1
        emit(
            "[yellow]Audio is unavailable. Check System Settings → Privacy & Security → Microphone and Sound devices.[/yellow]"
        )
    getter = credential_getter or MacOSKeychain().get
    for reference in profile.credentials.values():
        try:
            getter(reference)
        except Exception:
            problems += 1
            emit("[red]A saved endpoint credential is missing from macOS Keychain; rerun setup.[/red]")
    configured_url = profile.pipeline.get("responses_api_base_url")
    if configured_url and configured_url not in endpoint_urls():
        problems += 1
        emit("[red]The configured local endpoint is not currently available.[/red]")
    if profile.managed_services:
        checker = runtime_present or _managed_runtime_present
        if not checker():
            problems += 1
            emit("[red]The managed llama.cpp runtime is missing; rerun setup.[/red]")
    if problems:
        emit(f"[yellow]Doctor found {problems} issue(s).[/yellow]")
        return 1
    emit("[green]Local setup is ready.[/green]")
    return 0


def _managed_runtime_present() -> bool:
    root = Path.home() / "Library" / "Application Support" / "speech-to-speech" / "runtime"
    return any(root.glob("*/**/llama-server"))
