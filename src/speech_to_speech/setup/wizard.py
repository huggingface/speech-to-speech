from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from rich.console import Console
from rich.prompt import Confirm, IntPrompt

from speech_to_speech.setup.assets import (
    LLAMA_CPP_MACOS_ARM64_SHA256,
    LLAMA_CPP_MACOS_ARM64_URL,
    AssetInstaller,
)
from speech_to_speech.setup.catalog import curated_catalog
from speech_to_speech.setup.endpoints import EndpointCandidate, discover_endpoints, validate_selected_endpoint
from speech_to_speech.setup.keychain import MacOSKeychain
from speech_to_speech.setup.models import CredentialRef, ManagedService, SetupProfile
from speech_to_speech.setup.profiles import save_profile
from speech_to_speech.setup.system import (
    CachedModel,
    ModelChoice,
    SystemSnapshot,
    estimate_required_space,
    inspect_system,
    scan_model_caches,
)


class WizardIO(Protocol):
    def choose(self, prompt: str, options: Sequence[str], default: int = 0) -> int: ...

    def confirm(self, prompt: str, default: bool = True) -> bool: ...

    def print(self, message: object) -> None: ...


class RichWizardIO:
    def __init__(self, console: Console | None = None, log_path: Path | None = None) -> None:
        self.console = console or Console()
        self.log_path = log_path or Path.home() / "Library" / "Logs" / "speech-to-speech" / "setup.log"

    def _record(self, message: str) -> None:
        safe = re.sub(r"(?i)(authorization|api[_ -]?key|secret|token)(\s*[=:]\s*)\S+", r"\1\2[REDACTED]", message)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(re.sub(r"\[[^]]+\]", "", safe) + "\n")

    def choose(self, prompt: str, options: Sequence[str], default: int = 0) -> int:
        self.console.print(f"\n[bold]{prompt}[/bold]")
        for index, option in enumerate(options, 1):
            self.console.print(f"  {index}. {option}")
        return (
            IntPrompt.ask("Choose", choices=[str(index) for index in range(1, len(options) + 1)], default=default + 1)
            - 1
        )

    def confirm(self, prompt: str, default: bool = True) -> bool:
        return Confirm.ask(prompt, default=default)

    def print(self, message: object) -> None:
        self.console.print(message)
        self._record(str(message))


class SetupWizard:
    def __init__(
        self,
        *,
        io: WizardIO,
        inspect: Callable[[], SystemSnapshot],
        discover: Callable[[], list[EndpointCandidate]],
        scan_caches: Callable[[Sequence[Path]], tuple[list[CachedModel], list[str]]],
        install: Callable[[ModelChoice], Any],
        install_runtime: Callable[[], Any] | None = None,
        save: Callable[[SetupProfile], Path],
        validate_endpoint: Callable[..., bool] = validate_selected_endpoint,
        keychain: MacOSKeychain | None = None,
        custom_directories: Sequence[Path] = (),
        force: bool = False,
        offer_launch: bool = True,
        audio_check: Callable[[WizardIO], bool] | None = None,
    ) -> None:
        self.io = io
        self.inspect = inspect
        self.discover = discover
        self.scan_caches = scan_caches
        self.install = install
        self.install_runtime = install_runtime
        self.save = save
        self.validate_endpoint = validate_endpoint
        self.keychain = keychain or MacOSKeychain()
        self.custom_directories = custom_directories
        self.force = force
        self.offer_launch = offer_launch
        self.audio_check = audio_check

    def run(self) -> int:
        system = self.inspect()
        for diagnostic in system.diagnostics:
            self.io.print(f"[yellow]{diagnostic}[/yellow]")
        if not system.supported:
            self.io.print("[red]Setup stopped: native Apple Silicon macOS is required.[/red]")
            return 1

        cached, cache_diagnostics = self.scan_caches(self.custom_directories)
        for diagnostic in cache_diagnostics:
            self.io.print(f"[yellow]{diagnostic}[/yellow]")
        endpoints = self.discover()
        catalog = curated_catalog(system.memory_bytes)

        stt_endpoint, stt = self._choose_speech_endpoint(
            "Which speech-to-text model?", catalog.choices["stt"], endpoints, "transcriptions"
        )
        llm_endpoint, llm = self._choose_llm(endpoints, catalog.choices["llm"])
        tts_endpoint, tts = self._choose_speech_endpoint(
            "Which text-to-speech model?", catalog.choices["tts"], endpoints, "speech"
        )
        install_choices = [
            *((stt,) if not stt_endpoint else ()),
            *((llm,) if not llm_endpoint else ()),
            *((tts,) if not tts_endpoint else ()),
        ]
        estimate = estimate_required_space(install_choices, cached, free_bytes=system.free_bytes, force=self.force)
        self.io.print(
            f"Download: {_format_bytes(estimate.missing_bytes)}; safety reserve: "
            f"{_format_bytes(estimate.reserve_bytes)}; available: {_format_bytes(estimate.free_bytes)}."
        )
        if not estimate.can_install:
            self.io.print("[red]Not enough free space. Free space or rerun with --force.[/red]")
            return 1
        if not self.io.confirm("Download/install these models?", default=True):
            self.io.print("Setup cancelled; no profile was changed.")
            return 1

        profile = self._profile(stt, llm, tts, stt_endpoint, llm_endpoint, tts_endpoint)
        selected_endpoints = (
            ("stt", stt_endpoint, stt.model_id),
            ("llm", llm_endpoint, llm.model_id),
            ("tts", tts_endpoint, tts.model_id),
        )
        for stage, endpoint, model_id in selected_endpoints:
            if endpoint is None:
                continue
            api_key = self.keychain.get(profile.credentials[stage]) if stage in profile.credentials else None
            validation_stage = (
                "responses"
                if stage == "llm" and not endpoint.capabilities.chat_completions and endpoint.capabilities.responses
                else stage
            )
            if not self.validate_endpoint(endpoint.base_url, stage=validation_stage, model=model_id, api_key=api_key):
                self.io.print(f"[red]The selected {stage.upper()} endpoint did not pass validation.[/red]")
                return 1
        for choice in install_choices:
            installed = self.install(choice)
            if choice is llm and llm.runtime == "llama.cpp" and isinstance(installed, Path):
                matches = sorted(installed.rglob("*Q4_0*.gguf"))
                if not matches:
                    raise FileNotFoundError(f"The downloaded Gemma Q4_0 GGUF was not found under {installed}.")
                profile.managed_services[0] = replace(profile.managed_services[0], model_path=str(matches[0]))
        if llm.runtime == "llama.cpp" and self.install_runtime:
            self.install_runtime()
        path = self.save(profile)
        if not system.audio_input_count or not system.audio_output_count:
            self.io.print(
                "Audio device access is incomplete. Open System Settings → Privacy & Security → Microphone, "
                "then check Sound input and output devices."
            )
        else:
            self.io.print("Microphone and speaker devices found.")
            if self.audio_check:
                try:
                    audio_ok = self.audio_check(self.io)
                except Exception as error:
                    audio_ok = False
                    self.io.print(f"[yellow]Audio check could not run: {error}[/yellow]")
                if not audio_ok:
                    self.io.print(
                        "[yellow]Audio check was not confirmed. Check System Settings → Privacy & Security → Microphone.[/yellow]"
                    )
        self.io.print(f"Saved profile: {path}")
        if self.offer_launch and self.io.confirm("Launch speech-to-speech now?", default=True):
            return run_profiled_local(profile)
        return 0

    def _choose_model(self, prompt: str, choices: Sequence[ModelChoice]) -> ModelChoice:
        index = self.io.choose(prompt, [choice.label for choice in choices], default=0)
        return choices[index]

    def _choose_speech_endpoint(
        self,
        prompt: str,
        choices: Sequence[ModelChoice],
        endpoints: Sequence[EndpointCandidate],
        capability: str,
    ) -> tuple[EndpointCandidate | None, ModelChoice]:
        endpoint_options = [
            (endpoint, model)
            for endpoint in endpoints
            if endpoint.requires_auth or getattr(endpoint.capabilities, capability)
            for model in (endpoint.models or ("default",))
        ]
        labels = [choice.label for choice in choices]
        labels.extend(f"Already running: {model} at {endpoint.base_url}" for endpoint, model in endpoint_options)
        selected = self.io.choose(prompt, labels, default=0)
        if selected < len(choices):
            return None, choices[selected]
        endpoint, model_id = endpoint_options[selected - len(choices)]
        return endpoint, ModelChoice("endpoint", model_id, model_id, "endpoint", 0)

    def _choose_llm(
        self, endpoints: Sequence[EndpointCandidate], choices: Sequence[ModelChoice]
    ) -> tuple[EndpointCandidate | None, ModelChoice]:
        endpoint_options = [
            (endpoint, model)
            for endpoint in endpoints
            if endpoint.requires_auth or endpoint.capabilities.chat_completions or endpoint.capabilities.responses
            for model in (endpoint.models or ("default",))
        ]
        labels = [f"Already running: {model} at {endpoint.base_url}" for endpoint, model in endpoint_options]
        labels.extend(choice.label for choice in choices)
        selected = self.io.choose("Which language model?", labels, default=0)
        if selected < len(endpoint_options):
            endpoint, model_id = endpoint_options[selected]
            return endpoint, ModelChoice("llm", model_id, model_id, "endpoint", 0)
        return None, choices[selected - len(endpoint_options)]

    def _profile(
        self,
        stt: ModelChoice,
        llm: ModelChoice,
        tts: ModelChoice,
        stt_endpoint: EndpointCandidate | None,
        llm_endpoint: EndpointCandidate | None,
        tts_endpoint: EndpointCandidate | None,
    ) -> SetupProfile:
        pipeline: dict[str, Any] = {
            "stt": "parakeet-tdt",
            "parakeet_tdt_model_name": stt.model_id,
            "tts": "kokoro",
            "kokoro_model_name": tts.model_id,
            "model_name": llm.model_id,
        }
        profile = SetupProfile(pipeline=pipeline)
        if stt_endpoint:
            pipeline["stt"] = "openai"
            pipeline["openai_stt_base_url"] = stt_endpoint.base_url
            pipeline["openai_stt_model"] = stt.model_id
        if tts_endpoint:
            pipeline["tts"] = "openai"
            pipeline["openai_tts_base_url"] = tts_endpoint.base_url
            pipeline["openai_tts_model"] = tts.model_id
        if llm_endpoint:
            pipeline["llm_backend"] = (
                "chat-completions" if llm_endpoint.capabilities.chat_completions else "responses-api"
            )
            pipeline["responses_api_base_url"] = llm_endpoint.base_url
        elif llm.runtime == "llama.cpp":
            pipeline["llm_backend"] = "chat-completions"
            profile.managed_services.append(ManagedService("llm", f"{llm.model_id}:{llm.variant}", "llama.cpp"))
        else:
            pipeline["llm_backend"] = "mlx-lm"
        stored_by_url: dict[str, CredentialRef] = {}
        for stage, selected in (("stt", stt_endpoint), ("llm", llm_endpoint), ("tts", tts_endpoint)):
            if selected and selected.requires_auth:
                reference = stored_by_url.get(selected.base_url)
                if reference is None:
                    reference = self.keychain.prompt_and_store(selected.base_url)
                    stored_by_url[selected.base_url] = reference
                profile.credentials[stage] = reference
        return profile


def _format_bytes(value: int) -> str:
    return f"{value / 1024**3:.1f} GiB"


def _pipeline_args(config: dict[str, Any]) -> list[str]:
    arguments: list[str] = []
    for name, value in config.items():
        option = f"--{name}"
        if isinstance(value, bool):
            if value:
                arguments.append(option)
        elif value is not None:
            arguments.extend((option, str(value)))
    return arguments


def _find_llama_server() -> Path:
    root = Path.home() / "Library" / "Application Support" / "speech-to-speech" / "runtime"
    matches = sorted(root.glob("*/**/llama-server"), reverse=True)
    if not matches:
        raise FileNotFoundError("Managed llama.cpp is missing; rerun 'speech-to-speech setup'.")
    return matches[0]


def guided_audio_check(io: WizardIO) -> bool:
    import numpy as np
    import sounddevice

    sample_rate = 16000
    io.print("Audio check: say a short phrase after the tone; it will be played back once.")
    tone_time = np.arange(int(sample_rate * 0.2)) / sample_rate
    tone = (0.08 * np.sin(2 * np.pi * 660 * tone_time)).astype(np.float32)
    sounddevice.play(tone, sample_rate, blocking=True)
    recording = sounddevice.rec(int(sample_rate * 2), samplerate=sample_rate, channels=1, dtype="float32")
    sounddevice.wait()
    sounddevice.play(recording, sample_rate, blocking=True)
    return io.confirm("Did you hear your recorded phrase?", default=True)


def run_profiled_local(
    profile: SetupProfile,
    *,
    credential_getter: Callable[[Any], str] | None = None,
    pipeline_runner: Callable[[str, Sequence[str]], Any] | None = None,
    service_runner_factory: Callable[[], Any] | None = None,
) -> int:
    from speech_to_speech.s2s_pipeline import run_pipeline_command
    from speech_to_speech.setup.services import ManagedServiceRunner

    config = deepcopy(profile.pipeline)
    getter = credential_getter or MacOSKeychain().get
    credential_fields = {
        "stt": "openai_stt_api_key",
        "llm": "responses_api_api_key",
        "tts": "openai_tts_api_key",
    }
    for stage, reference in profile.credentials.items():
        if stage in credential_fields:
            config[credential_fields[stage]] = getter(reference)
    processes = []
    try:
        for service in profile.managed_services:
            runner = (
                service_runner_factory()
                if service_runner_factory
                else ManagedServiceRunner(llama_server=_find_llama_server())
            )
            process = runner.start(service)
            processes.append(process)
            if service.kind == "llm":
                config["responses_api_base_url"] = process.base_url
        (pipeline_runner or run_pipeline_command)("local", _pipeline_args(config))
    finally:
        for process in reversed(processes):
            process.stop()
    return 0


def run_setup(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="speech-to-speech setup", description="Configure local Apple Silicon models.")
    parser.add_argument("--force", action="store_true", help="Continue despite the recommended disk reserve.")
    parser.add_argument("--custom-model-dir", action="append", type=Path, default=[])
    parser.add_argument("--no-launch", action="store_true", help="Do not offer to launch after setup.")
    args = parser.parse_args(argv)
    root = Path.home() / "Library" / "Application Support" / "speech-to-speech"
    installer = AssetInstaller(root)
    wizard = SetupWizard(
        io=RichWizardIO(),
        inspect=inspect_system,
        discover=discover_endpoints,
        scan_caches=lambda custom: scan_model_caches(custom, managed_directory=root / "models"),
        install=installer.install,
        install_runtime=lambda: installer.install_runtime(LLAMA_CPP_MACOS_ARM64_URL, LLAMA_CPP_MACOS_ARM64_SHA256),
        save=save_profile,
        custom_directories=args.custom_model_dir,
        force=args.force,
        offer_launch=not args.no_launch,
        audio_check=guided_audio_check,
    )
    return wizard.run()
