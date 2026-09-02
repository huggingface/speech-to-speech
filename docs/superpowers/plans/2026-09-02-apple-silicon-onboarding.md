# Apple Silicon Onboarding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible one-line Apple Silicon installer and a rerunnable setup wizard that discovers local models/endpoints, checks resources, installs a curated Parakeet/Gemma-or-Qwen/Kokoro stack, validates audio, and launches a saved profile.

**Architecture:** Keep the release shell bootstrap thin and put all product logic in a new `speech_to_speech.setup` package. The setup package exposes independently testable system discovery, endpoint probing, Keychain, catalog, profile, managed-runtime, and wizard services; the existing CLI composes those services while preserving explicit legacy invocation behavior.

**Tech Stack:** Python 3.11, Rich, httpx, huggingface-hub, macOS `lsof`/`security`, uv, pytest.

## Global Constraints

- Apple Silicon macOS is the only supported platform for the bootstrap.
- The installed package, uv, Python, dependency constraints, llama.cpp archive, and checksums are version-pinned.
- Parakeet and Kokoro are the local STT/TTS defaults; a discovered endpoint is the preferred LLM.
- With no endpoint, Gemma 4 12B Q4_0 is preferred at 24 GiB unified memory or above; Qwen3 4B MLX 4-bit is preferred below it.
- Profiles and logs must never contain raw credentials or authorization headers.
- Existing explicit CLI arguments and positional JSON configuration keep their current behavior.
- No local listener receives an inference request unless the user selects it.

---

### Task 1: CLI and profile contract

**Files:**
- Create: `src/speech_to_speech/setup/models.py`
- Create: `src/speech_to_speech/setup/profiles.py`
- Modify: `src/speech_to_speech/cli.py`
- Test: `tests/test_setup_profiles.py`
- Test: `tests/test_cli_defaults.py`

**Interfaces:**
- Produces `SetupProfile`, `CredentialRef`, `ManagedService`, `load_profile()`, and `save_profile()`.
- Adds CLI commands `setup` and `doctor`; plain `local` uses the saved default profile only when no explicit pipeline arguments or positional JSON are provided.

- [ ] Write failing tests proving profile JSON round-trips without secrets, writes atomically, plain `local` selects it, and explicit legacy arguments bypass it.
- [ ] Run `pytest tests/test_setup_profiles.py tests/test_cli_defaults.py -q` and confirm the new tests fail.
- [ ] Implement schema-versioned dataclasses and atomic profile storage at `~/Library/Application Support/speech-to-speech/profiles/default.json`.
- [ ] Extend command parsing with `setup` and `doctor`, deferring command imports so `--help` remains lightweight.
- [ ] Run the focused tests and commit the passing slice.

### Task 2: System, cache, catalog, and disk discovery

**Files:**
- Create: `src/speech_to_speech/setup/system.py`
- Create: `src/speech_to_speech/setup/catalog.py`
- Test: `tests/test_setup_system.py`

**Interfaces:**
- Produces `SystemSnapshot`, `ModelChoice`, `CachedModel`, `DiskEstimate`, `inspect_system()`, `scan_model_caches()`, `curated_catalog()`, and `estimate_required_space()`.
- `DiskEstimate.can_install` is true when free bytes cover missing bytes plus `max(2 GiB, missing_bytes * 0.20)`.

- [ ] Write failing tests for Apple Silicon detection, Rosetta rejection, unified-memory tiers, cache normalization, custom model directories, exact safety reserve, and `--force` behavior.
- [ ] Run the focused tests and verify failure.
- [ ] Implement read-only system inspection using `platform`, `sysctl`, `shutil.disk_usage`, `sounddevice`, and `huggingface_hub.scan_cache_dir()` with errors converted into diagnostics.
- [ ] Define the curated defaults with exact backend/model identifiers and catalog fallback sizes.
- [ ] Run tests and commit.

### Task 3: Engine-independent endpoint discovery and Keychain

**Files:**
- Create: `src/speech_to_speech/setup/endpoints.py`
- Create: `src/speech_to_speech/setup/keychain.py`
- Test: `tests/test_setup_endpoints.py`
- Test: `tests/test_setup_keychain.py`

**Interfaces:**
- Produces `EndpointCandidate`, `EndpointCapabilities`, `discover_endpoints()`, `validate_selected_endpoint()`, and `MacOSKeychain`.
- Discovery performs only GET/HEAD/OPTIONS; validation performs the stage-specific minimal POST after selection.

- [ ] Add fake HTTP-server tests for arbitrary ports, `/v1/models`, all four supported inference routes, 401/403, malformed JSON, HTTP/HTTPS failure, timeout, and no unselected POSTs.
- [ ] Add subprocess-adapter tests proving keys use hidden input, Keychain account names derive from a SHA-256 URL digest, and secrets never appear in errors.
- [ ] Implement loopback listener enumeration through `/usr/sbin/lsof`, concurrent bounded probes, capability classification, and selected endpoint validation.
- [ ] Implement `security add-generic-password` / `find-generic-password` / `delete-generic-password` behind an injectable runner.
- [ ] Run tests and commit.

### Task 4: Resumable assets and managed llama.cpp lifecycle

**Files:**
- Create: `src/speech_to_speech/setup/assets.py`
- Create: `src/speech_to_speech/setup/services.py`
- Test: `tests/test_setup_assets.py`
- Test: `tests/test_setup_services.py`

**Interfaces:**
- Produces `AssetInstaller.install(choice, progress)`, `ManagedServiceRunner.start(spec)`, and `ManagedProcess.stop()`.
- The Gemma service uses an installer-owned, checksum-verified llama.cpp binary and a dynamically selected loopback port.

- [ ] Write failing tests for cache reuse, resumed Hub downloads, checksum mismatch, atomic runtime extraction, dynamic port conflicts, readiness, crash reporting, and stopping only owned processes.
- [ ] Implement Hugging Face snapshot/file downloads with explicit allow-patterns and Rich-compatible progress callbacks.
- [ ] Implement pinned llama.cpp archive installation and on-demand `llama-server` startup with `-hf ggml-org/gemma-4-12B-it-GGUF:Q4_0 -c 16384 -np 1 -fa on`.
- [ ] Run tests and commit.

### Task 5: Interactive wizard, audio validation, profile launch, and doctor

**Files:**
- Create: `src/speech_to_speech/setup/wizard.py`
- Create: `src/speech_to_speech/setup/doctor.py`
- Modify: `src/speech_to_speech/cli.py`
- Test: `tests/test_setup_wizard.py`
- Test: `tests/test_setup_doctor.py`

**Interfaces:**
- Produces `run_setup(force=False) -> int`, `run_doctor() -> int`, and `run_profiled_local(profile) -> int`.
- The three unconditional prompts select STT, LLM, and TTS; auth, custom directory, low-memory override, audio remediation, download confirmation, and launch are conditional.

- [ ] Write scripted-console tests for endpoint-first LLM selection, 24 GiB Gemma, low-memory Qwen fallback, Parakeet/Kokoro defaults, conditional auth, combined size summary, cancellation, guided audio, and launch prompt.
- [ ] Write doctor tests for healthy/degraded profiles, missing cache/runtime/Keychain/endpoint/audio, and redaction.
- [ ] Implement the Rich wizard and transactional flow: discover, choose, estimate, confirm, install, validate, audio-check, save, optionally launch.
- [ ] Implement profile resolution of Keychain credentials and managed-service URLs entirely in memory.
- [ ] Implement `doctor` and actionable macOS Privacy & Security guidance.
- [ ] Run focused tests and commit.

### Task 6: Versioned bootstrap and first-run documentation

**Files:**
- Create: `scripts/install-macos.sh`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Test: `tests/test_install_macos_script.py`

**Interfaces:**
- The script installs the release matching its embedded `SPEECH_TO_SPEECH_VERSION`, using pinned `UV_VERSION`, `PYTHON_VERSION`, and SHA-256 values.
- The release process updates those constants and the README tag together.

- [ ] Write shell-harness tests for Darwin/arm64 gating, Rosetta, 4 GiB bootstrap disk floor, checksum failure, lock-derived constraints, exact package installation, idempotency, and redacted log paths.
- [ ] Implement the bootstrap with `set -eu`, temporary-directory cleanup traps, checksum verification, tagged `pyproject.toml`/`uv.lock` downloads, `uv export --frozen`, constrained `uv tool install`, and direct wizard invocation.
- [ ] Document the one-liner, inspect-first alternative, expected model sizes, rerunning setup, doctor, profile path, and uninstall boundaries.
- [ ] Update release instructions so installer pins are treated as release metadata.
- [ ] Run tests and shell syntax validation, then commit.

### Task 7: Integration and regression verification

**Files:**
- Modify: `.github/workflows/ci.yml`
- Test: `tests/test_setup_integration.py`

**Interfaces:**
- Adds a macOS arm64 environment-install smoke job without production-sized model downloads.

- [ ] Add an integration test combining fake cache, fake endpoints, fake Keychain, fake audio, profile save, managed service resolution, and launch configuration.
- [ ] Add CI coverage for the bootstrap environment-only test hook and `speech-to-speech setup --help` / `doctor --help`.
- [ ] Run `pytest -q`, `ruff check .`, `mypy src/speech_to_speech`, and `git diff --check`.
- [ ] Manually inspect the final diff for credential leakage, destructive behavior, build artifacts, and release-scope violations.
- [ ] Commit the verified implementation.
