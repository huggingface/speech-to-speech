from pathlib import Path
from types import SimpleNamespace

from speech_to_speech.setup.catalog import curated_catalog
from speech_to_speech.setup.system import (
    GIB,
    CachedModel,
    ModelChoice,
    estimate_required_space,
    inspect_system,
    scan_model_caches,
)


def test_inspect_system_accepts_native_apple_silicon_and_records_memory_tier(tmp_path):
    snapshot = inspect_system(
        system="Darwin",
        machine="arm64",
        sysctl=lambda name: {"sysctl.proc_translated": 0, "hw.memsize": 24 * GIB}[name],
        disk_usage=lambda _: SimpleNamespace(free=30 * GIB),
        audio_query=lambda: [{"max_input_channels": 1, "max_output_channels": 0}],
        install_path=tmp_path,
    )

    assert snapshot.supported is True
    assert snapshot.memory_bytes == 24 * GIB
    assert snapshot.memory_tier == "gemma"
    assert snapshot.audio_input_count == 1


def test_inspect_system_rejects_rosetta():
    snapshot = inspect_system(
        system="Darwin",
        machine="x86_64",
        sysctl=lambda name: {"sysctl.proc_translated": 1, "hw.memsize": 16 * GIB}[name],
        disk_usage=lambda _: SimpleNamespace(free=10 * GIB),
        audio_query=lambda: [],
    )

    assert snapshot.supported is False
    assert any("Rosetta" in diagnostic for diagnostic in snapshot.diagnostics)


def test_curated_defaults_use_small_models_and_memory_appropriate_llm():
    low = curated_catalog(16 * GIB)
    high = curated_catalog(24 * GIB)

    assert low.defaults["stt"].model_id == "mlx-community/parakeet-tdt-0.6b-v3"
    assert low.defaults["tts"].model_id == "hexgrad/Kokoro-82M"
    assert low.defaults["llm"].model_id == "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    assert high.defaults["llm"].model_id == "ggml-org/gemma-4-12B-it-GGUF"
    assert high.defaults["llm"].variant == "Q4_0"


def test_scan_model_caches_normalizes_hugging_face_and_custom_directories(tmp_path):
    hf_repo = SimpleNamespace(repo_id="org/model", size_on_disk=123, repo_path=tmp_path / "hf-model")
    custom = tmp_path / "custom"
    (custom / "models--another--model" / "snapshots" / "abc").mkdir(parents=True)
    (custom / "models--another--model" / "blob.bin").write_bytes(b"abcd")

    models, diagnostics = scan_model_caches(
        custom_directories=[custom],
        scan_hf=lambda: SimpleNamespace(repos=[hf_repo]),
    )

    assert {(model.model_id, model.size_bytes) for model in models} == {
        ("org/model", 123),
        ("another/model", 4),
    }
    assert diagnostics == []


def test_scan_model_caches_includes_completed_installer_managed_models(tmp_path):
    managed = tmp_path / "models"
    model = managed / "mlx-community--parakeet-tdt-0.6b-v3"
    model.mkdir(parents=True)
    (model / ".complete").touch()
    (model / "weights.bin").write_bytes(b"weights")

    models, diagnostics = scan_model_caches(scan_hf=lambda: SimpleNamespace(repos=[]), managed_directory=managed)

    assert [(item.model_id, item.size_bytes) for item in models] == [("mlx-community/parakeet-tdt-0.6b-v3", 7)]
    assert diagnostics == []


def test_disk_estimate_uses_missing_bytes_plus_exact_safety_reserve():
    choice = ModelChoice("llm", "Example", "org/model", "mlx", 10 * GIB)
    cached = [CachedModel("org/model", 4 * GIB, Path("/cache/model"))]

    estimate = estimate_required_space([choice], cached, free_bytes=8 * GIB)

    assert estimate.missing_bytes == 6 * GIB
    assert estimate.reserve_bytes == 2 * GIB
    assert estimate.required_bytes == 8 * GIB
    assert estimate.can_install is True


def test_disk_estimate_blocks_shortfall_unless_forced():
    choice = ModelChoice("llm", "Large", "org/large", "mlx", 20 * GIB)

    blocked = estimate_required_space([choice], [], free_bytes=23 * GIB)
    forced = estimate_required_space([choice], [], free_bytes=23 * GIB, force=True)

    assert blocked.reserve_bytes == 4 * GIB
    assert blocked.can_install is False
    assert forced.can_install is True
