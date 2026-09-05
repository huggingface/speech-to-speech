import hashlib
from pathlib import Path

import pytest

from speech_to_speech.setup.assets import AssetInstaller
from speech_to_speech.setup.system import ModelChoice


def test_asset_installer_reuses_complete_model_cache(tmp_path):
    choice = ModelChoice("Parakeet", "org/model", "mlx", 10)
    target = tmp_path / "models" / "org--model"
    target.mkdir(parents=True)
    (target / ".complete").touch()
    calls = []

    installed = AssetInstaller(tmp_path, snapshot_download=lambda **kwargs: calls.append(kwargs)).install(choice)

    assert installed == target
    assert calls == []


def test_asset_installer_uses_resumable_hub_download_and_patterns(tmp_path):
    choice = ModelChoice("Gemma", "ggml-org/gemma", "llama.cpp", 10, allow_patterns=("*Q4_0*",))
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)
        return kwargs["local_dir"]

    installed = AssetInstaller(tmp_path, snapshot_download=download).install(choice)

    assert calls[0]["resume_download"] is True
    assert calls[0]["allow_patterns"] == ("*Q4_0*",)
    assert (installed / ".complete").exists()


def test_runtime_install_rejects_checksum_mismatch(tmp_path):
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"not the expected archive")

    def download(url, destination):
        destination.write_bytes(archive.read_bytes())

    installer = AssetInstaller(tmp_path, file_download=download)
    expected = hashlib.sha256(b"different").hexdigest()

    with pytest.raises(ValueError, match="checksum"):
        installer.install_runtime("https://example.test/llama.tar.gz", expected)

    assert not (tmp_path / "runtime").exists()
