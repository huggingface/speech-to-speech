from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from speech_to_speech.setup.system import ModelChoice

Progress = Callable[[str], None]

LLAMA_CPP_TAG = "b10760"
LLAMA_CPP_MACOS_ARM64_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/b10760/llama-b10760-bin-macos-arm64.tar.gz"
)
LLAMA_CPP_MACOS_ARM64_SHA256 = "4451e74e6f6d76838b6a10be8c0224d74f0fe2b2c9c23e9a4ff46c33855dd782"


def _snapshot_download(**kwargs: Any) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def _file_download(url: str, destination: Path) -> None:
    urllib.request.urlretrieve(url, destination)


class AssetInstaller:
    def __init__(
        self,
        root: Path,
        *,
        snapshot_download: Callable[..., str] = _snapshot_download,
        file_download: Callable[[str, Path], None] = _file_download,
    ) -> None:
        self.root = root
        self._snapshot_download = snapshot_download
        self._file_download = file_download

    def install(self, choice: ModelChoice, progress: Progress | None = None) -> Path:
        target = self.root / "models" / choice.model_id.replace("/", "--")
        marker = target / ".complete"
        if marker.is_file():
            if progress:
                progress(f"Using cached {choice.label}")
            return target
        target.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(f"Downloading {choice.label}")
        self._snapshot_download(
            repo_id=choice.model_id,
            local_dir=str(target),
            allow_patterns=choice.allow_patterns or None,
            resume_download=True,
        )
        marker.touch(mode=0o600)
        return target

    def install_runtime(self, url: str, sha256: str, progress: Progress | None = None) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="runtime-download-", dir=self.root) as temporary_name:
            temporary = Path(temporary_name)
            archive = temporary / "runtime.tar.gz"
            if progress:
                progress("Downloading the pinned llama.cpp runtime")
            self._file_download(url, archive)
            actual = hashlib.sha256(archive.read_bytes()).hexdigest()
            if actual != sha256:
                raise ValueError("llama.cpp archive checksum did not match the pinned release")
            extracted = temporary / "extracted"
            extracted.mkdir()
            with tarfile.open(archive, "r:gz") as package:
                self._safe_extract(package, extracted)
            destination = self.root / "runtime" / sha256[:12]
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged = destination.parent / f".{destination.name}.new"
            if staged.exists():
                shutil.rmtree(staged)
            shutil.move(str(extracted), staged)
            if destination.exists():
                shutil.rmtree(staged)
            else:
                os.replace(staged, destination)
            return destination

    @staticmethod
    def _safe_extract(package: tarfile.TarFile, destination: Path) -> None:
        root = destination.resolve()
        for member in package.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError("llama.cpp archive contains an unsafe path")
        package.extractall(destination, filter="data")
