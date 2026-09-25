"""Space updates synchronize managed source without removing unrelated files."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub.hf_api
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi


def test_space_update_removes_source_deleted_in_the_next_revision(monkeypatch, tmp_path):
    source_script = Path(__file__).resolve().parents[2] / "scripts" / "prepare_eval_space.py"
    spec = importlib.util.spec_from_file_location("review_prepare_eval_space", source_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repo = tmp_path / "checkout"
    repo.mkdir()
    retired = "src/speech_to_speech/retired.py"
    for name in ["README.md", "LICENSE", "MANIFEST.in", "pyproject.toml", "Dockerfile.eval", retired]:
        p = repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# test source\n")

    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    git("init")
    git("add", ".")
    git("-c", "user.name=Review", "-c", "user.email=review@localhost", "commit", "-m", "first revision")
    remote = {".gitattributes", "notes/keep.md"}
    api = HfApi(token=False)

    def commit(*, operations, **kwargs):
        for op in operations:
            if isinstance(op, CommitOperationAdd):
                remote.add(op.path_in_repo)
            elif isinstance(op, CommitOperationDelete):
                remote.discard(op.path_in_repo)
        return SimpleNamespace()

    monkeypatch.setattr(module, "ROOT", repo)
    monkeypatch.setattr(module, "HfApi", lambda: api)
    monkeypatch.setattr(api, "whoami", lambda: {"name": "review-user"})
    monkeypatch.setattr(api, "create_repo", lambda **kwargs: None)
    monkeypatch.setattr(api, "repo_info", lambda *args, **kwargs: SimpleNamespace(private=True))
    monkeypatch.setattr(api, "list_repo_files", lambda *args, **kwargs: list(remote))
    monkeypatch.setattr(api, "create_commit", commit)
    monkeypatch.setattr(huggingface_hub.hf_api, "is_xet_available", lambda: False, raising=False)
    monkeypatch.setattr(sys, "argv", [str(source_script)])
    module.main()  # Uses the actual HfApi.upload_folder operation builder; no network.
    assert retired in remote
    (repo / retired).unlink()
    replacement = "src/speech_to_speech/current.py"
    (repo / replacement).write_text("# replacement source\n")
    git("add", ".")
    git("-c", "user.name=Review", "-c", "user.email=review@localhost", "commit", "-m", "remove source")
    module.main()
    assert retired not in remote, "Deleted source remains in the Space and is copied into the next Docker image"

    assert replacement in remote
    assert {".gitattributes", "notes/keep.md", "README.md", "Dockerfile", "source-revision.txt"} <= remote
