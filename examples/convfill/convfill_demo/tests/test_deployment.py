"""Deployment destinations and credentials must be explicitly supplied."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def uploader():
    path = Path(__file__).resolve().parents[2] / "deploy" / "hf_space.py"
    spec = importlib.util.spec_from_file_location("convfill_space_uploader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_space_destination_is_required(monkeypatch):
    module = uploader()
    monkeypatch.setattr(sys, "argv", ["upload"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


@pytest.mark.parametrize("private", [True, False])
def test_hub_token_is_never_an_implicit_inference_key(monkeypatch, private):
    module = uploader()
    secrets = []
    monkeypatch.setattr(sys, "argv", ["upload", "--repo", "example/private-voice-demo"])
    monkeypatch.setenv("HF_TOKEN", "hub-upload-only")
    for stage in ("REASONER", "TALKER", "STT", "TTS"):
        monkeypatch.setenv("CONVFILL_" + stage + "_URL", "https://service.example/v1")
        monkeypatch.delenv("CONVFILL_" + stage + "_API_KEY", raising=False)
    monkeypatch.setenv("CONVFILL_STT_API_KEY", "explicit-inference-key")

    class Api:
        def __init__(self, token):
            assert token == "hub-upload-only"

        def create_repo(self, *args, **kwargs):
            assert kwargs["private"] is True

        def space_info(self, repo):
            return SimpleNamespace(private=private)

        def add_space_variable(self, *args):
            pass

        def add_space_secret(self, *args):
            secrets.append(args)

        def upload_folder(self, **kwargs):
            return SimpleNamespace(oid="test-commit")

    monkeypatch.setattr(module, "HfApi", Api)
    monkeypatch.setattr(module.shutil, "copytree", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.shutil, "copy2", lambda *args, **kwargs: None)
    if private:
        module.main()
        assert secrets == [("example/private-voice-demo", "CONVFILL_STT_API_KEY", "explicit-inference-key")]
    else:
        with pytest.raises(ValueError, match="public Space"):
            module.main()
        assert secrets == []


def test_frontend_upload_uses_only_explicit_session_service_credentials(monkeypatch):
    module = uploader()
    secrets, variables = [], []
    monkeypatch.setattr(sys, "argv", ["upload", "--repo", "example/private-demo", "--frontend-only"])
    monkeypatch.setenv("HF_TOKEN", "hub-upload-only")
    monkeypatch.setenv("LOAD_BALANCER_URL", "https://allocator.example")
    monkeypatch.setenv("LB_HF_TOKEN", "explicit-service-credential")
    monkeypatch.setenv("ALLOW_GUEST_SESSIONS", "1")
    monkeypatch.setenv("LIMIT_ANON_SEC", "300")
    monkeypatch.setenv("USAGE_HASH_SECRET", "explicit-cookie-signing-secret")
    for stage in ("REASONER", "TALKER", "STT", "TTS"):
        monkeypatch.delenv("CONVFILL_" + stage + "_URL", raising=False)
    for name in ("SERPER_API_KEY", "CONVFILL_SEARCH_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    class Api:
        def __init__(self, token):
            assert token == "hub-upload-only"

        def create_repo(self, *args, **kwargs):
            pass

        def space_info(self, repo):
            return SimpleNamespace(private=True)

        def add_space_variable(self, *args):
            variables.append(args)

        def add_space_secret(self, *args):
            secrets.append(args)

        def upload_folder(self, **kwargs):
            assert (Path(kwargs["folder_path"]) / "Dockerfile").exists()
            return SimpleNamespace(oid="test-frontend")

    monkeypatch.setattr(module, "HfApi", Api)
    monkeypatch.setattr(module.shutil, "copytree", lambda *args, **kwargs: None)
    module.main()
    assert secrets == [
        ("example/private-demo", "LOAD_BALANCER_URL", "https://allocator.example"),
        ("example/private-demo", "LB_HF_TOKEN", "explicit-service-credential"),
        ("example/private-demo", "USAGE_HASH_SECRET", "explicit-cookie-signing-secret"),
    ]
    assert ("example/private-demo", "ALLOW_GUEST_SESSIONS", "1") in variables
    assert ("example/private-demo", "LIMIT_ANON_SEC", "300") in variables
    assert not any("CONVFILL_" in args[1] for args in variables)
