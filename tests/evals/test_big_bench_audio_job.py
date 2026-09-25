"""Exercise the Job boundary without launching an engine or contacting the Hub."""

import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

from speech_to_speech.evals.big_bench_audio import __main__ as cli
from speech_to_speech.evals.big_bench_audio import job


@pytest.fixture(autouse=True)
def isolated_job_environment(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith("S2S_") or name in {"HF_TOKEN", "OPENAI_API_KEY", "JOB_ID"}:
            monkeypatch.delenv(name)
    monkeypatch.setenv("S2S_REPORT_DIR", str(tmp_path / "output"))

    def unexpected_call(*args, **kwargs):
        pytest.fail("Test attempted to launch a process or contact the Hub")

    monkeypatch.setattr(job.subprocess, "run", unexpected_call)
    monkeypatch.setattr(huggingface_hub, "HfApi", unexpected_call)


def capture_launch(monkeypatch, *, returncode=0, log=None):
    captured = {}

    def run(command):
        assert command[:3] == [sys.executable, "-m", "speech_to_speech.evals.big_bench_audio"]
        eval_args, server_args = cli._split_server_args(command[3:])
        args = cli.build_parser().parse_args(eval_args)
        captured.update(args=args, server_args=server_args, api_key=os.environ.get("OPENAI_API_KEY"))
        if log is not None:
            Path(args.spawn_log).write_text(log)
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(job.subprocess, "run", run)
    return captured


def test_default_job_uses_ggml_and_inherits_hf_auth_without_command_line_secrets(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-hub-credential")
    monkeypatch.setenv("JOB_ID", "job-123")
    captured = capture_launch(monkeypatch)

    assert job.main() == 0

    args = captured["args"]
    assert args.spawn and args.subset == "vibe" and args.limit is None
    assert args.label == "job-123"
    assert Path(args.out).name == "job-123.json"
    assert captured["api_key"] == "test-hub-credential"
    assert captured["server_args"] == [
        "--stt",
        "parakeet-tdt",
        "--tts",
        "qwen3",
        "--qwen3_tts_backend",
        "ggml",
        "--llm_backend",
        "chat-completions",
        "--model_name",
        "Qwen/Qwen3.5-9B:together",
        "--responses_api_base_url",
        "https://router.huggingface.co/v1",
    ]
    assert "test-hub-credential" not in str(vars(args)) + str(captured["server_args"])


def test_environment_overrides_preserve_prompt_quotes_and_literal_shell_characters(monkeypatch, tmp_path):
    instructions = 'Explain "why", then say Final answer: X. Keep $HOME and $(echo example) literal.'
    settings = {
        "S2S_LIMIT": "12",
        "S2S_SUBSET": str(tmp_path / "custom subset.json"),
        "S2S_LABEL": "experiment/custom prompt",
        "S2S_LLM_MODEL": "example/model:provider",
        "S2S_LLM_BASE_URL": "https://example.test/v1",
        "S2S_COMPARE": "me/results:reports/baseline.json",
        "S2S_PUSH_TO_HUB": "me/results",
        "S2S_EVAL_ARGS": shlex.join(["--instructions", instructions, "--silence-ms", "1200"]),
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    captured = capture_launch(monkeypatch)

    assert job.main() == 0

    args = captured["args"]
    assert args.limit == 12 and args.subset == settings["S2S_SUBSET"]
    assert args.label == settings["S2S_LABEL"]
    assert args.compare == settings["S2S_COMPARE"] and args.push_to_hub == "me/results"
    assert Path(args.out).name == "experiment-custom-prompt.json"
    assert args.instructions == instructions and args.silence_ms == 1200
    assert cli._runner_config(args, cli.DEFAULT_URL).instructions == instructions
    assert captured["server_args"][-4:] == [
        "--model_name",
        "example/model:provider",
        "--responses_api_base_url",
        "https://example.test/v1",
    ]


def test_full_server_override_keeps_provider_settings_and_existing_openai_key(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-hub-credential")
    monkeypatch.setenv("OPENAI_API_KEY", "test-provider-credential")
    monkeypatch.setenv("S2S_LLM_MODEL", "ignored/model")
    monkeypatch.setenv("S2S_LLM_BASE_URL", "https://ignored.test/v1")
    server_args = [
        "--stt",
        "parakeet-tdt",
        "--tts",
        "qwen3",
        "--qwen3_tts_backend",
        "ggml",
        "--llm_backend",
        "chat-completions",
        "--model_name",
        "Qwen/Qwen3.8-27B:cerebras",
        "--responses_api_base_url",
        "https://router.huggingface.co/v1",
        "--responses_api_reasoning_effort",
        "none",
    ]
    monkeypatch.setenv("S2S_SERVE_ARGS", shlex.join(server_args))
    captured = capture_launch(monkeypatch)

    assert job.main() == 0

    assert captured["server_args"] == server_args
    assert captured["api_key"] == "test-provider-credential"


@pytest.mark.parametrize("returncode", [0, 1])
def test_job_redacts_logs_before_upload_and_preserves_runner_exit_status(monkeypatch, capsys, returncode):
    secrets = ["test-hub-credential", "test-provider-credential", "test-realtime-credential"]
    for name, value in zip(("HF_TOKEN", "OPENAI_API_KEY", "S2S_API_KEY"), secrets):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("S2S_PUSH_TO_HUB", "me/results")
    captured = capture_launch(monkeypatch, returncode=returncode, log="auth: " + " ".join(secrets))
    uploads = []
    creates = []

    class FakeApi:
        def create_repo(self, **kwargs):
            creates.append(kwargs)

        def upload_file(self, **kwargs):
            uploads.append((kwargs, kwargs["path_or_fileobj"].read_text()))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    assert job.main() == returncode

    assert creates == [{"repo_id": "me/results", "repo_type": "dataset", "private": True, "exist_ok": True}]
    upload, contents = uploads[0]
    assert upload["path_in_repo"] == "logs/local.log" and upload["repo_id"] == "me/results"
    assert contents == "auth: <redacted> <redacted> <redacted>"
    assert Path(captured["args"].spawn_log).read_text() == contents
    output = capsys.readouterr().out
    assert all(secret not in output for secret in secrets)
    assert ("Server log" in output) == bool(returncode)


def test_startup_failure_without_log_returns_failure_without_hub_upload(monkeypatch):
    monkeypatch.setenv("S2S_PUSH_TO_HUB", "me/results")
    captured = capture_launch(monkeypatch, returncode=2)

    assert job.main() == 2
    assert not Path(captured["args"].spawn_log).exists()
    assert not Path(captured["args"].out).exists()
