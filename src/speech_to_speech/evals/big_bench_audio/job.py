"""HF Jobs entrypoint: configure the engine and retain its report and diagnostics."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from speech_to_speech.evals.big_bench_audio.hub import slugify


def main() -> int:
    output = Path(os.environ.get("S2S_REPORT_DIR", "/output"))
    output.mkdir(parents=True, exist_ok=True)
    label = os.environ.get("S2S_LABEL", os.environ.get("JOB_ID", "local"))
    name = slugify(label)
    server_log = output / f"{name}-server.log"
    # OpenAI-compatible clients read this environment variable; never put keys
    # into command-line arguments, printed commands, or report metadata.
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("HF_TOKEN", ""))
    default_server = [
        "--stt",
        "parakeet-tdt",
        "--tts",
        "qwen3",
        "--qwen3_tts_backend",
        "torch",
        "--llm_backend",
        "chat-completions",
        "--model_name",
        os.environ.get("S2S_LLM_MODEL", "Qwen/Qwen3.5-9B:together"),
        "--responses_api_base_url",
        os.environ.get("S2S_LLM_BASE_URL", "https://router.huggingface.co/v1"),
    ]
    args = [
        "run",
        "--spawn",
        "--spawn-log",
        str(server_log),
        "--subset",
        os.environ.get("S2S_SUBSET", "vibe"),
        "--label",
        label,
        "--out",
        str(output / f"{name}.json"),
    ]
    for env, flag in (("S2S_LIMIT", "--limit"), ("S2S_COMPARE", "--compare"), ("S2S_PUSH_TO_HUB", "--push-to-hub")):
        if os.environ.get(env):
            args.extend([flag, os.environ[env]])
    args.extend(shlex.split(os.environ.get("S2S_EVAL_ARGS", "")))
    server_args = shlex.split(os.environ["S2S_SERVE_ARGS"]) if os.environ.get("S2S_SERVE_ARGS") else default_server
    completed = subprocess.run(
        [sys.executable, "-m", "speech_to_speech.evals.big_bench_audio", *args, "--", *server_args]
    )
    # Startup failures have no report; surface the log in Job logs and preserve
    # it in the private result repo as well as the report when configured.
    if server_log.exists():
        log = server_log.read_text(errors="replace")
        for key in ("HF_TOKEN", "OPENAI_API_KEY", "S2S_API_KEY"):
            if os.environ.get(key):
                log = log.replace(os.environ[key], "<redacted>")
        server_log.write_text(log)
        if completed.returncode:
            print("\nServer log (last 100 lines):\n" + "\n".join(log.splitlines()[-100:]), flush=True)
        repo = os.environ.get("S2S_PUSH_TO_HUB")
        if repo:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
            api.upload_file(
                path_or_fileobj=server_log, path_in_repo=f"logs/{name}.log", repo_id=repo, repo_type="dataset"
            )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
