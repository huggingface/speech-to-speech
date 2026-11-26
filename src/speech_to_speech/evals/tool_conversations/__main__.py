"""Tool-use conversations through a running Realtime server.

Start a server whose LLM you want to test (``server-config`` writes a text-only config whose
speech services point at ``fake-speech``), then run the scripted conversations against it:

    python -m speech_to_speech.evals.tool_conversations fake-speech --port 8090 &
    python -m speech_to_speech.evals.tool_conversations server-config --out cfg.json \\
        --backend chat-completions --model google/gemma-4-31B-it --base-url https://router.huggingface.co/v1
    speech-to-speech serve cfg.json &
    python -m speech_to_speech.evals.tool_conversations run --out main.json --label main --reps 20
    python -m speech_to_speech.evals.tool_conversations run --out first.json --set first-turns --reps 10
    python -m speech_to_speech.evals.tool_conversations compare main.json pr.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from speech_to_speech.evals.tool_conversations.report import build_report, render_comparison, render_report
from speech_to_speech.evals.tool_conversations.runner import RunnerConfig, run_set
from speech_to_speech.evals.tool_conversations.scenarios import SCENARIO_SETS

DEFAULT_URL = "ws://127.0.0.1:8765/v1/realtime"


def _run(args: argparse.Namespace) -> int:
    headers = {}
    if args.token_env:
        token = os.environ.get(args.token_env, "").strip()
        if not token:
            print(f"{args.token_env} is not set", file=sys.stderr)
            return 2
        headers["Authorization"] = f"Bearer {token}"
    config = RunnerConfig(url=args.url, headers=headers, concurrency=args.concurrency)
    started = datetime.now(timezone.utc).isoformat()
    results = asyncio.run(run_set(config, SCENARIO_SETS[args.set], args.reps, seed=args.seed))
    meta = {
        "label": args.label,
        "url": args.url,
        "scenario_set": args.set,
        "reps": args.reps,
        "started_at": started,
        "notes": args.notes,
    }
    report = build_report(meta, results)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(render_report(report))
    return 0


def _compare(args: argparse.Namespace) -> int:
    reports = [json.loads(Path(path).read_text()) for path in args.reports]
    print(render_comparison(reports))
    return 0


def _fake_speech(args: argparse.Namespace) -> int:
    from speech_to_speech.evals.tool_conversations.fake_speech import serve

    serve(args.host, args.port)
    return 0


def _server_config(args: argparse.Namespace) -> int:
    """A text-only server config: real LLM, speech services pointed at fake-speech."""
    speech = f"http://127.0.0.1:{args.speech_port}/v1"
    config = {
        "host": "127.0.0.1",
        "port": args.port,
        "device": "cpu",
        "num_pipelines": args.pipelines,
        "detect_llm_output_language": True,
        "stt": "openai",
        "openai_stt_base_url": speech,
        "openai_stt_api_key": "unused",
        "openai_stt_model": "fake",
        "openai_stt_response_format": "json",
        "llm_backend": args.backend,
        "model_name": args.model,
        "responses_api_base_url": args.base_url,
        "responses_api_stream": True,
        "stream_batch_sentences": args.stream_batch_sentences,
        "tts": "openai",
        "openai_tts_base_url": speech,
        "openai_tts_api_key": "unused",
        "openai_tts_model": "fake",
        "openai_tts_voice": "aiden",
        "openai_tts_response_format": "pcm",
        "openai_tts_sample_rate": 24000,
        "openai_tts_stream": True,
    }
    if args.reasoning_effort:
        config["responses_api_reasoning_effort"] = args.reasoning_effort
    if args.api_key_env:
        key = os.environ.get(args.api_key_env, "").strip()
        if not key:
            print(f"{args.api_key_env} is not set", file=sys.stderr)
            return 2
        config["responses_api_api_key"] = key
    path = Path(args.out)
    path.write_text(json.dumps(config, indent=1))
    path.chmod(0o600)
    print(f"wrote {path} (contains the API key; keep it private)" if args.api_key_env else f"wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m speech_to_speech.evals.tool_conversations",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run scripted conversations against a Realtime server")
    run.add_argument("--url", default=DEFAULT_URL)
    run.add_argument("--out", required=True, help="report JSON path")
    run.add_argument("--label", default="", help="name shown in tables, e.g. 'main + Google template'")
    run.add_argument("--set", choices=sorted(SCENARIO_SETS), default="conversations")
    run.add_argument("--reps", type=int, default=8, help="repetitions of each conversation")
    run.add_argument("--concurrency", type=int, default=3, help="keep below the server's num_pipelines")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--token-env", help="env var holding a bearer token for the server, if it needs one")
    run.add_argument("--notes", default="", help="free text stored in the report (model, template, commit)")
    run.set_defaults(func=_run)

    compare = sub.add_parser("compare", help="compare reports; p-values are against the first one")
    compare.add_argument("reports", nargs="+")
    compare.set_defaults(func=_compare)

    fake = sub.add_parser("fake-speech", help="serve silent TTS and empty STT for text-only runs")
    fake.add_argument("--host", default="127.0.0.1")
    fake.add_argument("--port", type=int, default=8090)
    fake.set_defaults(func=_fake_speech)

    cfg = sub.add_parser("server-config", help="write a text-only server config for an LLM")
    cfg.add_argument("--out", required=True)
    cfg.add_argument("--backend", choices=("chat-completions", "responses-api"), default="chat-completions")
    cfg.add_argument("--model", required=True)
    cfg.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, ending in /v1")
    cfg.add_argument("--api-key-env", help="env var holding the LLM API key, written into the config")
    cfg.add_argument("--reasoning-effort", help="e.g. 'none' for router models that reason by default")
    cfg.add_argument("--port", type=int, default=8765)
    cfg.add_argument("--speech-port", type=int, default=8090)
    cfg.add_argument("--pipelines", type=int, default=4)
    cfg.add_argument("--stream-batch-sentences", type=int, default=2)
    cfg.set_defaults(func=_server_config)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
