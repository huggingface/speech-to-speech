"""Command line for the Big Bench Audio vibe check.

python -m speech_to_speech.evals.big_bench_audio run --out main.json
python -m speech_to_speech.evals.big_bench_audio run --out pr.json --compare main.json
python -m speech_to_speech.evals.big_bench_audio compare main.json pr.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from speech_to_speech.evals.big_bench_audio.dataset import (
    DEFAULT_SUBSET,
    available_subsets,
    build_subset,
    describe,
    load_subset,
)
from speech_to_speech.evals.big_bench_audio.hub import load_report, push_report
from speech_to_speech.evals.big_bench_audio.judge import JudgeConfig, judge_results
from speech_to_speech.evals.big_bench_audio.report import (
    ItemResult,
    build_report,
    compare_reports,
    render_comparison,
    render_report,
)
from speech_to_speech.evals.big_bench_audio.runner import (
    DEFAULT_INSTRUCTIONS,
    RunnerConfig,
    open_session,
    run_subset,
)

logger = logging.getLogger("big_bench_audio")

DEFAULT_URL = "ws://127.0.0.1:8765/v1/realtime"


def redact_server_args(args: Sequence[str]) -> list[str]:
    """Keep reproducible settings without persisting credentials."""
    result = []
    redact_next = False
    for arg in args:
        key = arg.partition("=")[0].lower().replace("-", "_")
        sensitive = any(word in key for word in ("api_key", "token", "secret", "password"))
        if redact_next:
            result.append("<redacted>")
            redact_next = False
        elif arg.startswith("--") and sensitive:
            result.append(arg.partition("=")[0] + "=<redacted>" if "=" in arg else arg)
            redact_next = "=" not in arg
        else:
            result.append(arg)
    return result


def runtime_metadata() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for name in ("speech-to-speech", "torch", "transformers", "nano-parakeet", "faster-qwen3-tts"):
        with suppress(PackageNotFoundError):
            packages[name] = version(name)
    return {
        "source_revision": os.environ.get("S2S_SOURCE_REVISION"),
        "job_id": os.environ.get("JOB_ID"),
        "python": platform.python_version(),
        "packages": packages,
    }


def _split_server_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split ``... -- --stt_backend whisper`` into eval args and server args."""
    if "--" not in argv:
        return argv, []
    index = argv.index("--")
    return argv[:index], argv[index + 1 :]


def _url_from_server_args(server_args: Sequence[str]) -> str:
    """Build the realtime URL implied by ``--host`` / ``--port`` in *server_args*."""
    host, port = "127.0.0.1", "8765"
    for index, arg in enumerate(server_args):
        value = None
        if arg.startswith("--host="):
            value = ("host", arg.partition("=")[2])
        elif arg.startswith("--port="):
            value = ("port", arg.partition("=")[2])
        elif arg in ("--host", "--port") and index + 1 < len(server_args):
            value = (arg.lstrip("-"), server_args[index + 1])
        if value is None:
            continue
        if value[0] == "host":
            host = "127.0.0.1" if value[1] in ("0.0.0.0", "") else value[1]
        else:
            port = value[1]
    return f"ws://{host}:{port}/v1/realtime"


async def _wait_until_ready(config: RunnerConfig, process: Optional[subprocess.Popen], timeout_s: float) -> None:
    """Poll the realtime endpoint until it accepts a session."""
    deadline = time.monotonic() + timeout_s
    last_error: Optional[str] = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode} before becoming ready")
        try:
            async with open_session(config):
                return
        except Exception as exc:  # noqa: BLE001 — the server is still warming up
            last_error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(2.0)
    raise TimeoutError(f"server not ready after {timeout_s:.0f}s (last error: {last_error})")


def _spawn_server(server_args: list[str], log_path: Optional[str]) -> tuple[subprocess.Popen, Optional[Any]]:
    """Start ``speech-to-speech serve`` with *server_args*, returning it and its log handle."""
    command = [sys.executable, "-m", "speech_to_speech.cli", "serve", *server_args]
    print(f"spawning: {' '.join(redact_server_args(command))}", flush=True)
    log_file = open(log_path, "w") if log_path else None
    process = subprocess.Popen(
        command,
        stdout=log_file if log_file else subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    if log_path:
        print(f"server log: {log_path}", flush=True)
    return process, log_file


def _stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=30)
    if process.poll() is None:
        process.kill()
        process.wait()


@asynccontextmanager
async def _server(args: argparse.Namespace, config: RunnerConfig, server_args: list[str]) -> Any:
    """Optionally spawn ``speech-to-speech serve``, and wait for the endpoint either way."""
    process: Optional[subprocess.Popen] = None
    log_file: Optional[Any] = None
    if args.spawn:
        process, log_file = _spawn_server(server_args, args.spawn_log)
    try:
        print(f"waiting for {config.url} ...", flush=True)
        await _wait_until_ready(config, process, args.spawn_timeout if args.spawn else config.connect_timeout_s)
        print("server ready", flush=True)
        yield
    finally:
        if process is not None:
            _stop_server(process)
        if log_file is not None:
            log_file.close()


def _print_progress(index: int, total: int, result: ItemResult) -> None:
    if result.error:
        mark, detail = "!", result.error
    elif result.final_correct is True:
        mark, detail = "+", f"{result.final_extracted} == {result.official_answer}"
    elif result.final_correct is False:
        mark, detail = "-", f"{result.final_extracted} != {result.official_answer}"
    else:
        mark, detail = "?", "unparsed"
    timing = f"ttfb {result.ttfb_s:.1f}s" if result.ttfb_s is not None else "ttfb n/a"
    print(f"[{index:>3}/{total}] {mark} {result.category:<17} id={result.id:<4} {timing:<11} {detail}", flush=True)


def _runner_config(args: argparse.Namespace, url: str) -> RunnerConfig:
    instructions = DEFAULT_INSTRUCTIONS
    if args.instructions_file:
        instructions = Path(args.instructions_file).read_text().strip()
    elif args.instructions:
        instructions = args.instructions
    return RunnerConfig(
        url=url,
        api_key=args.api_key or os.environ.get("S2S_API_KEY") or os.environ.get("HF_TOKEN"),
        instructions=instructions,
        voice=args.voice,
        speed=args.speed,
        trailing_silence_ms=args.trailing_silence_ms,
        silence_duration_ms=args.silence_ms,
        response_timeout_s=args.response_timeout,
        retries=args.retries,
    )


async def _run(args: argparse.Namespace, server_args: list[str]) -> int:
    subset = load_subset(args.subset).head(args.limit)
    stats = describe(subset)
    print(f"subset {subset.name}: {stats.total} items -- {stats.by_category}", flush=True)
    if args.dry_run:
        for category, answers in sorted(stats.by_answer.items()):
            print(f"  {category:<18} {dict(sorted(answers.items()))}")
        print("  ids:", [item.id for item in subset.items])
        return 0

    url = args.url or (_url_from_server_args(server_args) if args.spawn else DEFAULT_URL)
    config = _runner_config(args, url)

    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    async with _server(args, config, server_args):
        results = await run_subset(subset, config, progress=_print_progress)
    duration_s = time.monotonic() - started

    judge_meta: Optional[dict[str, Any]] = None
    if args.judge_model:
        judge_config = JudgeConfig(
            model=args.judge_model,
            base_url=args.judge_base_url,
            api_key=args.judge_api_key or os.environ.get("OPENAI_API_KEY"),
        )
        print(f"judging {len(results)} replies with {judge_config.model} ...", flush=True)
        await judge_results(results, judge_config)
        judge_meta = judge_config.to_dict()

    report = build_report(
        results=results,
        subset=subset,
        label=args.label or f"{subset.name}@{started_at:%Y-%m-%dT%H:%M}",
        target={
            "url": url,
            "server_args": redact_server_args(server_args),
            "speed": args.speed,
            "instructions": config.instructions,
            "silence_ms": config.silence_duration_ms,
            "trailing_silence_ms": config.trailing_silence_ms,
            **runtime_metadata(),
        },
        judge=judge_meta,
        started_at=started_at,
        duration_s=duration_s,
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {out_path}", flush=True)
    if args.push_to_hub:
        url = push_report(report, args.push_to_hub, token=os.environ.get("HF_TOKEN"))
        print(f"pushed {url}", flush=True)
    print()
    print(render_report(report))

    if args.compare:
        baseline = load_report(args.compare, token=os.environ.get("HF_TOKEN"))
        comparison = compare_reports(baseline, report)
        print()
        print(render_comparison(comparison))
        if args.fail_on_regression:
            overall = next((row for row in comparison.rows if row.scope == "OVERALL"), None)
            if overall is not None and overall.significant and (overall.delta or 0) < 0:
                return 1
    return 1 if any(item.error for item in results) else 0


def _compare(args: argparse.Namespace) -> int:
    token = os.environ.get("HF_TOKEN")
    baseline = load_report(args.baseline, token=token)
    candidate = load_report(args.candidate, token=token)
    comparison = compare_reports(baseline, candidate)
    print(render_comparison(comparison))
    if args.fail_on_regression:
        overall = next((row for row in comparison.rows if row.scope == "OVERALL"), None)
        if overall is not None and overall.significant and (overall.delta or 0) < 0:
            return 1
    return 0


def _show(args: argparse.Namespace) -> int:
    print(render_report(load_report(args.report, token=os.environ.get("HF_TOKEN"))))
    return 0


def _build_subset(args: argparse.Namespace) -> int:
    subset = build_subset(args.size, name=args.name, description=args.description, seed=args.seed)
    Path(args.out).write_text(json.dumps(subset.to_dict(), indent=2) + "\n")
    stats = describe(subset)
    print(f"wrote {args.out}: {stats.total} items -- {stats.by_category}")
    return 0


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--url", default=None, help=f"Realtime endpoint (default {DEFAULT_URL}).")
    parser.add_argument("--api-key", default=None, help="Bearer token; falls back to S2S_API_KEY or HF_TOKEN.")
    parser.add_argument(
        "--subset",
        default=DEFAULT_SUBSET,
        help=f"Bundled subset name or manifest path. Bundled: {', '.join(available_subsets()) or 'none'}.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N items (stays balanced).")
    parser.add_argument("--label", default=None, help="Name for this run in the report and comparisons.")
    parser.add_argument("--out", default=None, help="Write the JSON report here.")
    parser.add_argument(
        "--push-to-hub",
        default=None,
        help="Dataset repo to upload the report to, e.g. my-org/s2s-vibe-checks.",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="Baseline report to diff against: a local path, or '<repo_id>:<path-in-repo>'.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit 1 when overall accuracy drops beyond sampling noise.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the subset composition and exit.")

    parser.add_argument("--instructions", default=None, help="Override the eval system prompt.")
    parser.add_argument("--instructions-file", default=None, help="Read the eval system prompt from a file.")
    parser.add_argument("--voice", default=None, help="TTS voice to request.")
    parser.add_argument("--speed", type=float, default=1.0, help="Audio send pacing, 1.0 = real time.")
    parser.add_argument("--silence-ms", type=int, default=900, help="server_vad silence_duration_ms for the run.")
    parser.add_argument("--trailing-silence-ms", type=int, default=2000, help="Silence appended after each question.")
    parser.add_argument("--response-timeout", type=float, default=180.0, help="Per-item wait for a finished response.")
    parser.add_argument("--retries", type=int, default=1, help="Retries per item on connection or timeout errors.")

    parser.add_argument("--judge-model", default=None, help="Grade replies with this OpenAI-compatible model.")
    parser.add_argument("--judge-base-url", default=None, help="Base URL for the judge model.")
    parser.add_argument("--judge-api-key", default=None, help="API key for the judge; falls back to OPENAI_API_KEY.")

    parser.add_argument(
        "--spawn",
        action="store_true",
        help="Start `speech-to-speech serve` for the run; arguments after `--` go to it.",
    )
    parser.add_argument("--spawn-timeout", type=float, default=900.0, help="Wait for the spawned server to load.")
    parser.add_argument("--spawn-log", default=None, help="Write the spawned server's output here.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m speech_to_speech.evals.big_bench_audio",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the vibe check against a Realtime server.")
    _add_run_arguments(run_parser)

    compare_parser = subparsers.add_parser("compare", help="Diff two reports.")
    compare_parser.add_argument("baseline", help="Local path, or '<repo_id>:<path-in-repo>'.")
    compare_parser.add_argument("candidate", help="Local path, or '<repo_id>:<path-in-repo>'.")
    compare_parser.add_argument("--fail-on-regression", action="store_true")

    show_parser = subparsers.add_parser("show", help="Re-render a saved report.")
    show_parser.add_argument("report", help="Local path, or '<repo_id>:<path-in-repo>'.")

    subset_parser = subparsers.add_parser("build-subset", help="Draw a new balanced subset manifest.")
    subset_parser.add_argument("--size", type=int, default=40)
    subset_parser.add_argument("--name", default=DEFAULT_SUBSET)
    subset_parser.add_argument("--description", default="")
    subset_parser.add_argument("--seed", type=int, default=0)
    subset_parser.add_argument("--out", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    eval_args, server_args = _split_server_args(list(sys.argv[1:] if argv is None else argv))
    parser = build_parser()
    # `run` is the default command, so plain flags work: `... --limit 8 --out pr.json`.
    if not eval_args or (eval_args[0].startswith("-") and eval_args[0] not in ("-h", "--help")):
        eval_args = ["run", *eval_args]
    args = parser.parse_args(eval_args)

    if args.command == "compare":
        return _compare(args)
    if args.command == "show":
        return _show(args)
    if args.command == "build-subset":
        return _build_subset(args)
    return asyncio.run(_run(args, server_args))


if __name__ == "__main__":
    raise SystemExit(main())
