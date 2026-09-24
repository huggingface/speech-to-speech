import json
from contextlib import asynccontextmanager

import pytest

from speech_to_speech.evals.big_bench_audio import __main__ as cli
from speech_to_speech.evals.big_bench_audio.__main__ import redact_server_args
from speech_to_speech.evals.big_bench_audio.dataset import load_subset
from speech_to_speech.evals.big_bench_audio.report import ItemResult
from speech_to_speech.evals.big_bench_audio.runner import RunnerConfig


def test_credentials_are_removed_from_report_arguments():
    args = ["--responses_api_api_key", "secret1", "--api-key=secret2", "--model_name", "Qwen/test"]
    assert redact_server_args(args) == [
        "--responses_api_api_key",
        "<redacted>",
        "--api-key=<redacted>",
        "--model_name",
        "Qwen/test",
    ]


@pytest.mark.parametrize("limit", [0, -1])
def test_nonpositive_limits_are_rejected(limit):
    with pytest.raises(ValueError, match="limit"):
        load_subset().head(limit)


@pytest.mark.parametrize("speed", [0, -1, float("inf"), float("nan")])
def test_invalid_speed_is_rejected(speed):
    with pytest.raises(ValueError, match="speed"):
        RunnerConfig(speed=speed)


async def test_failed_item_report_is_saved_and_uploaded_before_returning_failure(monkeypatch, tmp_path):
    destination = tmp_path / "reports" / "failed.json"
    args = cli.build_parser().parse_args(
        ["run", "--limit", "4", "--out", str(destination), "--push-to-hub", "me/results"]
    )
    uploads = []

    @asynccontextmanager
    async def ready_server(*args):
        yield

    async def run_subset(subset, config, progress):
        results = [
            ItemResult(
                id=item.id,
                category=item.category,
                official_answer=item.official_answer,
                extracted=item.official_answer,
                correct=True,
                audio_out_bytes=1600,
            )
            for item in subset.items
        ]
        results[0].error = "provider unavailable"
        return results

    def push_report(report, repo_id, *, token):
        assert json.loads(destination.read_text()) == report
        uploads.append((repo_id, report))
        return "https://example.test/report.json"

    monkeypatch.setattr(cli, "_server", ready_server)
    monkeypatch.setattr(cli, "run_subset", run_subset)
    monkeypatch.setattr(cli, "push_report", push_report)

    assert await cli._run(args, []) == 1

    repo_id, report = uploads[0]
    assert repo_id == "me/results"
    assert report["totals"]["n"] == 4 and report["totals"]["errors"] == 1
    assert report["totals"]["correct"] == 3
    assert report["items"][0]["error"] == "provider unavailable"
