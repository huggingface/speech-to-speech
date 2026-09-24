import json
from contextlib import asynccontextmanager

import httpx
import pytest

from speech_to_speech.evals.big_bench_audio import __main__ as cli
from speech_to_speech.evals.big_bench_audio import runner
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


@pytest.mark.parametrize("stage", ["download", "decode", "engine"])
async def test_item_failure_preserves_and_uploads_other_results(monkeypatch, tmp_path, stage):
    destination = tmp_path / "reports" / "run.json"
    args = cli.build_parser().parse_args(
        ["run", "--limit", "4", "--retries", "1", "--out", str(destination), "--push-to-hub", "me/results"]
    )
    items = load_subset().head(4).items
    failed_id = items[1].id
    attempts = []
    completed = []
    uploads = []

    @asynccontextmanager
    async def ready_server(*args):
        yield

    def resolve(item, subset):
        attempts.append(item.id)
        if stage == "download" and item.id == failed_id:
            raise httpx.ReadTimeout("Audio download timed out")
        return item.id

    def decode(item_id):
        if stage == "decode" and item_id == failed_id:
            raise ValueError("Invalid audio")
        return b"\x00\x00"

    async def run_item(config, item, pcm):
        error = "provider unavailable" if stage == "engine" and item.id == failed_id else None
        if error is None:
            completed.append(item.id)
        return ItemResult(
            id=item.id,
            category=item.category,
            official_answer=item.official_answer,
            extracted=item.official_answer,
            correct=True,
            audio_out_bytes=100,
            error=error,
        )

    async def run_subset(subset, config, progress):
        config.settle_s = 0
        return await runner.run_subset(subset, config, progress=progress)

    def push_report(report, repo_id, *, token):
        assert repo_id == "me/results"
        assert json.loads(destination.read_text()) == report
        uploads.append(report)
        return "https://example.test/report.json"

    monkeypatch.setattr(cli, "_server", ready_server)
    monkeypatch.setattr(cli, "run_subset", run_subset)
    monkeypatch.setattr(cli, "push_report", push_report)
    monkeypatch.setattr(runner, "resolve_audio", resolve)
    monkeypatch.setattr(runner, "load_pcm16_mono", decode)
    monkeypatch.setattr(runner, "run_item", run_item)

    assert await cli._run(args, []) == 1
    assert completed == [item.id for item in items if item.id != failed_id]
    expected_attempts = [items[0].id, failed_id]
    if stage != "engine":  # Engine retries reuse the decoded audio.
        expected_attempts.append(failed_id)
    assert attempts == expected_attempts + [items[2].id, items[3].id]
    (report,) = uploads
    assert report["totals"]["n"] == 4
    assert report["totals"]["errors"] == 1
    assert report["totals"]["correct"] == 3
    error = report["items"][1]["error"]
    assert error.startswith("provider unavailable" if stage == "engine" else "audio_load_failed:")
