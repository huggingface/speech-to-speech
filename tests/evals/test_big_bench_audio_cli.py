import pytest

from speech_to_speech.evals.big_bench_audio.__main__ import redact_server_args
from speech_to_speech.evals.big_bench_audio.dataset import load_subset
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
