"""Reports published to the Hub get sortable names and can be read back by reference."""

import json

import pytest

from speech_to_speech.evals.big_bench_audio.hub import (
    load_report,
    push_report,
    report_filename,
    slugify,
)


class FakeApi:
    """Stands in for HfApi, recording what would have been sent."""

    def __init__(self):
        self.created = []
        self.uploaded = []

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def upload_file(self, **kwargs):
        self.uploaded.append(kwargs)


def report(label="pr-swaps-stt", started_at="2026-09-15T14:02:33+00:00"):
    return {"label": label, "started_at": started_at, "totals": {"accuracy": 0.65}}


def test_filenames_sort_chronologically_and_name_the_run():
    assert report_filename(report()) == "reports/20260915T140233-pr-swaps-stt.json"


def test_filenames_survive_a_label_with_slashes_and_spaces():
    name = report_filename(report(label="feat/new stt backend"))

    assert name == "reports/20260915T140233-feat-new-stt-backend.json"
    assert name.count("/") == 1


def test_an_unparseable_timestamp_still_produces_a_name():
    assert report_filename(report(started_at="not-a-date")).endswith("-pr-swaps-stt.json")


@pytest.mark.parametrize(
    ("label", "expected"),
    [("main", "main"), ("", "run"), ("---", "run"), ("a" * 200, "a" * 80)],
)
def test_slugify_always_yields_a_usable_segment(label, expected):
    assert slugify(label) == expected


def test_pushing_creates_the_dataset_repo_then_uploads_the_report():
    api = FakeApi()

    url = push_report(report(), "me/s2s-vibe-checks", api=api)

    assert api.created[0]["repo_type"] == "dataset"
    assert api.created[0]["exist_ok"] is True
    upload = api.uploaded[0]
    assert upload["repo_id"] == "me/s2s-vibe-checks"
    assert json.loads(upload["path_or_fileobj"])["label"] == "pr-swaps-stt"
    assert url.endswith("reports/20260915T140233-pr-swaps-stt.json")


def test_an_explicit_path_overrides_the_generated_one():
    api = FakeApi()

    push_report(report(), "me/runs", path_in_repo="baselines/main.json", api=api)

    assert api.uploaded[0]["path_in_repo"] == "baselines/main.json"


def test_a_local_report_loads_from_disk(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(report()))

    assert load_report(str(path))["label"] == "pr-swaps-stt"


def test_a_reference_without_a_repo_path_is_refused():
    with pytest.raises(FileNotFoundError, match="repo_id"):
        load_report("no-such-file.json")
