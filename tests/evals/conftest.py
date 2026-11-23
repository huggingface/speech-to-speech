"""Offline metadata for benchmark selection tests."""

import json

import huggingface_hub
import pytest

from speech_to_speech.evals.big_bench_audio.dataset import DATASET_REPO_ID, DATASET_REVISION, EvalItem


@pytest.fixture
def benchmark_metadata(monkeypatch, tmp_path):
    categories = ("formal_fallacies", "navigate", "object_counting", "web_of_lies")
    answers = (("invalid", "valid"), ("No", "Yes"), tuple(str(n) for n in range(2, 12)), ("No", "Yes"))
    records = []
    for category, choices in zip(categories, answers):
        for index in range(250):
            item_id = len(records)
            records.append(
                {
                    "id": item_id,
                    "category": category,
                    "official_answer": choices[index % len(choices)],
                    "file_name": f"data/question_{item_id}.mp3",
                }
            )
    path = tmp_path / "metadata.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records))

    def download(repo_id, filename, *, repo_type, revision):
        assert (repo_id, filename, repo_type, revision) == (
            DATASET_REPO_ID,
            "metadata.jsonl",
            "dataset",
            DATASET_REVISION,
        )
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return tuple(EvalItem.from_dict(record) for record in records)
