"""Offline metadata for full-benchmark selection tests."""

import json

import huggingface_hub
import pytest

from speech_to_speech.evals.big_bench_audio.dataset import DATASET_REPO_ID, DATASET_REVISION, EvalItem


@pytest.fixture
def benchmark_metadata(monkeypatch, tmp_path):
    categories = ("formal_fallacies", "navigate", "object_counting", "web_of_lies")
    records = [
        {
            "id": item_id,
            "category": categories[item_id // 250],
            "official_answer": str(item_id % 10),
            "file_name": f"data/question_{item_id}.mp3",
        }
        for item_id in range(1000)
    ]
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
