"""The bundled subset is balanced and pinned, and truncating it keeps it balanced."""

import json
from collections import Counter

import numpy as np
import pytest
import soundfile as sf

from speech_to_speech.evals.big_bench_audio.dataset import (
    DATASET_REPO_ID,
    SAMPLE_RATE_HZ,
    EvalItem,
    Subset,
    build_subset,
    describe,
    load_pcm16_mono,
    load_subset,
)

CATEGORIES = ("formal_fallacies", "navigate", "object_counting", "web_of_lies")


def fake_pool():
    """A stand-in for metadata.jsonl with the real category and answer shapes."""
    answers = {
        "formal_fallacies": ["valid", "invalid"],
        "navigate": ["Yes", "No"],
        "object_counting": [str(n) for n in range(2, 10)],
        "web_of_lies": ["Yes", "No"],
    }
    pool, item_id = [], 0
    for category in CATEGORIES:
        for _ in range(30):
            for answer in answers[category]:
                pool.append(
                    EvalItem(
                        id=item_id,
                        category=category,
                        official_answer=answer,
                        file_name=f"data/question_{item_id}.mp3",
                    )
                )
                item_id += 1
    return pool


def test_bundled_subset_is_pinned_to_one_dataset_revision():
    subset = load_subset("vibe")

    assert subset.dataset == DATASET_REPO_ID
    assert len(subset.revision) == 40
    assert len(subset) == 40


def test_bundled_subset_is_balanced_across_categories_and_answers():
    stats = describe(load_subset("vibe"))

    assert stats.by_category == dict.fromkeys(CATEGORIES, 10)
    assert stats.by_answer["formal_fallacies"] == {"invalid": 5, "valid": 5}
    assert stats.by_answer["navigate"] == {"No": 5, "Yes": 5}
    assert stats.by_answer["web_of_lies"] == {"No": 5, "Yes": 5}
    # Ten distinct counts rather than ten of the same number.
    assert len(stats.by_answer["object_counting"]) == 10


def test_truncating_the_subset_keeps_every_category_represented():
    head = load_subset("vibe").head(8)

    assert len(head) == 8
    assert Counter(item.category for item in head.items) == dict.fromkeys(CATEGORIES, 2)


def test_head_beyond_the_subset_is_the_whole_subset():
    subset = load_subset("vibe")

    assert subset.head(None) is subset
    assert len(subset.head(1000)) == 40


def test_build_subset_balances_categories_and_answers():
    subset = build_subset(40, items=fake_pool(), seed=7)
    stats = describe(subset)

    assert stats.by_category == dict.fromkeys(CATEGORIES, 10)
    assert stats.by_answer["navigate"] == {"No": 5, "Yes": 5}
    assert len({item.id for item in subset.items}) == 40


def test_build_subset_is_deterministic_for_a_seed():
    ids = [item.id for item in build_subset(20, items=fake_pool(), seed=3).items]

    assert ids == [item.id for item in build_subset(20, items=fake_pool(), seed=3).items]
    assert ids != [item.id for item in build_subset(20, items=fake_pool(), seed=4).items]


def test_build_subset_rejects_a_size_that_cannot_cover_every_category():
    with pytest.raises(ValueError, match="at least 4"):
        build_subset(3, items=fake_pool())


def test_subset_survives_a_manifest_round_trip(tmp_path):
    subset = load_subset("vibe")
    path = tmp_path / "subset.json"
    path.write_text(json.dumps(subset.to_dict()))

    assert load_subset(str(path)).items == subset.items


def test_a_manifest_from_an_unknown_schema_is_refused(tmp_path):
    path = tmp_path / "subset.json"
    path.write_text(json.dumps({"schema": "something-else", "name": "x", "items": []}))

    with pytest.raises(ValueError, match="Unsupported subset schema"):
        load_subset(str(path))


def test_missing_subset_names_the_bundled_ones():
    with pytest.raises(FileNotFoundError, match="vibe"):
        load_subset("not-a-subset")


def test_audio_is_decoded_to_mono_pcm16_at_the_pipeline_rate(tmp_path):
    path = tmp_path / "question.wav"
    sf.write(path, np.zeros(22050, dtype=np.float32), 22050)

    pcm = load_pcm16_mono(path)

    assert len(pcm) == SAMPLE_RATE_HZ * 2


def test_windows_style_file_names_are_normalized():
    item = EvalItem.from_dict(
        {"id": 3, "category": "navigate", "official_answer": "Yes", "file_name": "data\\/question_3.mp3"}
    )

    assert item.file_name == "data/question_3.mp3"


def test_subset_defaults_to_the_pinned_dataset():
    assert Subset(name="x", items=()).dataset == DATASET_REPO_ID
