"""Pinned Big Bench Audio subsets and the audio they point at.

A vibe check is only useful if two runs saw the same questions, so the subset is
a committed JSON manifest -- dataset id, dataset revision, and the exact item ids
with their official answers -- rather than a sample drawn at run time. Audio is
fetched from the Hub at the pinned revision and cached by ``huggingface_hub``.
"""

from __future__ import annotations

import json
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

DATASET_REPO_ID = "ArtificialAnalysis/big_bench_audio"
# Pinned so a manifest keeps pointing at the same 1000 recordings.
DATASET_REVISION = "af7bb9c25b015792583ca4da3ee27ec62cb79fe6"
DATASET_METADATA_FILE = "metadata.jsonl"

SUBSET_SCHEMA = "big-bench-audio-subset/1"
SUBSETS_DIR = Path(__file__).parent / "subsets"
DEFAULT_SUBSET = "vibe"

SAMPLE_RATE_HZ = 16000


@dataclass(frozen=True)
class EvalItem:
    """One spoken question and its official answer."""

    id: int
    category: str
    official_answer: str
    file_name: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EvalItem:
        return cls(
            id=int(raw["id"]),
            category=str(raw["category"]),
            official_answer=str(raw["official_answer"]),
            file_name=str(raw["file_name"]).replace("\\/", "/"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "official_answer": self.official_answer,
            "file_name": self.file_name,
        }


@dataclass(frozen=True)
class Subset:
    """A named, revision-pinned selection of dataset items."""

    name: str
    items: tuple[EvalItem, ...]
    dataset: str = DATASET_REPO_ID
    revision: str = DATASET_REVISION
    description: str = ""
    seed: Optional[int] = None

    def __len__(self) -> int:
        return len(self.items)

    def head(self, limit: Optional[int]) -> Subset:
        """First *limit* items. Manifest order round-robins categories, so a
        prefix stays balanced across them."""
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        if limit is None or limit >= len(self.items):
            return self
        return Subset(
            name=self.name,
            items=self.items[:limit],
            dataset=self.dataset,
            revision=self.revision,
            description=self.description,
            seed=self.seed,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SUBSET_SCHEMA,
            "name": self.name,
            "description": self.description,
            "dataset": self.dataset,
            "revision": self.revision,
            "seed": self.seed,
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Subset:
        schema = raw.get("schema")
        if schema != SUBSET_SCHEMA:
            raise ValueError(f"Unsupported subset schema {schema!r}; expected {SUBSET_SCHEMA!r}.")
        return cls(
            name=str(raw["name"]),
            items=tuple(EvalItem.from_dict(item) for item in raw["items"]),
            dataset=str(raw.get("dataset", DATASET_REPO_ID)),
            revision=str(raw.get("revision", DATASET_REVISION)),
            description=str(raw.get("description", "")),
            seed=raw.get("seed"),
        )


def available_subsets() -> list[str]:
    """Names of the subsets bundled with the package."""
    return sorted(path.stem for path in SUBSETS_DIR.glob("*.json"))


def load_subset(name_or_path: str = DEFAULT_SUBSET) -> Subset:
    """Load a bundled subset by name, or any manifest by path."""
    bundled = SUBSETS_DIR / f"{name_or_path}.json"
    path = bundled if bundled.is_file() else Path(name_or_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No subset {name_or_path!r}. Bundled subsets: {', '.join(available_subsets()) or 'none'}."
        )
    return Subset.from_dict(json.loads(path.read_text()))


def _answer_sort_key(category: str, answer: str) -> tuple[int, Any]:
    """Sort integer answers numerically and everything else alphabetically."""
    try:
        return (0, int(answer))
    except ValueError:
        return (1, answer)


def load_dataset_metadata(revision: str = DATASET_REVISION) -> list[EvalItem]:
    """Download and parse the dataset's ``metadata.jsonl`` (one small file, no audio)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        DATASET_REPO_ID,
        DATASET_METADATA_FILE,
        repo_type="dataset",
        revision=revision,
    )
    with open(path) as handle:
        return [EvalItem.from_dict(json.loads(line)) for line in handle if line.strip()]


def build_subset(
    size: int,
    *,
    name: str = DEFAULT_SUBSET,
    description: str = "",
    seed: int = 0,
    revision: str = DATASET_REVISION,
    items: Optional[list[EvalItem]] = None,
) -> Subset:
    """Draw a balanced subset of *size* items from the dataset.

    Balanced twice over: equally across the four categories, and within each
    category equally across its official answers -- so a model that always says
    "No" cannot clear 50% by luck. Items are emitted round-robin by category, so
    truncating the manifest keeps that balance.
    """
    pool = items if items is not None else load_dataset_metadata(revision)
    if not pool:
        raise ValueError("Dataset metadata is empty.")

    categories = sorted({item.category for item in pool})
    if size < len(categories):
        raise ValueError(f"size must be at least {len(categories)} to cover every category.")

    rng = random.Random(seed)
    buckets: dict[str, dict[str, list[EvalItem]]] = defaultdict(lambda: defaultdict(list))
    for item in pool:
        buckets[item.category][item.official_answer].append(item)

    per_category: dict[str, list[EvalItem]] = {}
    for index, category in enumerate(categories):
        # Spread the remainder over the first categories so the total lands on `size`.
        quota = size // len(categories) + (1 if index < size % len(categories) else 0)
        answers = sorted(buckets[category], key=lambda answer: _answer_sort_key(category, answer))
        queues = {answer: sorted(buckets[category][answer], key=lambda item: item.id) for answer in answers}
        for queue in queues.values():
            rng.shuffle(queue)

        picked: list[EvalItem] = []
        while len(picked) < quota:
            drained = True
            for answer in answers:
                if len(picked) == quota:
                    break
                if queues[answer]:
                    picked.append(queues[answer].pop())
                    drained = False
            if drained:
                raise ValueError(f"Category {category!r} has fewer than {quota} items.")
        per_category[category] = picked

    ordered: list[EvalItem] = []
    remaining = OrderedDict((category, list(per_category[category])) for category in categories)
    while remaining:
        for category in list(remaining):
            queue = remaining[category]
            ordered.append(queue.pop(0))
            if not queue:
                del remaining[category]

    return Subset(
        name=name,
        items=tuple(ordered),
        revision=revision,
        description=description,
        seed=seed,
    )


def resolve_audio(item: EvalItem, subset: Subset) -> Path:
    """Local path to *item*'s recording, downloading it into the Hub cache if needed."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            subset.dataset,
            item.file_name,
            repo_type="dataset",
            revision=subset.revision,
        )
    )


def load_pcm16_mono(path: Path, rate: int = SAMPLE_RATE_HZ) -> bytes:
    """Decode any dataset recording to mono little-endian PCM16 at *rate*."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    data, source_rate = sf.read(str(path), always_2d=True, dtype="float32")
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if source_rate != rate:
        mono = resample_poly(mono, rate, source_rate)
    return (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


@dataclass
class SubsetStats:
    """Per-category counts, used by ``--dry-run`` and the report header."""

    total: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    by_answer: dict[str, dict[str, int]] = field(default_factory=dict)


def describe(subset: Subset) -> SubsetStats:
    stats = SubsetStats(total=len(subset))
    for item in subset.items:
        stats.by_category[item.category] = stats.by_category.get(item.category, 0) + 1
        answers = stats.by_answer.setdefault(item.category, {})
        answers[item.official_answer] = answers.get(item.official_answer, 0) + 1
    return stats
