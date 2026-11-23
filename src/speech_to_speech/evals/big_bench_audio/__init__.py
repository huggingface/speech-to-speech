"""Big Bench Audio vibe check: spoken reasoning questions through the full cascade.

The harness streams a pinned subset of `ArtificialAnalysis/big_bench_audio
<https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio>`_ into a
running Realtime server, reads the assistant's spoken answer back off the
transcript stream, and grades it against the dataset's official answer. Reports
are plain JSON so two runs -- ``main`` and a pull request that swaps a backend --
can be diffed with the ``compare`` command.
"""

from speech_to_speech.evals.big_bench_audio.dataset import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    EvalItem,
    Subset,
    build_subset,
    load_subset,
    resolve_audio,
)
from speech_to_speech.evals.big_bench_audio.report import (
    REPORT_SCHEMA,
    ItemResult,
    aggregate,
    build_report,
    compare_reports,
    render_comparison,
    render_report,
)
from speech_to_speech.evals.big_bench_audio.scoring import (
    CATEGORY_SPECS,
    Grade,
    extract_answer,
    grade,
)

__all__ = [
    "CATEGORY_SPECS",
    "DATASET_REPO_ID",
    "DATASET_REVISION",
    "REPORT_SCHEMA",
    "EvalItem",
    "Grade",
    "ItemResult",
    "Subset",
    "aggregate",
    "build_report",
    "build_subset",
    "compare_reports",
    "extract_answer",
    "grade",
    "load_subset",
    "render_comparison",
    "render_report",
    "resolve_audio",
]
