"""Run reports: aggregation, rendering, and baseline-vs-candidate comparison.

Reports are plain JSON on purpose. The whole point of the harness is that a run
against ``main`` and a run against a pull request that swaps a backend produce
two files you can diff, so the schema is stable and every derived number is
recomputable from ``items``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from speech_to_speech.evals.big_bench_audio.dataset import Subset

REPORT_SCHEMA = "big-bench-audio-vibe/1"


@dataclass
class ItemResult:
    """One question's outcome, from audio in to graded answer out."""

    id: int
    category: str
    official_answer: str
    input_transcript: str = ""
    reply: str = ""
    extracted: Optional[str] = None
    correct: Optional[bool] = None
    method: Optional[str] = None
    judge_extracted: Optional[str] = None
    judge_correct: Optional[bool] = None
    ttfb_s: Optional[float] = None
    turn_s: Optional[float] = None
    audio_out_bytes: int = 0
    error: Optional[str] = None
    judge_completed: bool = False

    @property
    def final_extracted(self) -> Optional[str]:
        """Judge verdict when a judge ran, rule-based extraction otherwise."""
        # Older reports have a judge token but no explicit completion flag.
        return self.judge_extracted if self.judge_completed or self.judge_extracted is not None else self.extracted

    @property
    def final_correct(self) -> Optional[bool]:
        return self.judge_correct if self.judge_completed or self.judge_extracted is not None else self.correct

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["final_extracted"] = self.final_extracted
        payload["final_correct"] = self.final_correct
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ItemResult:
        fields = {key: raw.get(key) for key in cls.__dataclass_fields__ if key in raw}
        fields.setdefault("audio_out_bytes", 0)
        return cls(**fields)  # type: ignore[arg-type]


def _percentile(values: list[float], fraction: float) -> Optional[float]:
    """Linear-interpolated percentile; ``None`` for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _summarize(results: list[ItemResult]) -> dict[str, Any]:
    total = len(results)
    errors = sum(1 for item in results if item.error)
    parsed = sum(1 for item in results if item.final_extracted is not None)
    correct = sum(1 for item in results if item.final_correct is True and not item.error)
    return {
        "n": total,
        "errors": errors,
        "answered": total - errors,
        "parsed": parsed,
        "correct": correct,
        # Headline number: errors and unparsable replies count against the model,
        # because "never produced a gradable answer" is a real failure to compare.
        "accuracy": (correct / total) if total else None,
        "accuracy_parsed": (correct / parsed) if parsed else None,
        "parse_rate": (parsed / total) if total else None,
    }


def _latency(results: list[ItemResult]) -> dict[str, Any]:
    ttfb = [item.ttfb_s for item in results if item.ttfb_s is not None]
    turn = [item.turn_s for item in results if item.turn_s is not None]
    return {
        "ttfb_p50_s": _percentile(ttfb, 0.5),
        "ttfb_p95_s": _percentile(ttfb, 0.95),
        "turn_p50_s": _percentile(turn, 0.5),
        "turn_p95_s": _percentile(turn, 0.95),
    }


def aggregate(results: list[ItemResult]) -> dict[str, Any]:
    """Overall and per-category scores for one run."""
    by_category: dict[str, Any] = {}
    for category in sorted({item.category for item in results}):
        by_category[category] = _summarize([item for item in results if item.category == category])
    return {"totals": _summarize(results), "by_category": by_category, "latency": _latency(results)}


def build_report(
    *,
    results: list[ItemResult],
    subset: Subset,
    label: str,
    target: dict[str, Any],
    judge: Optional[dict[str, Any]] = None,
    started_at: Optional[datetime] = None,
    duration_s: Optional[float] = None,
) -> dict[str, Any]:
    """Assemble the JSON report for one run."""
    stamp = started_at or datetime.now(timezone.utc)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "label": label,
        "started_at": stamp.astimezone(timezone.utc).isoformat(),
        "duration_s": duration_s,
        "target": target,
        "subset": {
            "name": subset.name,
            "dataset": subset.dataset,
            "revision": subset.revision,
            "n": len(subset),
            "ids": [item.id for item in subset.items],
        },
        "judge": judge,
    }
    report.update(aggregate(results))
    report["items"] = [item.to_dict() for item in results]
    return report


def _pct(value: Optional[float]) -> str:
    return "  n/a" if value is None else f"{value * 100:5.1f}%"


def _secs(value: Optional[float]) -> str:
    return "  n/a" if value is None else f"{value:5.2f}s"


def render_report(report: dict[str, Any]) -> str:
    """Human-readable summary of one run."""
    totals = report["totals"]
    subset = report["subset"]
    latency = report.get("latency", {})
    lines = [
        f"Big Bench Audio vibe check -- {report.get('label') or 'unlabeled'}",
        f"  subset   {subset['name']} ({subset['n']} items) @ {subset['dataset']}@{subset['revision'][:7]}",
        f"  target   {report.get('target', {}).get('url', '?')}",
    ]
    if report.get("judge"):
        lines.append(f"  judge    {report['judge'].get('model', '?')}")
    lines += [
        "",
        f"  {'category':<18} {'acc':>6} {'correct':>9} {'parsed':>8} {'errors':>7}",
        f"  {'-' * 18} {'-' * 6} {'-' * 9} {'-' * 8} {'-' * 7}",
    ]
    for category, stats in report.get("by_category", {}).items():
        lines.append(
            f"  {category:<18} {_pct(stats['accuracy'])} {stats['correct']:>4}/{stats['n']:<4} "
            f"{stats['parsed']:>4}/{stats['n']:<3} {stats['errors']:>7}"
        )
    lines += [
        f"  {'-' * 18} {'-' * 6} {'-' * 9} {'-' * 8} {'-' * 7}",
        f"  {'OVERALL':<18} {_pct(totals['accuracy'])} {totals['correct']:>4}/{totals['n']:<4} "
        f"{totals['parsed']:>4}/{totals['n']:<3} {totals['errors']:>7}",
        "",
        f"  latency  ttfb p50 {_secs(latency.get('ttfb_p50_s'))}  p95 {_secs(latency.get('ttfb_p95_s'))}"
        f"   turn p50 {_secs(latency.get('turn_p50_s'))}  p95 {_secs(latency.get('turn_p95_s'))}",
    ]
    if report.get("duration_s") is not None:
        lines.append(f"  wall     {report['duration_s'] / 60:.1f} min")
    return "\n".join(lines)


@dataclass
class ComparisonRow:
    """One scope (overall, or a category) compared across two runs."""

    scope: str
    baseline_accuracy: Optional[float]
    candidate_accuracy: Optional[float]
    baseline_n: int
    candidate_n: int
    delta: Optional[float] = None


@dataclass
class Comparison:
    """Result of diffing two reports."""

    baseline_label: str
    candidate_label: str
    rows: list[ComparisonRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    latency_delta: dict[str, Optional[float]] = field(default_factory=dict)


def _row(scope: str, base: dict[str, Any], cand: dict[str, Any]) -> ComparisonRow:
    base_acc, cand_acc = base.get("accuracy"), cand.get("accuracy")
    row = ComparisonRow(
        scope=scope,
        baseline_accuracy=base_acc,
        candidate_accuracy=cand_acc,
        baseline_n=base.get("n", 0),
        candidate_n=cand.get("n", 0),
    )
    if base_acc is None or cand_acc is None:
        return row
    row.delta = cand_acc - base_acc
    return row


def compare_reports(baseline: dict[str, Any], candidate: dict[str, Any]) -> Comparison:
    """Show observed score and latency differences, warning about mismatched runs."""
    comparison = Comparison(
        baseline_label=baseline.get("label") or "baseline",
        candidate_label=candidate.get("label") or "candidate",
    )

    base_subset, cand_subset = baseline.get("subset", {}), candidate.get("subset", {})
    if base_subset.get("ids") != cand_subset.get("ids"):
        comparison.warnings.append(
            "Runs used different question sets "
            f"({base_subset.get('name')}/{base_subset.get('n')} vs "
            f"{cand_subset.get('name')}/{cand_subset.get('n')}); accuracies are not comparable."
        )
    if base_subset.get("revision") != cand_subset.get("revision"):
        comparison.warnings.append("Runs used different dataset revisions.")
    if bool(baseline.get("judge")) != bool(candidate.get("judge")):
        comparison.warnings.append("Only one run used an LLM judge; grading is not like-for-like.")

    categories = sorted(set(baseline.get("by_category", {})) | set(candidate.get("by_category", {})))
    for category in categories:
        comparison.rows.append(
            _row(
                category,
                baseline.get("by_category", {}).get(category, {}),
                candidate.get("by_category", {}).get(category, {}),
            )
        )
    comparison.rows.append(_row("OVERALL", baseline.get("totals", {}), candidate.get("totals", {})))

    for key in ("ttfb_p50_s", "turn_p50_s"):
        base_value = baseline.get("latency", {}).get(key)
        cand_value = candidate.get("latency", {}).get(key)
        comparison.latency_delta[key] = None if base_value is None or cand_value is None else cand_value - base_value
    return comparison


def render_comparison(comparison: Comparison) -> str:
    """Human-readable side-by-side of two runs."""
    lines = [
        f"Big Bench Audio vibe check -- {comparison.baseline_label} -> {comparison.candidate_label}",
        "",
        f"  {'category':<18} {'base':>6} {'cand':>6} {'delta':>8}  n (base/cand)",
        f"  {'-' * 18} {'-' * 6} {'-' * 6} {'-' * 8}  {'-' * 24}",
    ]
    for row in comparison.rows:
        delta = "   n/a" if row.delta is None else f"{row.delta * 100:+6.1f}pp"
        separator = f"  {'-' * 18} {'-' * 6} {'-' * 6} {'-' * 8}  {'-' * 24}"
        if row.scope == "OVERALL":
            lines.append(separator)
        lines.append(
            f"  {row.scope:<18} {_pct(row.baseline_accuracy)} {_pct(row.candidate_accuracy)} {delta:>8}"
            f"  {row.baseline_n}/{row.candidate_n}"
        )

    ttfb = comparison.latency_delta.get("ttfb_p50_s")
    turn = comparison.latency_delta.get("turn_p50_s")
    lines += [
        "",
        f"  latency  ttfb p50 {'n/a' if ttfb is None else f'{ttfb:+.2f}s'}"
        f"   turn p50 {'n/a' if turn is None else f'{turn:+.2f}s'}",
    ]
    for warning in comparison.warnings:
        lines.append(f"  WARNING: {warning}")
    return "\n".join(lines)
