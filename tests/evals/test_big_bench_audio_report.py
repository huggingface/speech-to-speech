"""Scores aggregate the way the report claims; comparisons show observed differences."""

import pytest

from speech_to_speech.evals.big_bench_audio.dataset import EvalItem, Subset
from speech_to_speech.evals.big_bench_audio.report import (
    REPORT_SCHEMA,
    ItemResult,
    aggregate,
    build_report,
    compare_reports,
    render_comparison,
    render_report,
)


def make_item(category="navigate", *, correct=None, error=None, extracted=None, ttfb=None, turn=None):
    return ItemResult(
        id=1,
        category=category,
        official_answer="Yes",
        extracted=extracted if extracted is not None else ("yes" if correct is not None else None),
        correct=correct,
        error=error,
        ttfb_s=ttfb,
        turn_s=turn,
    )


def make_subset(n=4):
    items = tuple(
        EvalItem(id=index, category="navigate", official_answer="Yes", file_name=f"data/question_{index}.mp3")
        for index in range(n)
    )
    return Subset(name="test", items=items)


def test_errors_and_unparsed_replies_count_against_accuracy():
    results = [
        make_item(correct=True),
        make_item(correct=False),
        make_item(error="response_timeout"),
        make_item(),
    ]
    totals = aggregate(results)["totals"]

    assert totals == {
        "n": 4,
        "errors": 1,
        "answered": 3,
        "parsed": 2,
        "correct": 1,
        "accuracy": 0.25,
        "accuracy_parsed": 0.5,
        "parse_rate": 0.5,
    }


def test_judge_verdict_overrides_the_rule_based_one():
    result = make_item(correct=False, extracted="no")
    result.judge_extracted = "yes"
    result.judge_correct = True

    assert result.final_correct is True
    assert aggregate([result])["totals"]["correct"] == 1


def test_categories_are_scored_separately():
    results = [
        make_item("navigate", correct=True),
        make_item("navigate", correct=False),
        make_item("web_of_lies", correct=True),
    ]
    by_category = aggregate(results)["by_category"]

    assert by_category["navigate"]["accuracy"] == 0.5
    assert by_category["web_of_lies"]["accuracy"] == 1.0


def test_latency_percentiles_ignore_items_that_never_answered():
    results = [make_item(correct=True, ttfb=1.0, turn=3.0), make_item(error="no_response")]
    latency = aggregate(results)["latency"]

    assert latency["ttfb_p50_s"] == 1.0
    assert latency["turn_p50_s"] == 3.0


def test_report_records_the_exact_questions_that_ran():
    report = build_report(
        results=[make_item(correct=True)],
        subset=make_subset(3),
        label="baseline",
        target={"url": "ws://localhost:8765/v1/realtime"},
    )

    assert report["schema"] == REPORT_SCHEMA
    assert report["subset"]["ids"] == [0, 1, 2]
    assert report["totals"]["correct"] == 1
    assert "Big Bench Audio" in render_report(report)


def _report(label, correct, total, *, subset=None, latency=None):
    results = [make_item(correct=index < correct) for index in range(total)]
    report = build_report(
        results=results,
        subset=subset or make_subset(total),
        label=label,
        target={"url": "ws://x"},
    )
    if latency is not None:
        report["latency"]["ttfb_p50_s"] = latency
    return report


@pytest.mark.parametrize("before, after, total, delta", [(20, 22, 40, 0.05), (38, 14, 40, -0.6), (4, 0, 4, -1.0)])
def test_comparisons_show_observed_deltas_and_sample_sizes(before, after, total, delta):
    comparison = compare_reports(_report("main", before, total), _report("pr", after, total))
    overall = comparison.rows[-1]

    assert overall.scope == "OVERALL"
    assert overall.delta == pytest.approx(delta)
    assert overall.baseline_n == overall.candidate_n == total
    rendered = render_comparison(comparison)
    assert f"{delta * 100:+.1f}pp" in rendered
    assert f"{total}/{total}" in rendered


def test_comparing_different_question_sets_warns_loudly():
    baseline = _report("main", 2, 4, subset=make_subset(4))
    candidate = _report("pr", 2, 3, subset=make_subset(3))
    comparison = compare_reports(baseline, candidate)

    assert any("different question sets" in warning for warning in comparison.warnings)
    assert "WARNING" in render_comparison(comparison)


def test_judging_only_one_side_warns():
    baseline = _report("main", 2, 4)
    candidate = _report("pr", 2, 4)
    candidate["judge"] = {"model": "gpt-4o-mini"}

    assert any("judge" in warning for warning in compare_reports(baseline, candidate).warnings)


def test_latency_delta_is_candidate_minus_baseline():
    comparison = compare_reports(_report("main", 2, 4, latency=1.0), _report("pr", 2, 4, latency=1.75))

    assert comparison.latency_delta["ttfb_p50_s"] == 0.75


def test_item_results_survive_a_json_round_trip():
    original = make_item(correct=True, ttfb=1.25)
    restored = ItemResult.from_dict(original.to_dict())

    assert restored.correct is True
    assert restored.ttfb_s == 1.25


def test_older_reports_without_judge_completion_flag_keep_their_verdict():
    raw = make_item(correct=True).to_dict()
    raw.pop("judge_completed")
    assert ItemResult.from_dict(raw).final_correct is True
    raw.update(judge_extracted="no", judge_correct=False)
    restored = ItemResult.from_dict(raw)
    assert restored.final_extracted == "no"
    assert restored.final_correct is False


def test_failed_turn_with_correct_transcript_does_not_pass():
    result = make_item(correct=True, error="split_turn (2 responses)")
    totals = aggregate([result])["totals"]
    assert totals["correct"] == 0
    assert totals["accuracy"] == 0
    assert totals["errors"] == 1
