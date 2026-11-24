"""Metrics, rendering and comparison for tool-conversation runs.

Reports are plain JSON holding every conversation, so every number here can be recomputed
from ``results``. Rates are counts with their denominators; comparisons add a two-sided Fisher
exact test so a difference of a few cases is not read as a regression.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from jsonschema.validators import validator_for

from speech_to_speech.evals.tool_conversations.scenarios import EXPRESSION_TOOLS, LONG_RUNNING_TOOLS, TOOLS

REPORT_SCHEMA = "tool-conversations/1"

_ARGUMENT_VALIDATORS = {tool["name"]: validator_for(tool["parameters"])(tool["parameters"]) for tool in TOOLS}

# Tool-call markup written into spoken text instead of being parsed as a call.
CALL_MARKUP = re.compile(r"<\|?tool_call|call:\w+\{|<\|\"\|>|<function=|\[TOOL_CALLS\]", re.IGNORECASE)

# (key, label, numerator, denominator) — what a person reads in the table.
METRICS = (
    ("right_tool", "Right tool called", "right_tool", "tool_turns"),
    ("valid_arguments", "Schema-valid arguments", "valid_arguments", "argument_calls"),
    ("correct_arguments", "Expected argument values matched", "correct_arguments", "scored_argument_calls"),
    ("said_no_call", "Said it, no call", "said_no_call", "tool_turns"),
    ("first_turn_misses", "  of which on a conversation's first turn", "first_turn_misses", "first_tool_turns"),
    ("spoke_first", "Spoke before calling", "spoke_first", "tool_turns"),
    ("spoke_first_expression", "Spoke before an expression tool", "spoke_first_expression", "expression_turns"),
    ("call_markup_spoken", "Call markup in spoken text", "call_markup_spoken", "user_turns"),
    ("extra_call", "Call when none was needed", "extra_call", "free_turns"),
    ("late_result_spoken", "Late result spoken", "late_result_spoken", "late_results"),
    ("late_result_recalled", "Long-running tool called again", "repeated_long_running", "late_results"),
    ("crashed", "Conversations that crashed", "crashed", "conversations"),
)


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"Invalid JSON constant: {token}")


def _normalized(value: Any) -> Any:
    return value.strip().casefold() if isinstance(value, str) else value


def _matches(actual: Any, expected: Any) -> bool:
    # A tuple lists every accepted value; reports store it as a JSON list.
    accepted = expected if isinstance(expected, (list, tuple)) else (expected,)
    return any(_normalized(actual) == _normalized(value) for value in accepted)


def _count_arguments(turn: dict[str, Any], counts: dict[str, int]) -> None:
    tool = turn["expected_tool"]
    validator = _ARGUMENT_VALIDATORS.get(tool)
    if validator is None:
        return
    expected = turn.get("expected_arguments")
    for response in turn["rounds"]:
        for name, raw in zip(response["calls"], response.get("arguments", [])):
            if name != tool:
                continue
            counts["argument_calls"] += 1
            counts["scored_argument_calls"] += expected is not None
            if not isinstance(raw, str):
                continue
            try:
                arguments = json.loads(raw, parse_constant=_reject_json_constant)
            except (ValueError, RecursionError):
                continue
            if not validator.is_valid(arguments):
                continue
            counts["valid_arguments"] += 1
            if expected is not None:
                counts["correct_arguments"] += all(
                    key in arguments and _matches(arguments[key], value) for key, value in expected.items()
                )


def summarize(results: list[dict[str, Any]]) -> dict[str, int]:
    """Count outcomes over all conversations."""
    c: dict[str, int] = {key: 0 for metric in METRICS for key in metric[2:]}
    c["server_errors"] = 0
    for row in results:
        c["conversations"] += 1
        if "crash" in row:
            c["crashed"] += 1
            continue
        c["server_errors"] += len(row.get("errors") or [])
        for turn in row.get("turns") or []:
            if turn["kind"] == "late_result":
                c["late_results"] += 1
                c["late_result_spoken"] += bool(turn["text"])
                c["repeated_long_running"] += any(name in LONG_RUNNING_TOOLS for name in turn["calls"])
                continue
            c["user_turns"] += 1
            rounds = turn["rounds"]
            first = rounds[0] if rounds else {"text": "", "calls": []}
            called = [name for r in rounds for name in r["calls"]]
            expected = turn["expected_tool"]
            c["call_markup_spoken"] += any(CALL_MARKUP.search(r["text"] or "") for r in rounds)
            long_calls = sum(called.count(name) for name in LONG_RUNNING_TOOLS)
            c["repeated_long_running"] += max(0, long_calls - (expected in LONG_RUNNING_TOOLS))
            if expected:
                _count_arguments(turn, c)
                c["tool_turns"] += 1
                c["right_tool"] += expected in called
                missed = not first["calls"] and bool(first["text"])
                c["said_no_call"] += missed
                c["spoke_first"] += bool(first["calls"]) and bool(first["text"])
                if turn["index"] == 0:
                    c["first_tool_turns"] += 1
                    c["first_turn_misses"] += missed
                if expected in EXPRESSION_TOOLS:
                    c["expression_turns"] += 1
                    c["spoke_first_expression"] += bool(first["calls"]) and bool(first["text"])
            else:
                c["free_turns"] += 1
                c["extra_call"] += any(name not in LONG_RUNNING_TOOLS for name in called)
    return c


def build_report(meta: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema": REPORT_SCHEMA, **meta, "summary": summarize(results), "results": results}


def _rate(summary: dict[str, int], num: str, den: str) -> str:
    if not summary.get(den):
        return "-"
    return f"{summary[num]}/{summary[den]} ({100 * summary[num] / summary[den]:.1f}%)"


def render_report(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        f"## {report.get('label') or report.get('url')}",
        "",
        f"{s['conversations']} conversations, set `{report.get('scenario_set')}`, {s['server_errors']} server errors",
        "",
        "| Metric | Result |",
        "|---|---|",
    ]
    lines += [f"| {label} | {_rate(s, num, den)} |" for _key, label, num, den in METRICS]
    return "\n".join(lines)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for the table [[a, b], [c, d]]."""
    n1, n2, k = a + b, c + d, a + c
    total = math.comb(n1 + n2, k)

    def p(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, k - x) / total

    observed = p(a)
    low, high = max(0, k - n2), min(k, n1)
    return min(1.0, sum(p(x) for x in range(low, high + 1) if p(x) <= observed * (1 + 1e-9)))


def render_comparison(reports: list[dict[str, Any]]) -> str:
    """One column per report; the p-value compares the first report with each other one."""
    names = [r.get("label") or r.get("url") or f"run {i}" for i, r in enumerate(reports)]
    header = "| Metric | " + " | ".join(names) + " |"
    lines = [header, "|" + "---|" * (len(reports) + 1)]
    base = reports[0]["summary"]
    for _key, label, num, den in METRICS:
        cells = []
        for i, report in enumerate(reports):
            s = report["summary"]
            cell = _rate(s, num, den)
            if i and s.get(den) and base.get(den):
                pval = fisher_exact(base[num], base[den] - base[num], s[num], s[den] - s[num])
                cell += f" p={pval:.2g}"
            cells.append(cell)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
