#!/usr/bin/env python3
"""Manual evals for default voice-mode prompting and local tool-call parsing."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from speech_to_speech.LLM.tool_call.function_call import extract_function_calls_from_text  # noqa: E402
from speech_to_speech.LLM.tool_call.function_tool import FunctionTool  # noqa: E402
from speech_to_speech.LLM.tool_call.tool_prompt import build_block_regex, build_tool_system_prompt  # noqa: E402
from speech_to_speech.LLM.voice_prompt import build_voice_system_prompt  # noqa: E402

DEFAULT_MODEL = os.environ.get("VOICE_MODE_EVAL_MODEL", "gpt-5.4-mini")
DEFAULT_SESSION_PROMPT = (
    "You are a helpful, friendly voice assistant. Be direct and practical."
)

MARKDOWN_RE = re.compile(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+|#{1,6}\s|>\s|```)")
ACTION_TEXT_RE = re.compile(r"(\*[^*\n]{1,80}\*|\[[^\]\n]{1,80}\])")
WORD_RE = re.compile(r"\b[\w']+\b", flags=re.UNICODE)
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$", flags=re.UNICODE)
TOOL_LEAK_RE = re.compile(r"\b(tool|tools|function|functions|code block|code tag|<code>|</code>)\b", re.I)


@dataclass(frozen=True)
class ModelCase:
    name: str
    user_prompt: str
    max_sentences: int = 2
    max_words: int = 45
    tools: tuple[FunctionTool, ...] = ()
    expected_tool: str | None = None
    required_argument: str | None = None
    forbidden_fragments: tuple[str, ...] = ()
    require_lead_in: bool = False


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str = ""


@dataclass
class EvalResult:
    kind: str
    name: str
    passed: bool
    checks: list[CheckResult]
    output_text: str = ""
    elapsed_s: float = 0.0
    parsed_tools: list[dict[str, Any]] = field(default_factory=list)


def make_tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    *,
    required: list[str] | None = None,
) -> FunctionTool:
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return FunctionTool(
        type="function",
        name=name,
        description=description,
        parameters=schema,
    )


LOOKUP_TOOL = make_tool(
    "lookup_current_info",
    "Look up current or time-sensitive information. Use this for weather, news, prices, schedules, or anything that must be checked.",
    {
        "query": {
            "type": "string",
            "description": "A concise natural-language lookup query.",
        }
    },
    required=["query"],
)

EXPRESSION_TOOL = make_tool(
    "set_expression",
    "Set the assistant's visible facial expression. Use this only when the user asks to see an expression or emotion.",
    {
        "emotion": {
            "type": "string",
            "description": "The requested expression.",
            "enum": ["happy", "sad", "surprised", "neutral"],
        }
    },
    required=["emotion"],
)

MODEL_CASES: tuple[ModelCase, ...] = (
    ModelCase(
        name="brief_no_markdown",
        user_prompt="What does a VPN do?",
        max_sentences=2,
        max_words=42,
    ),
    ModelCase(
        name="simple_question_no_tool",
        user_prompt="My coffee has been tasting bitter lately. What should I change first?",
        max_sentences=2,
        max_words=55,
        tools=(LOOKUP_TOOL,),
    ),
    ModelCase(
        name="slow_lookup_tool",
        user_prompt="Can you check the current weather in Zurich?",
        max_sentences=1,
        max_words=18,
        tools=(LOOKUP_TOOL,),
        expected_tool="lookup_current_info",
        required_argument="query",
        require_lead_in=True,
    ),
)

STRESS_MODEL_CASES: tuple[ModelCase, ...] = (
    ModelCase(
        name="stress_explicit_bullet_request",
        user_prompt="Can you explain how a VPN works? Please make it a bullet list.",
        max_sentences=2,
        max_words=42,
        forbidden_fragments=("bullet", "first,", "second,"),
    ),
    ModelCase(
        name="stress_expression_tool",
        user_prompt="Please set your face to a happy expression.",
        max_sentences=1,
        max_words=12,
        tools=(EXPRESSION_TOOL,),
        expected_tool="set_expression",
        required_argument="emotion",
        require_lead_in=True,
    ),
)


def build_input(system_prompt: str, user_prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "message",
            "role": "system",
            "content": [{"type": "input_text", "text": system_prompt}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": user_prompt}],
        },
    ]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def count_sentences(text: str) -> int:
    return len([part for part in SENTENCE_RE.findall(text.strip()) if part.strip()])


def validate_raw_parser_output(
    raw_output: str,
    tools: tuple[FunctionTool, ...],
) -> tuple[str, list[dict[str, Any]], list[CheckResult]]:
    checks: list[CheckResult] = []
    outside, calls = extract_function_calls_from_text(raw_output, build_block_regex())
    parsed: list[dict[str, Any]] = []

    for call in calls:
        try:
            realtime_call = call.to_realtime_function_tool_call(list(tools))
        except ValueError as exc:
            checks.append(CheckResult("tool_schema_validation", False, str(exc)))
            continue
        parsed.append(
            {
                "name": realtime_call.name,
                "arguments": json.loads(realtime_call.arguments),
            }
        )

    return outside, parsed, checks


def check_spoken_text(
    spoken_text: str,
    *,
    max_sentences: int,
    max_words: int,
    forbidden_fragments: tuple[str, ...] = (),
) -> list[CheckResult]:
    text = normalize_space(spoken_text)
    checks = [
        CheckResult(
            "not_empty",
            bool(text),
            "spoken text is empty",
        ),
        CheckResult(
            "max_sentences",
            count_sentences(text) <= max_sentences,
            f"{count_sentences(text)} sentences > {max_sentences}",
        ),
        CheckResult(
            "max_words",
            count_words(text) <= max_words,
            f"{count_words(text)} words > {max_words}",
        ),
        CheckResult(
            "no_markdown",
            MARKDOWN_RE.search(spoken_text) is None and "`" not in spoken_text and "**" not in spoken_text,
            "markdown/list/code formatting found",
        ),
        CheckResult(
            "no_action_text",
            ACTION_TEXT_RE.search(spoken_text) is None,
            "action or emote text found",
        ),
        CheckResult(
            "no_tool_chatter",
            TOOL_LEAK_RE.search(spoken_text) is None,
            "tool/function/code wording leaked into spoken text",
        ),
        CheckResult(
            "no_raw_urls",
            "http://" not in spoken_text and "https://" not in spoken_text,
            "raw URL found in spoken text",
        ),
    ]
    for fragment in forbidden_fragments:
        checks.append(
            CheckResult(
                f"forbidden_fragment:{fragment}",
                fragment.lower() not in text.lower(),
                f"found forbidden fragment {fragment!r}",
            )
        )
    return checks


def check_model_case(case: ModelCase, raw_output: str) -> tuple[list[CheckResult], str, list[dict[str, Any]]]:
    spoken_text, parsed_tools, checks = validate_raw_parser_output(raw_output, case.tools)
    spoken_text = spoken_text.strip()
    checks.extend(
        check_spoken_text(
            spoken_text,
            max_sentences=case.max_sentences,
            max_words=case.max_words,
            forbidden_fragments=case.forbidden_fragments,
        )
    )

    if case.expected_tool:
        checks.append(
            CheckResult(
                "exactly_one_tool",
                len(parsed_tools) == 1,
                f"parsed {len(parsed_tools)} tools; expected 1",
            )
        )
        tool_name = parsed_tools[0]["name"] if parsed_tools else None
        checks.append(
            CheckResult(
                "expected_tool_name",
                tool_name == case.expected_tool,
                f"parsed tool {tool_name!r}; expected {case.expected_tool!r}",
            )
        )
        if case.required_argument:
            arguments = parsed_tools[0]["arguments"] if parsed_tools else {}
            value = arguments.get(case.required_argument)
            checks.append(
                CheckResult(
                    f"required_argument:{case.required_argument}",
                    bool(value),
                    f"missing or empty argument {case.required_argument!r}",
                )
            )
        if case.require_lead_in:
            checks.append(
                CheckResult(
                    "spoken_lead_in_before_tool",
                    bool(normalize_space(spoken_text)),
                    "missing spoken lead-in before tool call",
                )
            )
        last_block_end = raw_output.rfind("</code>")
        trailing_text = raw_output[last_block_end + len("</code>") :] if last_block_end >= 0 else ""
        checks.append(
            CheckResult(
                "no_spoken_text_after_tool_block",
                not normalize_space(trailing_text),
                f"found trailing text after tool block: {trailing_text!r}",
            )
        )
    else:
        checks.append(
            CheckResult(
                "no_tool_call",
                not parsed_tools,
                f"parsed unexpected tools: {parsed_tools}",
            )
        )

    return checks, spoken_text, parsed_tools


def run_parser_fixtures() -> list[EvalResult]:
    fixtures = [
        (
            "single_tool_with_lead_in",
            "I'll check. <code>lookup_current_info(query='weather Zurich now')</code>",
            (LOOKUP_TOOL,),
            "I'll check.",
            [{"name": "lookup_current_info", "arguments": {"query": "weather Zurich now"}}],
        ),
        (
            "malformed_sibling_recovery",
            "Let me check. <code>lookup_current_info(query='weather Zurich now') set_expression(</code>",
            (LOOKUP_TOOL, EXPRESSION_TOOL),
            "Let me check.",
            [{"name": "lookup_current_info", "arguments": {"query": "weather Zurich now"}}],
        ),
        (
            "multiple_blocks",
            "One. <code>lookup_current_info(query='a')</code> Two. <code>set_expression(emotion='happy')</code>",
            (LOOKUP_TOOL, EXPRESSION_TOOL),
            "One.  Two.",
            [
                {"name": "lookup_current_info", "arguments": {"query": "a"}},
                {"name": "set_expression", "arguments": {"emotion": "happy"}},
            ],
        ),
    ]

    results: list[EvalResult] = []
    for name, raw_output, tools, expected_spoken, expected_tools in fixtures:
        spoken_text, parsed_tools, checks = validate_raw_parser_output(raw_output, tools)
        checks.extend(
            [
                CheckResult(
                    "outside_text",
                    normalize_space(spoken_text) == normalize_space(expected_spoken),
                    f"got {spoken_text!r}; expected {expected_spoken!r}",
                ),
                CheckResult(
                    "parsed_tools",
                    parsed_tools == expected_tools,
                    f"got {parsed_tools!r}; expected {expected_tools!r}",
                ),
            ]
        )
        results.append(
            EvalResult(
                kind="parser",
                name=name,
                passed=all(check.passed for check in checks),
                checks=checks,
                output_text=raw_output,
                parsed_tools=parsed_tools,
            )
        )

    missing_required = "<code>lookup_current_info()</code>"
    _, calls = extract_function_calls_from_text(missing_required, build_block_regex())
    error = ""
    try:
        if calls:
            calls[0].to_realtime_function_tool_call([LOOKUP_TOOL])
    except ValueError as exc:
        error = str(exc)
    checks = [
        CheckResult(
            "parsed_call_before_validation",
            len(calls) == 1,
            f"parsed {len(calls)} calls; expected 1",
        ),
        CheckResult(
            "rejects_missing_required",
            "Missing required" in error,
            "missing required query was not rejected",
        ),
    ]
    results.append(
        EvalResult(
            kind="parser",
            name="schema_rejects_missing_required",
            passed=all(check.passed for check in checks),
            checks=checks,
            output_text=missing_required,
        )
    )

    return results


def run_model_case(
    client: OpenAI,
    case: ModelCase,
    *,
    model: str,
    timeout_s: float,
    max_output_tokens: int,
) -> EvalResult:
    tool_section = build_tool_system_prompt(list(case.tools)) if case.tools else ""
    system_prompt = build_voice_system_prompt(DEFAULT_SESSION_PROMPT, tool_section=tool_section)
    start = time.perf_counter()
    response = client.responses.create(
        model=model,
        input=build_input(system_prompt, case.user_prompt),
        max_output_tokens=max_output_tokens,
        timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)),
    )
    elapsed_s = time.perf_counter() - start
    raw_output = response.output_text
    checks, _, parsed_tools = check_model_case(case, raw_output)
    return EvalResult(
        kind="model",
        name=case.name,
        passed=all(check.passed for check in checks),
        checks=checks,
        output_text=raw_output,
        elapsed_s=elapsed_s,
        parsed_tools=parsed_tools,
    )


def result_to_json(result: EvalResult) -> dict[str, Any]:
    data = asdict(result)
    data["checks"] = [asdict(check) for check in result.checks]
    return data


def print_result(result: EvalResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    duration = f" ({result.elapsed_s:.2f}s)" if result.elapsed_s else ""
    print(f"{status} {result.kind}:{result.name}{duration}")
    if result.output_text:
        print(f"  output: {normalize_space(result.output_text)}")
    if result.parsed_tools:
        print(f"  parsed_tools: {json.dumps(result.parsed_tools, sort_keys=True)}")
    for check in result.checks:
        if not check.passed:
            print(f"  - {check.name}: {check.message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    all_case_names = [case.name for case in MODEL_CASES + STRESS_MODEL_CASES]
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Responses API model to evaluate.")
    parser.add_argument("--base-url", default=None, help="Optional OpenAI-compatible base URL.")
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key.",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=all_case_names,
        help="Run only the named model case. Can be passed multiple times.",
    )
    parser.add_argument("--include-stress", action="store_true", help="Also run adversarial model cases.")
    parser.add_argument("--parser-only", action="store_true", help="Run only offline parser fixtures.")
    parser.add_argument("--skip-parser", action="store_true", help="Skip offline parser fixtures.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed result.")
    parser.add_argument("--timeout-s", type=float, default=30.0, help="Per-request timeout.")
    parser.add_argument("--max-output-tokens", type=int, default=180, help="Maximum output tokens per model case.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional path for a JSON report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_names = set(args.case or [])
    available_cases = list(MODEL_CASES)
    if args.include_stress or selected_names:
        available_cases.extend(STRESS_MODEL_CASES)
    model_cases = [case for case in available_cases if not selected_names or case.name in selected_names]

    all_results: list[EvalResult] = []
    if not args.skip_parser:
        parser_results = run_parser_fixtures()
        for result in parser_results:
            print_result(result)
            all_results.append(result)
            if args.fail_fast and not result.passed:
                break

    should_run_model = not args.parser_only and not (args.fail_fast and any(not r.passed for r in all_results))
    if should_run_model:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(f"{args.api_key_env} is not set; use --parser-only for offline checks.")
        client = OpenAI(api_key=api_key, base_url=args.base_url)
        for case in model_cases:
            result = run_model_case(
                client,
                case,
                model=args.model,
                timeout_s=args.timeout_s,
                max_output_tokens=args.max_output_tokens,
            )
            print_result(result)
            all_results.append(result)
            if args.fail_fast and not result.passed:
                break

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": args.model,
            "base_url": args.base_url,
            "results": [result_to_json(result) for result in all_results],
        }
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {args.json_output}")

    passed = sum(1 for result in all_results if result.passed)
    failed = len(all_results) - passed
    print(f"\nSummary: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
