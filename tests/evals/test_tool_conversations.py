"""Tool-conversation eval: metrics, comparison and the runner against a scripted server."""

import asyncio
import json
import stat

import pytest

from speech_to_speech.evals.tool_conversations.__main__ import main
from speech_to_speech.evals.tool_conversations.report import fisher_exact, render_comparison, render_report, summarize
from speech_to_speech.evals.tool_conversations.runner import RunnerConfig, run_set
from speech_to_speech.evals.tool_conversations.scenarios import CONVERSATIONS, RESULTS, Turn


def _user(index, expected, *rounds):
    return {
        "kind": "user",
        "index": index,
        "ask": "?",
        "expected_tool": expected,
        "rounds": [{"text": text, "calls": calls} for text, calls in rounds],
    }


def test_summary_counts_each_outcome():
    results = [
        {
            "conversation": 0,
            "rep": 0,
            "errors": [],
            "turns": [
                _user(0, "get_weather", ("I'll check.", [])),  # said it, no call, on the first turn
                _user(1, "dance", ("Sure!", ["dance"])),  # spoke before an expression tool
                _user(2, "get_time", ("", ["get_time"]), ("It's two.", [])),
                _user(3, None, ("", ["search_web"]), ("Lima.", [])),  # call when none was needed
                _user(4, "set_timer", ("Done. <call:set_timer{minutes:3}>", [])),  # markup spoken
                {"kind": "late_result", "tool": "research_report", "text": "Ready!", "calls": []},
            ],
        },
        {"conversation": 1, "rep": 0, "crash": "ConnectionClosedError: boom"},
    ]
    s = summarize(results)
    assert (s["tool_turns"], s["right_tool"], s["said_no_call"]) == (4, 2, 2)
    assert (s["first_tool_turns"], s["first_turn_misses"]) == (1, 1)
    assert (s["spoke_first"], s["expression_turns"], s["spoke_first_expression"]) == (1, 1, 1)
    assert (s["free_turns"], s["extra_call"], s["call_markup_spoken"]) == (1, 1, 1)
    assert (s["late_results"], s["late_result_spoken"], s["repeated_long_running"]) == (1, 1, 0)
    assert (s["conversations"], s["crashed"]) == (2, 1)


def test_argument_scores_distinguish_selection_schema_and_values():
    timer = _user(0, "set_timer")
    timer["expected_arguments"] = {"minutes": 10}
    retries = [
        '{"minutes":60,"label":"eggs"}',
        "{}",
        '{"minutes":10}',
        '{"minutes":"10","label":"eggs"}',
        '{"minutes":10,"label":"eggs","extra":NaN}',
        "{",
        "[]",
        "",
    ]
    timer["rounds"] = [
        {
            "text": "",
            "calls": ["get_time", "set_timer"],
            "arguments": ['{"city":"Tokyo"}', '{"minutes":10,"label":"eggs"}'],
        },
        {"text": "", "calls": ["set_timer"] * len(retries), "arguments": retries},
    ]
    search = _user(1, "search_web")
    search["rounds"] = [{"text": "", "calls": ["search_web"], "arguments": ['{"query":"lunar landing"}']}]
    miss = _user(2, "get_weather", ("Sorry.", []))
    miss["expected_arguments"] = {"city": "Lisbon"}
    s = summarize([{"turns": [timer, search, miss]}])
    assert (s["right_tool"], s["tool_turns"]) == (2, 3)
    assert (s["valid_arguments"], s["argument_calls"]) == (3, 10)
    assert (s["correct_arguments"], s["scored_argument_calls"]) == (1, 9)


def test_argument_values_accept_any_listed_alternative():
    weather = _user(0, "get_weather")
    # A report stores the scenario's tuple of accepted values as a JSON list.
    weather["expected_arguments"] = {"city": ["Quebec", "Quebec City"]}
    weather["rounds"] = [
        {
            "text": "",
            "calls": ["get_weather"] * 3,
            "arguments": ['{"city":"quebec city"}', '{"city":"Quebec"}', '{"city":"Montreal"}'],
        }
    ]
    s = summarize([{"turns": [weather]}])
    assert (s["correct_arguments"], s["scored_argument_calls"]) == (2, 3)


_FALLBACK = "I'm having trouble responding right now. Please try again."


def _failed_turns_row():
    failed = _user(0, "set_timer", (_FALLBACK, []))
    failed["rounds"][0]["status"] = "failed"
    called_then_failed = _user(1, "get_time", ("", ["get_time"]), (_FALLBACK, []))
    called_then_failed["rounds"][1]["status"] = "failed"
    free = _user(2, None, (_FALLBACK, []))
    free["rounds"][0]["status"] = "failed"
    answered = _user(3, "dance", ("", ["dance"]))
    answered["rounds"][0]["status"] = "completed"
    late = {"kind": "late_result", "tool": "research_report", "text": _FALLBACK, "calls": [], "status": "failed"}
    return {
        "conversation": 0,
        "rep": 0,
        "errors": ["response_failed"] * 4,
        "turns": [failed, called_then_failed, free, answered, late],
    }


def test_failed_responses_are_counted_apart_from_model_behaviour():
    s = summarize([_failed_turns_row()])
    assert (s["failed_turns"], s["attempted_turns"]) == (4, 5)
    # Only the completed turn is scored; the fallback apology is not "said it, no call".
    assert (s["right_tool"], s["tool_turns"], s["said_no_call"]) == (1, 1, 0)
    assert (s["user_turns"], s["free_turns"], s["late_results"]) == (1, 0, 0)


def test_comparison_recounts_reports_from_their_turns():
    row = _failed_turns_row()
    stale = summarize([row])
    stale.update(failed_turns=0, tool_turns=3, said_no_call=2)  # as written before failed turns were set apart
    table = render_comparison(
        [{"label": "old", "summary": stale, "results": [row]}, {"label": "new", "summary": stale}]
    )
    assert "| Said it, no call | 0/1 (0.0%) | 2/3 (66.7%) p=1 |" in table
    assert "| Turns whose response failed | 4/5 (80.0%) | 0/5 (0.0%) p=0.048 |" in table


def test_fisher_exact_matches_known_values():
    assert fisher_exact(5, 5, 5, 5) == pytest.approx(1.0)
    assert fisher_exact(0, 10, 10, 0) == pytest.approx(1.0825e-05, rel=1e-3)
    assert fisher_exact(4, 172, 0, 176) == pytest.approx(0.1229, rel=1e-2)


def test_comparison_renders_one_column_per_report():
    current = {"label": "new", "summary": summarize([])}
    legacy = {"label": "old", "summary": dict(current["summary"])}
    for key in ("valid_arguments", "argument_calls", "correct_arguments", "scored_argument_calls"):
        del legacy["summary"][key]
    current["summary"].update(valid_arguments=1, argument_calls=2, correct_arguments=1, scored_argument_calls=2)
    assert "| Expected argument values matched | - |" in render_report(legacy)
    for reports, cells in (([legacy, current], "- | 1/2 (50.0%)"), ([current, legacy], "1/2 (50.0%) | -")):
        table = render_comparison(reports)
        assert table.splitlines()[0] == f"| Metric | {reports[0]['label']} | {reports[1]['label']} |"
        assert f"| Expected argument values matched | {cells} |" in table


class _ScriptedServer:
    """Emits a report call and a weather call, then records when the client returns each result."""

    def __init__(self):
        self.received = []

    async def handler(self, ws):
        responses = 0
        async for message in ws:
            event = json.loads(message)
            self.received.append(event)
            if event["type"] != "response.create":
                continue
            responses += 1
            if responses == 1:
                output = [{"type": "function_call", "name": "research_report", "call_id": "call_r", "arguments": "{}"}]
            elif responses == 2:
                output = [
                    {
                        "type": "function_call",
                        "name": "get_weather",
                        "call_id": "call_w",
                        "arguments": '{"city":" LISBON "}',
                    }
                ]
            else:
                output = [{"type": "message", "content": [{"type": "output_text", "text": f"reply {responses}"}]}]
            await ws.send(json.dumps({"type": "response.done", "response": {"status": "completed", "output": output}}))


def test_runner_sends_the_long_running_result_two_turns_later():
    import websockets

    server = _ScriptedServer()
    conversation = (Turn("Write a report.", "research_report"), CONVERSATIONS[0][0], Turn("And now?", None))

    async def scenario():
        async with websockets.serve(server.handler, "127.0.0.1", 0) as ws_server:
            port = ws_server.sockets[0].getsockname()[1]
            return await run_set(RunnerConfig(url=f"ws://127.0.0.1:{port}"), (conversation,), reps=1)

    [row] = asyncio.run(scenario())
    kinds = [turn["kind"] for turn in row["turns"]]
    assert kinds == ["user", "user", "late_result", "user"]
    outputs = [e["item"] for e in server.received if e.get("item", {}).get("type") == "function_call_output"]
    assert [(o["call_id"], json.loads(o["output"])) for o in outputs] == [
        ("call_w", RESULTS["get_weather"]),
        ("call_r", RESULTS["research_report"]),
    ]
    s = summarize([row])
    assert (s["right_tool"], s["late_results"], s["late_result_spoken"]) == (2, 1, 1)
    assert (s["valid_arguments"], s["argument_calls"]) == (1, 2)
    assert (s["correct_arguments"], s["scored_argument_calls"]) == (1, 1)


def test_server_config_is_private_and_points_speech_at_the_stand_in(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_KEY", "secret-value")
    path = tmp_path / "cfg.json"
    assert (
        main(
            [
                "server-config",
                "--out",
                str(path),
                "--model",
                "m",
                "--base-url",
                "http://llm/v1",
                "--api-key-env",
                "LLM_KEY",
                "--speech-port",
                "9999",
            ]
        )
        == 0
    )
    config = json.loads(path.read_text())
    assert config["openai_tts_base_url"] == config["openai_stt_base_url"] == "http://127.0.0.1:9999/v1"
    assert config["responses_api_api_key"] == "secret-value"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
