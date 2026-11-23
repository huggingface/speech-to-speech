"""Tool-conversation eval: metrics, comparison and the runner against a scripted server."""

import asyncio
import json
import stat

import pytest

from speech_to_speech.evals.tool_conversations.__main__ import main
from speech_to_speech.evals.tool_conversations.report import fisher_exact, render_comparison, summarize
from speech_to_speech.evals.tool_conversations.runner import RunnerConfig, run_set
from speech_to_speech.evals.tool_conversations.scenarios import RESULTS, Turn


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


def test_fisher_exact_matches_known_values():
    assert fisher_exact(5, 5, 5, 5) == pytest.approx(1.0)
    assert fisher_exact(0, 10, 10, 0) == pytest.approx(1.0825e-05, rel=1e-3)
    assert fisher_exact(4, 172, 0, 176) == pytest.approx(0.1229, rel=1e-2)


def test_comparison_renders_one_column_per_report():
    report = {"label": "a", "summary": summarize([])}
    table = render_comparison([report, {**report, "label": "b"}])
    assert table.splitlines()[0] == "| Metric | a | b |"


class _ScriptedServer:
    """Answers like a Realtime server: a report call, then speech; records what the client sent."""

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
            output = (
                [{"type": "function_call", "name": "research_report", "call_id": "call_r", "arguments": "{}"}]
                if responses == 1
                else [{"type": "message", "content": [{"type": "output_text", "text": f"reply {responses}"}]}]
            )
            await ws.send(json.dumps({"type": "response.done", "response": {"status": "completed", "output": output}}))


def test_runner_sends_the_long_running_result_two_turns_later():
    import websockets

    server = _ScriptedServer()
    conversation = (Turn("Write a report.", "research_report"), Turn("Hi?", None), Turn("And now?", None))

    async def scenario():
        async with websockets.serve(server.handler, "127.0.0.1", 0) as ws_server:
            port = ws_server.sockets[0].getsockname()[1]
            return await run_set(RunnerConfig(url=f"ws://127.0.0.1:{port}"), (conversation,), reps=1)

    [row] = asyncio.run(scenario())
    kinds = [turn["kind"] for turn in row["turns"]]
    assert kinds == ["user", "user", "late_result", "user"]
    outputs = [e["item"] for e in server.received if e.get("item", {}).get("type") == "function_call_output"]
    assert [(o["call_id"], json.loads(o["output"])) for o in outputs] == [("call_r", RESULTS["research_report"])]
    s = summarize([row])
    assert (s["right_tool"], s["late_results"], s["late_result_spoken"]) == (1, 1, 1)


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
