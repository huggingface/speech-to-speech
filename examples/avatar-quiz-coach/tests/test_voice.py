"""The conversational coach's tool layer: instructions, tool mapping and tool results.

Run from examples/avatar-quiz-coach:  python -m pytest tests
"""

import asyncio
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coach import game as g  # noqa: E402
from coach import voice  # noqa: E402
from test_game import make_pack  # noqa: E402


@pytest.fixture
def game():
    pack = make_pack()
    for q in pack["questions"]:
        q["key_points"] = [f"keypoint alpha {q['id']}", f"keypoint beta {q['id']}"]
    return g.Game(
        id="t",
        pack=pack,
        player="Sam",
        verified=False,
        practice=False,
        question_ids=[q["id"] for q in pack["questions"]],
        coach="Ada",
    )


def test_instructions_carry_state_but_never_answers(game):
    text = voice.instructions(game)
    q = game.question()
    assert q["question"] in text and "(1 of 3)" in text
    assert "Sam" in text and "Ada" in text
    for item in game.pack["questions"]:
        assert item["answer"] not in text
        for point in item["key_points"]:
            assert point not in text
    game.player = "Anonymous"
    assert "Anonymous" not in voice.instructions(game) and "Anonymous" not in voice.opening(game)


def test_opening_is_casual_and_reads_the_first_question(game):
    q = game.question()["question"]
    start = voice.opening(game).split("Start the quiz now.")[1]
    assert q in start and "Sam" in start
    assert str(game.total) not in start.replace(q, "")
    # Back from reading up: no second hello.
    back = voice.opening(game, greeted=True).split("Start the quiz now.")[1]
    assert not any(w in back for w in ("Hey", "Hi ")) and q in back
    # The example line varies from quiz to quiz.
    starts = {voice.opening(game, rng=random.Random(seed)).split("something like")[1][:40] for seed in range(20)}
    assert len(starts) > 2
    game.practice = True
    assert "missed" in voice.opening(game).split("Start the quiz now.")[1]


def test_game_opening_skips_the_count(game):
    assert str(game.total) not in game.opening().replace(game.question()["question"], "")


def test_tools_map_onto_turn_intents():
    names = {t["name"] for t in voice.TOOLS}
    assert names == set(voice.TOOL_INTENTS)
    assert voice.TOOL_INTENTS["check_answer"] is None
    assert {voice.TOOL_INTENTS[n] for n in names if n != "check_answer"} <= g.BUTTON_INTENTS


def run(coro):
    return asyncio.run(coro)


def test_tool_outputs(game, monkeypatch):
    async def wrong(game, transcript):
        return {
            "intent": "answer",
            "verdict": "incorrect",
            "score": 0,
            "points_hit": [],
            "say": "Think about the order.",
        }, "stub"

    monkeypatch.setattr(g, "_decide", wrong)
    retry = json.loads(voice.tool_output(game, run(g.handle_turn(game, "no idea really"))))
    assert retry["verdict"] == "incorrect" and retry["tries_left"] == 2 and retry["say"] == "Think about the order."
    assert "next_question" not in retry

    skipped = json.loads(voice.tool_output(game, run(g.handle_turn(game, "(skip)", "skip"))))
    assert skipped["verdict"] == "skipped" and skipped["next_question"] == game.question()["question"]
    assert skipped["say"].startswith("No worries. The answer is")
    assert game.question()["question"] not in skipped["say"]  # the next question is asked separately

    hint = json.loads(voice.tool_output(game, run(g.handle_turn(game, "(hint)", "hint"))))
    assert hint["hint"] == game.question()["hint"]

    done = json.loads(voice.tool_output(game, run(g.handle_turn(game, "(stop)", "stop"))))
    assert done["quiz_finished"] and "say" not in done  # the coach already acknowledged the stop
    assert "over" in voice.instructions(game)


def test_retract_undoes_a_half_answer(game, monkeypatch):
    verdicts = iter(["partial", "correct"])

    async def decide(game, transcript):
        return {"intent": "answer", "verdict": next(verdicts), "score": 50, "points_hit": [0], "say": "Go on."}, "stub"

    monkeypatch.setattr(g, "_decide", decide)
    history = len(game.history)
    out = run(g.handle_turn(game, "first half"))
    assert out["retry"] and game.attempt == 2
    assert g.retract(game)
    res = game.result()
    assert game.attempt == 1 and res.answers == [] and res.clues == [] and len(game.history) == history
    assert not g.retract(game)  # only once
    out = run(g.handle_turn(game, "first half second half"))
    assert out["verdict"] == "correct" and out["points"] == g.BASE_POINTS[game.pack["questions"][0]["difficulty"]]
    assert not g.retract(game)  # a resolved question can't be taken back
