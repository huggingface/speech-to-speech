"""Scoring and turn rules, with the grader LLM stubbed out.

Run from examples/avatar-quiz-coach:  python -m pytest tests
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coach import game as g  # noqa: E402
from coach import llm  # noqa: E402
from coach.leaderboard import Leaderboard  # noqa: E402
from coach.llm import extract_json  # noqa: E402
from coach.research import normalize_pack  # noqa: E402


def make_pack(levels=(1, 2, 3)):
    return {
        "id": "p1",
        "title": "Test topic",
        "slug": "test-topic",
        "emoji": "🧪",
        "brief": "A test.",
        "difficulty": "pro",
        "questions": [
            {
                "id": i,
                "question": f"Question {i}?",
                "answer": f"Answer {i}.",
                "key_points": ["a", "b"],
                "hint": f"Hint {i}",
                "fun_fact": "",
                "difficulty": level,
            }
            for i, level in enumerate(levels)
        ],
    }


@pytest.fixture
def grader(monkeypatch):
    """Queue grader replies; each turn pops the next one."""
    replies: list[dict] = []
    calls: list[list] = []

    async def fake_chat_json(messages, models, **kwargs):
        calls.append(messages)
        return replies.pop(0), "stub"

    monkeypatch.setattr(llm, "chat_json", fake_chat_json)
    return replies, calls


def turn(game, text, intent=None):
    return asyncio.run(g.handle_turn(game, text, intent))


def answer(verdict, say="ok", hits=(0,)):
    return {"intent": "answer", "verdict": verdict, "score": 90, "points_hit": list(hits), "say": say}


def test_correct_answers_build_streak_and_multiplier(grader):
    replies, _ = grader
    game = g.GameStore().create(make_pack((2, 2, 2)), "ana", False)
    replies += [answer("correct"), answer("correct"), answer("correct")]
    points = [turn(game, "x")["points"] for _ in range(3)]
    assert points == [150, round(150 * 1.1), round(150 * 1.2)]
    assert game.finished and game.best_streak == 3
    # Perfect round bonus lands in the final score.
    assert game.score == sum(points) + g.PERFECT_BONUS
    badge_ids = {b["id"] for b in game.summary["badges"]}
    assert {"flawless", "on-fire", "no-hints"} <= badge_ids


def test_wrong_answers_get_clues_not_the_answer(grader):
    replies, _ = grader
    game = g.GameStore().create(make_pack((2, 1)), "ana", False)
    replies += [answer("partial", "Good start. What does it predict?")]
    out = turn(game, "half an answer")
    assert out["retry"] and out["points"] == 0 and "reveal" not in out
    assert out["clue"] == "Good start. What does it predict?" and out["attempts_left"] == 1
    assert game.position == 0 and game.attempt == 2
    assert "Question 0?" not in out["say"]
    replies += [answer("correct", "Yes!")]
    out = turn(game, "full answer")
    assert out["points"] == round(150 * g.ATTEMPT_FACTORS[1])
    assert game.position == 1 and game.attempt == 1
    assert out["say"].startswith("Yes!") and "Question 1?" in out["say"]
    assert out["reveal"]["answer"] == "Answer 0."


def test_answer_revealed_only_after_last_try(grader):
    replies, _ = grader
    game = g.GameStore().create(make_pack((2, 1)), "ana", False)
    replies += [answer("incorrect", "Clue one?"), answer("incorrect", "Clue two?"), answer("partial", "It is X.")]
    assert "reveal" not in turn(game, "wrong")
    assert "reveal" not in turn(game, "wrong again")
    out = turn(game, "half right")
    assert out["verdict"] == "partial" and out["reveal"]["answer"] == "Answer 0."
    assert out["points"] == round(150 * g.PARTIAL_FRACTION * g.ATTEMPT_FACTORS[2])
    assert game.position == 1 and game.streak == 0
    assert game.results[0].clues == ["Clue one?", "Clue two?"]


def test_stuck_student_gets_a_clue(grader):
    replies, _ = grader
    game = g.GameStore().create(make_pack((1, 1)), "ana", False)
    replies += [{"intent": "stuck", "say": "No worries. What happens at each step?"}]
    out = turn(game, "I have no idea")
    assert out["verdict"] == "stuck" and out["retry"] and game.position == 0
    assert out["say"] == "No worries. What happens at each step?"


def test_hint_button_skips_llm_and_costs_points(grader):
    replies, calls = grader
    game = g.GameStore().create(make_pack((1, 1)), "ana", False)
    out = turn(game, "", "hint")
    assert not calls and out["say"] == "Here's a hint: Hint 0"
    out = turn(game, "", "hint")
    assert out["say"] == g.HINT_REPEAT  # buttons skip the model, so no fresh clue
    replies += [answer("correct")]
    out = turn(game, "answer")
    assert out["points"] == round(100 * g.HINT_FACTOR)


def test_extra_hint_requests_get_fresh_clues(grader):
    replies, _ = grader
    game = g.GameStore().create(make_pack((1, 1)), "ana", False)
    turn(game, "", "hint")
    replies += [{"intent": "hint", "say": "Think of the last word."}]
    out = turn(game, "another hint please")
    assert out["clue"] == "Think of the last word." and game.attempt == 1


def test_skip_and_repeat(grader):
    _, calls = grader
    game = g.GameStore().create(make_pack((1, 1)), "ana", False)
    out = turn(game, "", "repeat")
    assert out["say"] == "Sure. Question 0?" and game.position == 0
    out = turn(game, "", "skip")
    assert out["verdict"] == "skipped" and game.position == 1
    assert "Answer 0." in out["say"] and "Question 1?" in out["say"]
    assert not calls


def test_stop_finishes_and_lists_missed(grader):
    game = g.GameStore().create(make_pack((1, 2, 3)), "ana", False)
    out = turn(game, "", "stop")
    assert out["finished"] and game.finished
    assert out["summary"]["missed"] == [0, 1, 2]
    assert out["summary"]["rank"]["title"] == "Rookie"


def test_grader_failure_keeps_state(monkeypatch):
    async def boom(*a, **k):
        raise llm.LLMError("down")

    monkeypatch.setattr(llm, "chat_json", boom)
    game = g.GameStore().create(make_pack((1,)), "ana", False)
    out = turn(game, "an answer")
    assert out["intent"] == "error" and game.position == 0 and game.score == 0


def test_clean_decision_sanitises_model_output():
    d = g._clean_decision({"intent": "ANSWER", "verdict": "Correct", "score": "140", "points_hit": [0, 5, "1", 0]}, 2)
    assert d == {"intent": "answer", "verdict": "correct", "score": 100, "points_hit": [0, 1], "say": ""}
    d = g._clean_decision({"intent": "dance"}, 2)
    assert d["intent"] == "chitchat"


def test_practice_games_use_subset():
    game = g.GameStore().create(make_pack((1, 2, 3)), "ana", False, [2, 0])
    assert game.practice and game.total == 2
    assert game.public_question()["id"] == 2


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here you go: {"a": {"b": "}"}} hope it helps') == {"a": {"b": "}"}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_normalize_pack_sorts_and_clamps():
    raw = {
        "title": "T",
        "questions": [
            {"question": "hard?", "answer": "h", "key_points": ["k"] * 6, "difficulty": 9},
            {"question": "easy?", "answer": "e", "difficulty": "1"},
            {"question": "", "answer": "dropped"},
            {"question": "mid?", "answer": "m", "difficulty": 2},
        ],
    }
    pack = normalize_pack(raw, topic="t", difficulty="pro", count=8)
    assert [q["question"] for q in pack["questions"]] == ["easy?", "mid?", "hard?"]
    assert [q["id"] for q in pack["questions"]] == [0, 1, 2]
    assert len(pack["questions"][2]["key_points"]) == 4
    assert pack["questions"][0]["key_points"] == ["e"]


def test_leaderboard_ranks_best_run_per_player(tmp_path):
    board = Leaderboard(str(tmp_path))
    base = {
        "slug": "s",
        "title": "S",
        "emoji": "x",
        "accuracy": 1,
        "correct": 1,
        "total": 1,
        "best_streak": 1,
        "rank": "r",
        "verified": False,
    }
    board.add({**base, "player": "ana", "score": 300})
    board.add({**base, "player": "bob", "score": 500})
    placement = board.add({**base, "player": "ana", "score": 200})
    assert placement["topic_rank"] == 2 and placement["players_on_topic"] == 2
    rows = board.top("s")
    assert [(r["player"], r["score"]) for r in rows] == [("bob", 500), ("ana", 300)]
    # Persisted as JSON lines and reloaded.
    again = Leaderboard(str(tmp_path))
    assert len(again.top()) == 2
    assert all(json.loads(line) for line in (tmp_path / "leaderboard" / "leaderboard.jsonl").read_text().splitlines())


def test_featured_packs_are_well_formed():
    for path in (ROOT / "packs").glob("*.json"):
        pack = json.loads(path.read_text())
        assert pack["id"] and pack["slug"] and pack["title"]
        ids = [q["id"] for q in pack["questions"]]
        assert ids == list(range(len(ids))) and len(ids) >= 5
        levels = [q["difficulty"] for q in pack["questions"]]
        assert levels == sorted(levels)
        for q in pack["questions"]:
            assert q["question"] and q["answer"] and q["key_points"]


def test_leaked_terms_flags_answer_phrases_only():
    q = {
        "question": "What year was the transformer introduced, and in which paper?",
        "answer": "In 2017, in the paper Attention Is All You Need.",
        "key_points": ["2017", "Attention Is All You Need"],
    }
    assert g.leaked_terms("It came out in 2017.", q, []) == ["2017"]
    assert "all you need" in g.leaked_terms("Think of the All You Need paper.", q, [])
    assert g.leaked_terms("The title plays on a Beatles song.", q, []) == []
    # What the student already said is no secret.
    assert g.leaked_terms("Yes, 2017 is right. Now the title?", q, ["was it 2017"]) == []


def test_leaking_clue_is_rewritten_or_replaced(grader):
    replies, calls = grader
    pack = make_pack((1, 1))
    pack["questions"][0]["answer"] = "It predicts the next word."
    game = g.GameStore().create(pack, "ana", False)
    replies += [answer("incorrect", "No, it predicts the next word."), {"say": "Still the next word!"}]
    out = turn(game, "classification?")
    assert len(calls) == 2  # one rewrite attempt
    assert out["retry"] and "next word" not in out["say"]
    assert out["say"] == "Hint 0"  # fell back to the prepared hint


def test_prefetched_grade_is_reused_by_the_matching_turn(grader):
    replies, calls = grader
    game = g.GameStore().create(make_pack((1, 1)), "ana", False)
    replies += [answer("correct", "Yes!")]

    async def flow():
        assert g.prefetch(game, "It predicts the next word.")
        assert not g.prefetch(game, "it predicts the next word")  # same text, already running
        assert game.score == 0  # prefetching never changes the game
        return await g.handle_turn(game, "It predicts the next word")

    out = asyncio.run(flow())
    assert len(calls) == 1 and out["verdict"] == "correct" and out["model"].endswith("(prefetched)")
    assert game.speculative == {}


def test_prefetch_for_other_text_is_ignored(grader):
    replies, calls = grader
    game = g.GameStore().create(make_pack((1, 1)), "ana", False)
    replies += [answer("partial", "Half."), answer("correct", "Yes!")]

    async def flow():
        g.prefetch(game, "It predicts")
        await asyncio.sleep(0)
        return await g.handle_turn(game, "It predicts the next word")

    out = asyncio.run(flow())
    assert len(calls) == 2 and out["verdict"] == "correct"
