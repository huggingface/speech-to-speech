"""The coach as a live conversation: the speech backend's LLM talks, the grader judges.

The backend's model holds the conversation (it answers small talk, repeats or
explains a question, and can be interrupted at any time). It never knows the
reference answers: when the student attempts one, it calls a tool, the app
runs the grader (game.handle_turn) and hands back what to say, a clue that
has been checked for leaks, or the verdict and the next question.
"""

from __future__ import annotations

import json
from typing import Any

from .game import MAX_ATTEMPTS, Game

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "check_answer",
        "description": (
            "Grade the student's attempt at the current quiz question. Call it as soon as they attempt an "
            "answer, even a partial, hesitant or probably wrong one, or say they don't know. Returns the "
            "verdict and what to tell them: a clue for another try, or the result and the next question."
        ),
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string", "description": "What the student said, in their words."}},
            "required": ["answer"],
        },
    },
    {
        "type": "function",
        "name": "give_hint",
        "description": "The student asks for a hint or for help with the current question. Costs them a few points.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "skip_question",
        "description": "The student wants to skip the current question or to hear its answer.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "end_quiz",
        "description": "The student wants to stop the quiz now.",
        "parameters": {"type": "object", "properties": {}},
    },
]

# Tool name -> handle_turn intent (None: grade the words).
TOOL_INTENTS = {"check_answer": None, "give_hint": "hint", "skip_question": "skip", "end_quiz": "stop"}

RULES = """You are {coach}, a quiz coach on a live voice call with {player}, quizzing them on "{title}". Talk like a warm, upbeat tutor: short spoken sentences, a few at a time, and let them do most of the talking.

The quiz has {n} questions, asked one at a time in order. A correct answer earns points, more on the first try and with a streak going. There are up to {tries} tries per question, and a wrong try gets a clue. A hint costs a few points.

You do not know the answers. Only the check_answer tool decides whether an answer is right. Never answer a quiz question yourself, never say an answer is right or wrong before check_answer returns, and never give clues or hints of your own.

When to use the tools:
- They attempt an answer to the current question, even a partial, hesitant or probably wrong one, or they say they don't know: call check_answer with what they said, without saying anything first.
- They ask for a hint or for help: call give_hint.
- They want to skip the question or hear the answer: call skip_question.
- They want to stop the quiz: call end_quiz.
Anything else needs no tool. If they ask you to repeat the question or to say that again, or they cut in before you finished asking it, ask the current question again word for word. If they ask what a word means or what kind of answer you want, explain without giving the answer away. Answer small talk in a few words and bring them back to the question. Keep the quiz moving: never ask whether they are ready, just ask the question.

When a tool returns, tell the student its "say" text in your own voice, adding no facts of your own, then do what its "next" says. Read every quiz question exactly as written."""


def _name(game: Game) -> str:
    return game.player if game.player and game.player != "Anonymous" else "the student"


def instructions(game: Game) -> str:
    """Session instructions for the backend's LLM, with the quiz's current state."""
    text = RULES.format(
        coach=game.coach,
        player=_name(game),
        title=game.pack.get("title", "this topic"),
        n=game.total,
        tries=MAX_ATTEMPTS,
    )
    if game.finished:
        s = game.summary or {}
        return (
            text + f"\n\nThe quiz is over: {s.get('correct', 0)} of {s.get('total', game.total)} correct, "
            f"{game.score} points. Chat briefly if they want to, but ask no more quiz questions."
        )
    q = game.question()
    tries = f", try {game.attempt} of {MAX_ATTEMPTS}" if game.attempt > 1 else ""
    streak = f", on a streak of {game.streak}" if game.streak >= 2 else ""
    return (
        text + f'\n\nCurrent question ({game.position + 1} of {game.total}{tries}): "{q["question"]}"\n'
        f"Score so far: {game.score} points{streak}."
    )


def opening(game: Game) -> str:
    """Per-response instructions that start the call."""
    q = game.question()
    lead = (
        f"say you'll drill the {game.total} questions they missed"
        if game.practice
        else f'say you have {game.total} questions on "{game.pack.get("title", "this topic")}" '
        "and that you'll help with clues if they get stuck"
    )
    return (
        instructions(game) + f"\n\nStart the call now: greet {_name(game)} in a few words, "
        f'{lead}, then ask the first question exactly as written: "{q["question"]}"'
    )


def tool_output(game: Game, out: dict[str, Any]) -> str:
    """What the backend's LLM sees after a tool runs (compact JSON)."""
    result: dict[str, Any]
    if out.get("finished"):
        s = out.get("summary") or {}
        result = {
            "quiz_finished": True,
            # A stop request was already acknowledged by the coach itself.
            "say": "" if out.get("intent") == "stop" else out.get("say", ""),
            "correct": s.get("correct"),
            "total": s.get("total"),
            "score": s.get("score"),
            "next": "Then wrap up warmly in one or two sentences with their score. The app shows the full results.",
        }
    elif out.get("retry"):
        result = {
            "verdict": out.get("verdict") or "not yet",
            # attempts_left counts the tries after the next one.
            "tries_left": (out.get("attempts_left") or 0) + 1,
            "say": out.get("clue") or out.get("say", ""),
            "next": "Let them try again. Do not reveal or hint at the answer beyond this clue.",
        }
    elif out.get("reveal"):
        q = game.public_question()
        result = {
            "verdict": out.get("verdict"),
            "points": out.get("points", 0),
            "score": game.score,
            "streak": game.streak,
            "say": out.get("feedback") or out.get("say", ""),
            "next_question": q["question"] if q else None,
            "next": f"Then ask question {q['index'] + 1} of {q['total']} exactly as written." if q else "",
        }
    elif out.get("hint"):
        result = {"hint": out["hint"], "say": out.get("say", ""), "next": "Then let them answer."}
    elif out.get("intent") == "error":
        result = {"error": "The grader is unavailable.", "next": "Ask them to say their answer again."}
    else:
        result = {"say": out.get("say", ""), "next": "Then wait for their answer."}
    return json.dumps({k: v for k, v in result.items() if v not in (None, "")}, ensure_ascii=False)
