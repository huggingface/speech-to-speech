"""Quiz game state, scoring rules and the coach's per-turn decisions.

The browser forwards each final user transcript here. A fast LLM classifies
what the student meant (an answer, a hint request, "skip", ...) and grades
answers against the pack's reference answer and key points. Everything that
affects the score (points, streaks, multipliers, badges) is decided here with
plain rules, so the leaderboard never depends on the model's arithmetic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from . import llm

logger = logging.getLogger(__name__)

BASE_POINTS = {1: 100, 2: 150, 3: 200}
PARTIAL_FRACTION = 0.5
RETRY_FACTOR = 0.75
HINT_FACTOR = 0.7
STREAK_STEP = 0.1
STREAK_CAP = 1.5
PERFECT_BONUS = 250
MAX_TURNS = 120
INTENTS = {"answer", "hint", "repeat", "skip", "clarify", "chitchat", "stop"}
VERDICTS = {"correct", "partial", "incorrect"}

GRADER_SYSTEM = """You are Viva, a warm, upbeat quiz coach having a spoken conversation with a student. You get the current question, its reference answer and key points, and what the student just said. The student's words come from speech recognition: expect misheard words, missing punctuation and garbled technical terms. Judge the meaning, never the wording or spelling.

1. Decide the intent of what the student said:
- "answer": any attempt to answer, even wrong, partial or hedged ("I think it's...")
- "hint": asks for a hint or help
- "repeat": asks you to repeat or rephrase the question, or did not hear it
- "skip": does not know, gives up, or asks to skip or reveal the answer
- "clarify": asks what the question means or about its scope
- "stop": wants to end the quiz now
- "chitchat": anything else (small talk, noise, filler like "um", unrelated words)
If the student says an answer AND something else (for example an answer followed by "can I get a hint?"), the intent is "answer": grade what they said.
Grade student_said. On attempt 2, grade it together with previous_answer_this_question (the student is completing their first try). Other earlier messages in the conversation were already graded.

2. If intent is "answer", grade it:
- "correct": conveys the essential idea of the reference answer; small omissions are fine
- "partial": right direction but misses an essential key point, or too vague to be sure
- "incorrect": wrong, or does not address the question
Also give score 0-100 and points_hit, the indices of key_points the answer covered.

3. Write "say": your next spoken words, one or two short sentences, at most 40 words. It is read aloud by text-to-speech: plain words only, no markdown, lists, emojis, parentheses, slashes or symbols; spell out abbreviations that sound awkward. Never repeat the question itself; the app reads questions.
- correct: praise specifically and vary it; you may add the fun fact in a few words.
- partial and retry_available is true: say what was right, then ask one short nudge question toward a missing key point WITHOUT revealing it.
- partial without a retry, or incorrect: be kind and brief, then give the correct answer in one sentence.
- skip: be encouraging, then give the answer in one sentence.
- hint: write an empty string; the app reads the hint.
- repeat: write an empty string.
- clarify: clarify the question without giving away the answer.
- chitchat: reply in a few words and invite them to answer.
- stop: a short friendly goodbye.

Return ONLY a JSON object: {"intent": "...", "verdict": "correct|partial|incorrect or null", "score": number or null, "points_hit": [indices], "say": "..."}"""

REPORT_SYSTEM = """You are Viva, a supportive quiz coach. Given a student's quiz results, write a short coaching report. Be specific to the questions, honest and encouraging.
Return ONLY JSON:
{"headline": "max 8 words", "strengths": ["one or two things they clearly know"], "review": ["up to three concrete concepts to review, phrased as study tips"], "say": "two or three spoken sentences wrapping up: mention the score naturally, one strength, one thing to review. It is read aloud by text-to-speech: no lists, markdown, slashes or symbols."}"""

TRANSITIONS = ["Next question.", "Here's the next one.", "Moving on.", "Okay, next.", "Let's keep going."]
STREAK_CALLOUTS = {
    3: ["That's three in a row!", "Three in a row, you're on fire!"],
    5: ["Five in a row, unstoppable!", "Five straight, incredible!"],
    7: ["Seven in a row. Legendary!"],
}
HINT_REPEAT = "That was my only hint. Give it your best shot!"


def rank_for(accuracy: float) -> dict[str, str]:
    for threshold, title, icon in (
        (0.95, "Grandmaster", "🏆"),
        (0.8, "Expert", "🥇"),
        (0.6, "Scholar", "🥈"),
        (0.4, "Apprentice", "🥉"),
        (0.0, "Rookie", "🌱"),
    ):
        if accuracy >= threshold:
            return {"title": title, "icon": icon}
    return {"title": "Rookie", "icon": "🌱"}


def streak_multiplier(streak: int) -> float:
    """Multiplier for an answer that extends the streak to ``streak``."""
    return min(STREAK_CAP, 1.0 + STREAK_STEP * max(0, streak - 1))


@dataclass
class QuestionResult:
    qid: int
    verdict: str = "pending"  # correct | partial | incorrect | skipped | pending
    score: int = 0
    points: int = 0
    attempts: int = 0
    hint_used: bool = False
    points_hit: list[int] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    feedback: str = ""


@dataclass
class Game:
    id: str
    pack: dict[str, Any]
    player: str
    verified: bool
    practice: bool
    question_ids: list[int]
    created: float = field(default_factory=time.time)
    position: int = 0
    attempt: int = 1
    streak: int = 0
    best_streak: int = 0
    score: int = 0
    turns: int = 0
    finished: bool = False
    finished_at: float | None = None
    results: dict[int, QuestionResult] = field(default_factory=dict)
    history: list[dict[str, str]] = field(default_factory=list)
    summary: dict[str, Any] | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # ── Views ────────────────────────────────────────────────────────────

    @property
    def total(self) -> int:
        return len(self.question_ids)

    def question(self, position: int | None = None) -> dict[str, Any]:
        qid = self.question_ids[self.position if position is None else position]
        return self.pack["questions"][qid]

    def result(self) -> QuestionResult:
        q = self.question()
        return self.results.setdefault(q["id"], QuestionResult(qid=q["id"]))

    def public_question(self, position: int | None = None) -> dict[str, Any] | None:
        position = self.position if position is None else position
        if position >= self.total:
            return None
        q = self.question(position)
        return {
            "index": position,
            "total": self.total,
            "id": q["id"],
            "question": q["question"],
            "difficulty": q["difficulty"],
            "points": BASE_POINTS[q["difficulty"]],
            "has_hint": bool(q.get("hint")),
        }

    def max_score(self) -> int:
        top = 0
        for i, qid in enumerate(self.question_ids):
            q = self.pack["questions"][qid]
            top += round(BASE_POINTS[q["difficulty"]] * streak_multiplier(i + 1))
        return top + PERFECT_BONUS

    def state(self) -> dict[str, Any]:
        return {
            "game_id": self.id,
            "score": self.score,
            "streak": self.streak,
            "best_streak": self.best_streak,
            "multiplier": streak_multiplier(self.streak + 1),
            "position": self.position,
            "total": self.total,
            "attempt": self.attempt,
            "finished": self.finished,
            "max_score": self.max_score(),
            "results": [
                {"qid": r.qid, "verdict": r.verdict, "points": r.points}
                for r in (self.results.get(qid) for qid in self.question_ids)
                if r is not None
            ],
        }

    # ── Script pieces ────────────────────────────────────────────────────

    def ask_current(self, lead: str | None = None) -> str:
        q = self.public_question()
        assert q is not None
        if lead is None:
            lead = "Last question!" if q["index"] == q["total"] - 1 else random.choice(TRANSITIONS)
        return f"{lead} {q['question']}"

    def opening(self) -> str:
        brief = self.pack.get("brief", "").strip()
        n = self.total
        if self.practice:
            intro = f"Let's drill the {n} question{'s' if n != 1 else ''} you missed."
        else:
            intro = f"{brief} I have {n} questions for you. Ask for a hint anytime, but it costs points."
        return f"{intro} {self.ask_current('First question.')}".strip()


class GameStore:
    def __init__(self, ttl_s: float = 3 * 3600) -> None:
        self._games: dict[str, Game] = {}
        self._ttl = ttl_s

    def create(self, pack: dict[str, Any], player: str, verified: bool, question_ids: list[int] | None = None) -> Game:
        self._sweep()
        ids = question_ids if question_ids else [q["id"] for q in pack["questions"]]
        game = Game(
            id=secrets.token_urlsafe(12),
            pack=pack,
            player=player,
            verified=verified,
            practice=question_ids is not None,
            question_ids=ids,
        )
        self._games[game.id] = game
        return game

    def get(self, game_id: str) -> Game | None:
        return self._games.get(game_id)

    def _sweep(self) -> None:
        cutoff = time.time() - self._ttl
        for gid in [g.id for g in self._games.values() if g.created < cutoff]:
            del self._games[gid]


# ── Turn handling ───────────────────────────────────────────────────────────


def _grader_messages(game: Game, transcript: str) -> list[dict[str, str]]:
    q = game.question()
    res = game.result()
    payload = {
        "topic": game.pack.get("title"),
        "question": q["question"],
        "reference_answer": q["answer"],
        "key_points": q["key_points"],
        "fun_fact": q.get("fun_fact", ""),
        "attempt": game.attempt,
        "retry_available": game.attempt == 1,
        "previous_answer_this_question": res.answers[-1] if res.answers else None,
        "recent_conversation": game.history[-6:],
        "student_said": transcript,
    }
    return [
        {"role": "system", "content": GRADER_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _clean_decision(raw: dict[str, Any], n_points: int) -> dict[str, Any]:
    intent = str(raw.get("intent", "")).strip().lower()
    if intent not in INTENTS:
        intent = "answer" if raw.get("verdict") else "chitchat"
    verdict = str(raw.get("verdict") or "").strip().lower()
    if verdict not in VERDICTS:
        verdict = None
    if intent == "answer" and verdict is None:
        verdict = "incorrect"
    try:
        score = max(0, min(100, int(raw.get("score") or 0)))
    except (TypeError, ValueError):
        score = 0
    hits = []
    for h in raw.get("points_hit") or []:
        try:
            idx = int(h)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n_points and idx not in hits:
            hits.append(idx)
    say = " ".join(str(raw.get("say") or "").split())
    return {"intent": intent, "verdict": verdict, "score": score, "points_hit": sorted(hits), "say": say}


def _award(game: Game, res: QuestionResult, verdict: str) -> tuple[int, float]:
    q = game.question()
    base = BASE_POINTS[q["difficulty"]]
    factor = 1.0
    if game.attempt > 1:
        factor *= RETRY_FACTOR
    if res.hint_used:
        factor *= HINT_FACTOR
    multiplier = 1.0
    if verdict == "correct":
        game.streak += 1
        game.best_streak = max(game.best_streak, game.streak)
        multiplier = streak_multiplier(game.streak)
        points = round(base * factor * multiplier)
    elif verdict == "partial":
        game.streak = 0
        points = round(base * PARTIAL_FRACTION * factor)
    else:
        game.streak = 0
        points = 0
    res.verdict = verdict
    res.points = points
    game.score += points
    return points, multiplier


def _reveal(game: Game, res: QuestionResult) -> dict[str, Any]:
    q = game.question()
    return {
        "qid": q["id"],
        "answer": q["answer"],
        "key_points": q["key_points"],
        "points_hit": res.points_hit,
        "verdict": res.verdict,
        "points": res.points,
    }


BUTTON_INTENTS = {"hint", "repeat", "skip", "stop"}


async def handle_turn(game: Game, transcript: str, intent: str | None = None) -> dict[str, Any]:
    """Process one student utterance (or a button press) and return the coach's next move."""
    transcript = " ".join(transcript.split())[:600]
    if game.finished:
        return {"intent": "stop", "say": "", "state": game.state(), "finished": True}
    game.turns += 1
    if game.turns > MAX_TURNS:
        return finish_now(game, "We've been at this a while, so let's wrap up here.")

    q = game.question()
    res = game.result()
    if intent in BUTTON_INTENTS:
        model = "button"
        decision = {"intent": intent, "verdict": None, "score": 0, "points_hit": [], "say": ""}
    else:
        try:
            raw, model = await llm.chat_json(
                _grader_messages(game, transcript), llm.GRADER_MODELS, max_tokens=900, temperature=0.3, timeout=12
            )
        except llm.LLMError as exc:
            logger.error("grader unavailable: %s", exc)
            return {
                "intent": "error",
                "say": "Sorry, I lost my train of thought for a second. Could you say that again?",
                "state": game.state(),
                "question": game.public_question(),
            }
        decision = _clean_decision(raw, len(q["key_points"]))
    intent, verdict, say = decision["intent"], decision["verdict"], decision["say"]
    game.history.append({"role": "student", "text": transcript})

    out: dict[str, Any] = {"intent": intent, "verdict": None, "points": 0, "model": model}
    resolved = False

    if intent == "answer":
        res.answers.append(transcript)
        res.attempts += 1
        res.score = decision["score"]
        res.points_hit = sorted(set(res.points_hit) | set(decision["points_hit"]))
        if verdict == "partial" and game.attempt == 1:
            game.attempt = 2
            out["verdict"] = "partial"
            out["retry"] = True
            if not say:
                say = "You're on the right track. Can you add a bit more?"
        else:
            points, multiplier = _award(game, res, verdict or "incorrect")
            out.update({"verdict": res.verdict, "points": points, "multiplier": multiplier})
            resolved = True
            if res.verdict == "correct" and game.streak in STREAK_CALLOUTS:
                out["callout"] = random.choice(STREAK_CALLOUTS[game.streak])
                say = f"{say} {out['callout']}".strip()
    elif intent == "hint":
        if res.hint_used or not q.get("hint"):
            say = HINT_REPEAT
        else:
            res.hint_used = True
            out["hint"] = q["hint"]
            say = f"Here's a hint: {q['hint']}"
    elif intent == "repeat":
        say = f"Sure. {q['question']}"
    elif intent == "skip":
        res.answers.append(transcript)
        game.streak = 0
        res.verdict = "skipped"
        resolved = True
        out["verdict"] = "skipped"
        if not say:
            say = f"No worries. The answer is: {q['answer']}"
    elif intent == "stop":
        return finish_now(game, say or "No problem, let's stop here.")
    elif not say:
        say = "Take your time. What's your answer?"

    if resolved:
        res.feedback = say
        out["reveal"] = _reveal(game, res)
        game.position += 1
        game.attempt = 1
        if game.position >= game.total:
            game.history.append({"role": "coach", "text": say})
            out.update(_finish(game))
            out["say"] = say
            out["state"] = game.state()
            return out
        say = f"{say} {game.ask_current()}"

    game.history.append({"role": "coach", "text": say})
    out.update({"say": say.strip(), "state": game.state(), "question": game.public_question()})
    return out


def finish_now(game: Game, say: str) -> dict[str, Any]:
    out = {"intent": "stop", "say": say, "verdict": None, "points": 0}
    out.update(_finish(game))
    out["state"] = game.state()
    return out


def _finish(game: Game) -> dict[str, Any]:
    game.finished = True
    game.finished_at = time.time()
    results = [game.results.get(qid) or QuestionResult(qid=qid) for qid in game.question_ids]
    answered = [r for r in results if r.verdict != "pending"]
    correct = sum(1 for r in results if r.verdict == "correct")
    partial = sum(1 for r in results if r.verdict == "partial")
    bonus = 0
    if game.total >= 3 and correct == game.total:
        bonus = PERFECT_BONUS
        game.score += bonus
    accuracy = (correct + 0.5 * partial) / game.total if game.total else 0.0
    badges = []
    if bonus:
        badges.append({"id": "flawless", "icon": "💎", "label": "Flawless"})
    if game.best_streak >= 5:
        badges.append({"id": "unstoppable", "icon": "⚡", "label": "Unstoppable"})
    elif game.best_streak >= 3:
        badges.append({"id": "on-fire", "icon": "🔥", "label": "On fire"})
    if not any(r.hint_used for r in results) and accuracy >= 0.6 and len(answered) == game.total:
        badges.append({"id": "no-hints", "icon": "🧠", "label": "No hints needed"})
    if any(r.verdict == "correct" and r.attempts > 1 for r in results):
        badges.append({"id": "comeback", "icon": "💪", "label": "Comeback"})
    hard = [r for r in results if game.pack["questions"][r.qid]["difficulty"] == 3]
    if hard and all(r.verdict == "correct" for r in hard):
        badges.append({"id": "hard-hitter", "icon": "🎯", "label": "Hard hitter"})
    game.summary = {
        "score": game.score,
        "max_score": game.max_score(),
        "correct": correct,
        "partial": partial,
        "total": game.total,
        "accuracy": round(accuracy, 3),
        "best_streak": game.best_streak,
        "perfect_bonus": bonus,
        "duration_s": round((game.finished_at or time.time()) - game.created),
        "rank": rank_for(accuracy),
        "badges": badges,
        "missed": [r.qid for r in results if r.verdict in {"incorrect", "skipped", "partial", "pending"}],
        "review": [
            {
                "qid": r.qid,
                "question": game.pack["questions"][r.qid]["question"],
                "answer": game.pack["questions"][r.qid]["answer"],
                "key_points": game.pack["questions"][r.qid]["key_points"],
                "points_hit": r.points_hit,
                "verdict": r.verdict,
                "points": r.points,
                "your_answer": r.answers[-1] if r.answers else "",
                "hint_used": r.hint_used,
            }
            for r in results
        ],
    }
    return {"finished": True, "summary": game.summary}


async def coaching_report(game: Game) -> dict[str, Any]:
    """LLM-written wrap-up; falls back to a template when the model is down."""
    summary = game.summary or {}
    payload = {
        "topic": game.pack.get("title"),
        "score": summary.get("score"),
        "correct": summary.get("correct"),
        "partial": summary.get("partial"),
        "total": summary.get("total"),
        "rank": summary.get("rank", {}).get("title"),
        "questions": [
            {
                "question": r["question"],
                "verdict": r["verdict"],
                "student_answer": r["your_answer"],
                "key_points_missed": [p for i, p in enumerate(r["key_points"]) if i not in r["points_hit"]],
            }
            for r in summary.get("review", [])
        ],
    }
    try:
        raw, _ = await llm.chat_json(
            [
                {"role": "system", "content": REPORT_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            llm.GRADER_MODELS,
            max_tokens=1200,
            temperature=0.5,
            timeout=15,
        )
        report = {
            "headline": str(raw.get("headline") or "").strip(),
            "strengths": [str(s) for s in raw.get("strengths") or []][:2],
            "review": [str(s) for s in raw.get("review") or []][:3],
            "say": " ".join(str(raw.get("say") or "").split()),
        }
        if report["say"]:
            return report
    except llm.LLMError as exc:
        logger.warning("report fallback: %s", exc)
    rank = summary.get("rank", {}).get("title", "Rookie")
    return {
        "headline": f"{rank} on {game.pack.get('title')}",
        "strengths": [],
        "review": [],
        "say": (
            f"That's a wrap! You scored {summary.get('score', 0)} points with "
            f"{summary.get('correct', 0)} out of {summary.get('total', 0)} correct. "
            "Check the review to see what to brush up on."
        ),
    }
