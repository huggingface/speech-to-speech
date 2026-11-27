"""Quiz game state, scoring rules and the coach's per-turn decisions.

The browser forwards each final user transcript here. A fast LLM classifies
what the student meant (an answer, a hint request, "skip", ...) and grades
answers against the pack's reference answer and key points. Everything that
affects the score (points, streaks, multipliers, badges) is decided here with
plain rules, so the leaderboard never depends on the model's arithmetic.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import random
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from . import llm

logger = logging.getLogger(__name__)

BASE_POINTS = {1: 100, 2: 150, 3: 200}
PARTIAL_FRACTION = 0.5
# Points for a correct answer on the first, second and third try. The coach
# never reveals the answer before the last try unless the student asks.
ATTEMPT_FACTORS = (1.0, 0.6, 0.35)
MAX_ATTEMPTS = len(ATTEMPT_FACTORS)
HINT_FACTOR = 0.8
STREAK_STEP = 0.1
STREAK_CAP = 1.5
PERFECT_BONUS = 250
MAX_TURNS = 120
INTENTS = {"answer", "stuck", "hint", "repeat", "skip", "clarify", "chitchat", "stop"}
VERDICTS = {"correct", "partial", "incorrect"}

GRADER_SYSTEM = """You are {coach}, a warm, encouraging quiz coach in a spoken conversation with a student. Your job is to help the student find each answer themselves, not to tell them. You get the current question, its reference answer and key points, the attempt number, what the student said earlier on this question, and what they just said. Their words come from speech recognition: expect misheard words and garbled technical terms. Judge the meaning, never the wording.

1. Decide the intent of what the student just said:
- "answer": any attempt to answer, even wrong, partial or hedged ("I think it's...")
- "stuck": they don't know or aren't sure, without asking to move on ("no idea", "hmm, I'm not sure")
- "hint": asks for a hint or for help
- "skip": explicitly wants to move on or hear the answer ("skip", "just tell me", "next question")
- "repeat": asks to repeat or rephrase the question, or didn't hear it
- "clarify": asks what the question means or about its scope
- "stop": wants to end the quiz
- "chitchat": anything else (small talk, filler, noise)
If they give an answer and also say something else, the intent is "answer".

2. If the intent is "answer", grade everything the student has said about this question so far (earlier_attempts plus student_said):
- "correct": gets across what the question asks, the way a fair teacher hearing a spoken answer would accept it, even if it is short. The key points in required_point_indices describe a full answer. When the question asks for several things (a year and a title, two components, three steps), each one must be there. When it asks about one idea, getting its core across is enough, without every detail or nuance of the key points. The other key points and extra details in the reference answer are context: never require them, never ask for them as a follow-up.
- "partial": right direction but misses something the question asks for
- "incorrect": wrong or off target
Their own words count, however informal and without the technical term or the component's exact name ("the error signal fades as it goes back through the layers" covers "vanishing gradients"); never ask them for a term. Saying what something is for implies the gap it fills ("we add X so the model knows Y" covers "without X the model can't know Y"), and a why question is answered by the reason in plain terms, without also explaining the mechanism behind it. A vague answer that could fit many questions ("to make it work better") is not enough. Key points that restate each other or follow from one another count as one. A key point that lists alternatives joined by "or" is covered by any one of them.
Also give score 0-100 and points_hit, the indices of key_points they have covered.

3. Write "say", your next spoken words: at most two sentences and 40 words, plain speech for text-to-speech (no markdown, lists, emojis, parentheses, slashes or symbols). Never read the question itself; the app does that.
- correct: praise something specific they said; optionally add the fun fact in a few words.
- partial or incorrect with attempts_left above 0: in a few words, acknowledge anything right, then give ONE clue or guiding question that leads toward a missing key point.
- stuck with attempts_left above 0: reassure them, then give a clue or an easier question that gets them started.
- A clue must leave the answer for the student to say: never say the answer, the missing key point, or its key term. With attempts_left 2, point in the right direction. With attempts_left 1, be very direct, for example a sentence with a blank where the key term goes ("at every step the model guesses the blank word"), but still never say the key term. When the question has several parts (a year and a title, two components), never say any part the student has not said yet; give clues for each part instead.
- partial or incorrect with attempts_left 0: be kind, then explain the answer simply, linking it to what they said.
- stuck with attempts_left 0, or skip: give the answer in one encouraging sentence.
- hint: a clue that does not give the answer away (used when the prepared hint was already given).
- repeat: an empty string.
- clarify: explain what is being asked without giving the answer away.
- chitchat: reply in a few words and invite their answer.
- stop: a short friendly goodbye.
Each clue must build on the clues already given and be more specific than them.

Return ONLY a JSON object: {"intent": "...", "verdict": "correct|partial|incorrect or null", "score": number or null, "points_hit": [indices], "say": "..."}"""

REPORT_SYSTEM = """You are {coach}, a supportive quiz coach. Given a student's quiz results, write a short coaching report. Be specific to the questions, honest and encouraging.
Return ONLY JSON:
{"headline": "max 8 words", "strengths": ["one or two things they clearly know"], "review": ["up to three concrete concepts to review, phrased as study tips"], "say": "two or three spoken sentences wrapping up: mention the score naturally, one strength, one thing to review. It is read aloud by text-to-speech: no lists, markdown, slashes or symbols."}"""

# Said after the verdict (and fun fact), right before the next question.
TRANSITIONS = [
    "Next question.",
    "Here's the next one.",
    "Moving on.",
    "Okay, next one.",
    "Let's keep going.",
    "Alright, next up.",
    "On to the next one.",
    "Now, try this one.",
    "Here's another one.",
    "Okay, let's move on.",
    "Alright, moving on.",
    "Let's see how you do with this one.",
]
LAST_TRANSITIONS = ["Last question!", "And now, the final question.", "Here's the last one.", "Okay, last one."]
STREAK_CALLOUTS = {
    3: ["That's three in a row!", "Three in a row, you're on fire!"],
    5: ["Five in a row, unstoppable!", "Five straight, incredible!"],
    7: ["Seven in a row. Legendary!"],
}
HINT_REPEAT = "That was my only hint. Give it your best shot!"
# Said when the student's tries together cover every required key point but the
# grader still called the latest one partial.
COVERED_PRAISE = [
    "That's it! Put together, that's the whole answer.",
    "Yes, there it is! Together that covers it.",
    "That's it, you've got all of it now.",
]
GENERIC_CLUE = "Not quite yet. Take another look at what the question is really asking, and give it one more try."

_STOPWORDS = set(
    """a an the and or but of to in on at by for from with without into onto over under about as is are was were be been
    being it its this that these those they them their there here what which who whom whose when where why how do does did
    can could would should will shall may might must not no so than then too very just also only each every any all some one
    you your we our us he she his her him i me my mine yours itself themselves that's it's""".split()
)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("-", " "))


def _content(text: str) -> list[str]:
    return [w for w in _words(text) if w not in _STOPWORDS]


def leaked_terms(clue: str, q: dict[str, Any], said: list[str]) -> list[str]:
    """Answer phrases a clue gives away that the student has not said yet.

    Secrets are pairs of adjacent content words from the reference answer and
    key points (``next word``, ``attention need``) plus any number (a year),
    leaving out what the question itself or the student already said.
    """
    known = set(_content(q["question"])) | {w for text in said for w in _content(text)}
    secrets: set[tuple[str, ...]] = set()
    for source in [q["answer"], *q["key_points"]]:
        words = _content(source)
        for w in words:
            if w.isdigit() and w not in known:
                secrets.add((w,))
        for a, b in zip(words, words[1:]):
            if a != b and not (a in known and b in known):
                secrets.add((a, b))
        # Three-word phrases keep titles made of short words ("all you need").
        raw = _words(source)
        for gram in zip(raw, raw[1:], raw[2:]):
            if any(w not in _STOPWORDS and w not in known for w in gram):
                secrets.add(gram)
    clue_words = _content(clue)
    raw_clue = _words(clue)
    grams = set(zip(clue_words, clue_words[1:])) | {(w,) for w in clue_words}
    grams |= set(zip(raw_clue, raw_clue[1:], raw_clue[2:]))
    return [" ".join(g) for g in secrets if g in grams]


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
    clues: list[str] = field(default_factory=list)
    feedback: str = ""


@dataclass
class Game:
    id: str
    pack: dict[str, Any]
    player: str
    verified: bool
    practice: bool
    question_ids: list[int]
    coach: str = "Max"
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
    # Grades started from speculative transcripts, keyed by question state and text.
    speculative: dict[tuple[int, int, int, str], asyncio.Future[Any]] = field(default_factory=dict)
    # State before the last graded turn that kept the question open (see retract).
    undo: dict[str, Any] | None = None
    last_transition: str = ""

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
            "max_attempts": MAX_ATTEMPTS,
            "attempt_factor": ATTEMPT_FACTORS[min(self.attempt, MAX_ATTEMPTS) - 1],
            "hint_used": bool(self.position < self.total and self.result().hint_used),
            "finished": self.finished,
            "max_score": self.max_score(),
            "results": [
                {"qid": r.qid, "verdict": r.verdict, "points": r.points}
                for r in (self.results.get(qid) for qid in self.question_ids)
                if r is not None
            ],
        }

    # ── Script pieces ────────────────────────────────────────────────────

    def transition(self) -> str:
        """A line leading into the current question, never the same twice in a row."""
        q = self.public_question()
        pool = LAST_TRANSITIONS if q and q["index"] == q["total"] - 1 else TRANSITIONS
        self.last_transition = random.choice([t for t in pool if t != self.last_transition] or pool)
        return self.last_transition

    def ask_current(self, lead: str | None = None) -> str:
        q = self.public_question()
        assert q is not None
        return f"{self.transition() if lead is None else lead} {q['question']}"

    def opening(self) -> str:
        # Short on purpose: the topic summary is on screen, and every spoken
        # second here is a second before the first question.
        # Whole sentences: TTS voices mangle fragments like "Stuck?".
        if self.practice:
            intro = "Let's go over the ones you missed."
        else:
            intro = f"Okay, let's dive into {self.pack.get('title', 'this topic')}."
        first = self.ask_current("Here's the first one.")
        return f"{intro} {first}".strip()


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


def required_points(q: dict[str, Any]) -> list[int]:
    """Indices of the key points a correct answer must contain (all of them when unmarked)."""
    return list(q.get("required") or range(len(q["key_points"])))


def _grader_messages(game: Game, transcript: str) -> list[dict[str, str]]:
    q = game.question()
    res = game.result()
    payload = {
        "topic": game.pack.get("title"),
        "question": q["question"],
        "reference_answer": q["answer"],
        "key_points": q["key_points"],
        "required_point_indices": required_points(q),
        "prepared_hint": q.get("hint", ""),
        "fun_fact": q.get("fun_fact", ""),
        "attempt": game.attempt,
        "attempts_left": MAX_ATTEMPTS - game.attempt,
        "earlier_attempts": res.answers,
        "clues_given": res.clues,
        "recent_conversation": game.history[-6:],
        "student_said": transcript,
    }
    return [
        {"role": "system", "content": GRADER_SYSTEM.replace("{coach}", game.coach)},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


async def _rewrite_clue(game: Game, transcript: str, decision: dict[str, Any], leaks: list[str]) -> str | None:
    messages = _grader_messages(game, transcript) + [
        {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)},
        {
            "role": "user",
            "content": (
                f"Your say gives the answer away (it says: {', '.join(leaks)}). Rewrite say as a clue that leaves "
                'the answer for the student to find, without those words. Return ONLY JSON: {"say": "..."}'
            ),
        },
    ]
    try:
        raw, _ = await llm.chat_json(messages, llm.GRADER_ROUTES, max_tokens=600, temperature=0.4, timeout=8)
    except llm.LLMError:
        return None
    return " ".join(str(raw.get("say") or "").split()) or None


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
    factor = ATTEMPT_FACTORS[min(game.attempt, MAX_ATTEMPTS) - 1]
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
        "required": required_points(q),
        "points_hit": res.points_hit,
        "verdict": res.verdict,
        "points": res.points,
    }


BUTTON_INTENTS = {"hint", "repeat", "skip", "stop"}
MAX_PREFETCHES = 6


def _normalize(text: str) -> str:
    return " ".join(_words(text))


def _speculation_key(game: Game, transcript: str) -> tuple[int, int, int, str]:
    return (game.position, game.attempt, len(game.history), _normalize(transcript))


async def _decide(game: Game, transcript: str) -> tuple[dict[str, Any], str] | None:
    """Grade one utterance against the current question; no game state changes."""
    q = game.question()
    res = game.result()
    try:
        raw, model = await llm.chat_json(
            _grader_messages(game, transcript), llm.GRADER_ROUTES, max_tokens=900, temperature=0.3, timeout=12
        )
    except llm.LLMError as exc:
        logger.error("grader unavailable: %s", exc)
        return None
    decision = _clean_decision(raw, len(q["key_points"]))
    said = [*res.answers, transcript]
    gives_clue = game.attempt < MAX_ATTEMPTS and (
        (decision["intent"] == "answer" and decision["verdict"] != "correct") or decision["intent"] == "stuck"
    )
    if (gives_clue or decision["intent"] == "hint") and (leaks := leaked_terms(decision["say"], q, said)):
        # Models sometimes hand over the answer inside a "clue": ask once
        # for a rewrite; coach_again falls back to the hint if it still leaks.
        fixed = await _rewrite_clue(game, transcript, decision, leaks)
        if fixed and not leaked_terms(fixed, q, said):
            decision["say"] = fixed
    return decision, model


def prefetch(game: Game, transcript: str) -> bool:
    """Start grading a speculative transcript while the backend's turn is still open.

    The speech backend shares each pause's transcript before it commits the
    turn (reopen grace). Grading it now hides the grader's latency inside that
    wait; the authoritative transcript reuses the result if it matches.
    """
    transcript = " ".join(transcript.split())[:600]
    if game.finished or not _normalize(transcript):
        return False
    key = _speculation_key(game, transcript)
    current = {k: t for k, t in game.speculative.items() if k[:3] == key[:3]}
    if key in current or len(current) >= MAX_PREFETCHES:
        game.speculative = current
        return False
    current[key] = asyncio.ensure_future(_decide(game, transcript))
    game.speculative = current
    return True


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
    snapshot = {
        "position": game.position,
        "attempt": game.attempt,
        "history": len(game.history),
        "turns": game.turns - 1,
        "result": dataclasses.replace(
            res, answers=list(res.answers), clues=list(res.clues), points_hit=list(res.points_hit)
        ),
    }
    game.undo = None
    if intent in BUTTON_INTENTS:
        model = "button"
        decision = {"intent": intent, "verdict": None, "score": 0, "points_hit": [], "say": ""}
    else:
        # A grade prefetched from the speculative transcript is reused when the
        # authoritative transcript matches and the question state is unchanged.
        task = game.speculative.pop(_speculation_key(game, transcript), None)
        game.speculative.clear()
        decided = await task if task is not None else await _decide(game, transcript)
        if decided is None:
            return {
                "intent": "error",
                "say": "Sorry, I lost my train of thought for a second. Could you say that again?",
                "state": game.state(),
                "question": game.public_question(),
            }
        decision, model = decided
        if task is not None:
            model = f"{model} (prefetched)"
    intent, verdict, say = decision["intent"], decision["verdict"], decision["say"]
    game.history.append({"role": "student", "text": transcript})

    out: dict[str, Any] = {"intent": intent, "verdict": None, "points": 0, "model": model}
    resolved = False

    def coach_again(fallback: str) -> None:
        """Keep the question open: one more try, with a clue instead of the answer."""
        nonlocal say
        game.attempt += 1
        say = say or fallback
        if leaked_terms(say, q, [*res.answers, transcript]):
            say = fallback if fallback not in " ".join(res.clues) else GENERIC_CLUE
        res.clues.append(say)
        out.update({"retry": True, "clue": say, "attempt": game.attempt, "attempts_left": MAX_ATTEMPTS - game.attempt})

    if intent == "answer":
        res.answers.append(transcript)
        res.attempts += 1
        res.score = decision["score"]
        res.points_hit = sorted(set(res.points_hit) | set(decision["points_hit"]))
        required = set(required_points(q))
        if verdict != "correct" and required <= set(res.points_hit):
            # Across their tries they have said everything a correct answer needs:
            # the ticks on screen and the verdict must agree.
            verdict = "correct"
            fact = q.get("fun_fact", "")
            say = random.choice(COVERED_PRAISE) + (f" Fun fact: {fact}" if fact else "")
        if verdict != "correct" and game.attempt < MAX_ATTEMPTS:
            out["verdict"] = verdict
            coach_again(q.get("hint") or "Not quite yet. Think about it from another angle.")
        else:
            points, multiplier = _award(game, res, verdict or "incorrect")
            if res.verdict == "correct":
                # A correct answer covers every required point, whatever the grader ticked.
                res.points_hit = sorted(set(res.points_hit) | required)
            out.update({"verdict": res.verdict, "points": points, "multiplier": multiplier})
            resolved = True
            if res.verdict == "correct" and game.streak in STREAK_CALLOUTS:
                out["callout"] = random.choice(STREAK_CALLOUTS[game.streak])
                say = f"{say} {out['callout']}".strip()
    elif intent == "hint" and not res.hint_used and q.get("hint"):
        # The prepared hint first: it is written to nudge without giving away.
        res.hint_used = True
        out["hint"] = q["hint"]
        say = f"Here's a hint: {q['hint']}"
        res.clues.append(say)
    elif intent == "hint":
        # More help after the prepared hint: a fresh clue from the coach, if any.
        if say and leaked_terms(say, q, [*res.answers, transcript]):
            say = ""
        if say:
            res.clues.append(say)
            out["clue"] = say
        else:
            say = HINT_REPEAT
    elif intent == "stuck":
        if game.attempt < MAX_ATTEMPTS:
            out["verdict"] = "stuck"
            coach_again(q.get("hint") or "No problem. Think about what the question is really asking.")
        else:
            game.streak = 0
            res.verdict = "skipped"
            out["verdict"] = "skipped"
            resolved = True
            say = say or f"No worries. The answer is: {q['answer']}"
    elif intent == "repeat":
        say = f"Sure. {q['question']}"
    elif intent == "skip":
        game.streak = 0
        res.verdict = "skipped"
        resolved = True
        out["verdict"] = "skipped"
        if not say:
            say = f"No worries. The answer is: {q['answer']}"
    elif intent == "stop":
        return finish_now(game, say or "No problem, let's stop here.")
    elif not say:
        say = "Go on, I'm listening."

    if resolved:
        res.feedback = say
        out["feedback"] = say
        out["reveal"] = _reveal(game, res)
        game.position += 1
        game.attempt = 1
        if game.position >= game.total:
            game.history.append({"role": "coach", "text": say})
            out.update(_finish(game))
            out["say"] = say
            out["state"] = game.state()
            return out
        out["transition"] = game.transition()
        say = f"{say} {game.ask_current(out['transition'])}"

    game.history.append({"role": "coach", "text": say})
    out.update({"say": say.strip(), "state": game.state(), "question": game.public_question()})
    if out.get("retry"):
        game.undo = snapshot
    return out


def retract(game: Game) -> bool:
    """Take back the last grade that gave a clue: the student was still talking.

    Speech recognition ends a turn at a long pause, so the first half of an
    answer can be graded on its own. When the student goes on before hearing
    the result, the try is not counted and the halves are graded together.
    """
    snap = game.undo
    game.undo = None
    if snap is None or game.finished or snap["position"] != game.position:
        return False
    game.attempt = snap["attempt"]
    game.turns = snap["turns"]
    game.results[snap["result"].qid] = snap["result"]
    del game.history[snap["history"] :]
    game.speculative.clear()
    return True


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
                "required": required_points(game.pack["questions"][r.qid]),
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
                {"role": "system", "content": REPORT_SYSTEM.replace("{coach}", game.coach)},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            llm.GRADER_ROUTES,
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
