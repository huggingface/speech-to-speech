"""Turn a spoken assistant reply into a comparable Big Bench Audio answer.

Everything here is pure text work: no network, no audio, no pipeline. The
realtime plumbing lives in :mod:`speech_to_speech.evals.big_bench_audio.runner`.

Two extraction paths are tried in order:

``marker``
    The reply ends with an explicit answer sentence (``Final answer: No``). The
    default eval instructions ask for exactly this, so it is the usual path, and
    the first candidate *inside* the marker tail wins.
``scan``
    No marker survived transcription. The whole reply is scanned and the **last**
    candidate wins, because these questions need multi-step reasoning and models
    state their conclusion after working through it.

A reply with no candidate at all grades as ``correct=None`` (unparsed) rather
than as a wrong answer, so a run can tell "the model reasoned badly" apart from
"the model never answered in a gradable form".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

AnswerKind = Literal["choice", "integer"]
ExtractionMethod = Literal["marker", "scan"]


@dataclass(frozen=True)
class CategorySpec:
    """How one Big Bench Audio category spells its answers."""

    kind: AnswerKind
    labels: tuple[str, ...] = ()


CATEGORY_SPECS: dict[str, CategorySpec] = {
    "formal_fallacies": CategorySpec("choice", ("valid", "invalid")),
    "navigate": CategorySpec("choice", ("yes", "no")),
    "object_counting": CategorySpec("integer"),
    "web_of_lies": CategorySpec("choice", ("yes", "no")),
}


@dataclass(frozen=True)
class Grade:
    """Result of grading one reply."""

    extracted: Optional[str]
    correct: Optional[bool]
    method: Optional[ExtractionMethod]

    @property
    def parsed(self) -> bool:
        return self.extracted is not None


# "Final answer: X", "the answer is X", "my answer: X", "answer -- X".
_MARKER_RE = re.compile(
    r"\b(?:final\s+answer|the\s+answer\s+is|my\s+answer\s+is|my\s+answer|answer)\b\s*(?:is\s*)?[:\-–—]?\s*",
    re.IGNORECASE,
)

# Deliberately narrow. "correct" and "negative" read as yes/no in conversation but
# also appear as ordinary words in these questions ("the correct answer is No",
# "the negative direction"), and a wrong extraction is worse than an unparsed one.
_YES_RE = re.compile(r"\b(?:yes|yeah|yep|yup)\b", re.IGNORECASE)
_NO_RE = re.compile(r"\b(?:no|nope|nah)\b", re.IGNORECASE)
_VALID_RE = re.compile(r"\b(?:in\s*-?\s*valid|invalid|valid)\b", re.IGNORECASE)
# Matches "not valid" and "not deductively valid", but not "not sure this is valid".
_NEGATION_RE = re.compile(r"\b(?:not|isn't|aren't|never|n't)\b(?:\s+\w+ly)?\s*$", re.IGNORECASE)

_NUMBER_WORDS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_NUMBER_RE = re.compile(r"\b(\d{1,3}|" + "|".join(_NUMBER_WORDS) + r")\b", re.IGNORECASE)

# A marker tail is only the answer sentence; anything past it is fresh rambling.
_MARKER_TAIL_CHARS = 60


def _marker_tail(text: str) -> Optional[str]:
    """Return the text just after the last answer marker, or ``None``."""
    last: Optional[re.Match[str]] = None
    for match in _MARKER_RE.finditer(text):
        last = match
    if last is None:
        return None
    return text[last.end() : last.end() + _MARKER_TAIL_CHARS]


def _negated(text: str, start: int) -> bool:
    """Whether the token at *start* is preceded by a negation within a short window."""
    return bool(_NEGATION_RE.search(text[max(0, start - 24) : start]))


def _extract_yes_no(text: str, *, first: bool) -> Optional[str]:
    matches = [("yes", m) for m in _YES_RE.finditer(text)]
    matches += [("no", m) for m in _NO_RE.finditer(text)]
    if not matches:
        return None
    matches.sort(key=lambda pair: pair[1].start())
    label, match = matches[0] if first else matches[-1]
    if _negated(text, match.start()):
        return "no" if label == "yes" else "yes"
    return label


def _extract_valid(text: str, *, first: bool) -> Optional[str]:
    matches = list(_VALID_RE.finditer(text))
    if not matches:
        return None
    match = matches[0] if first else matches[-1]
    label = "valid" if match.group(0).lower().replace("-", "").replace(" ", "") == "valid" else "invalid"
    if label == "valid" and _negated(text, match.start()):
        return "invalid"
    return label


def _extract_integer(text: str, *, first: bool) -> Optional[str]:
    matches = list(_NUMBER_RE.finditer(text))
    if not matches:
        return None
    token = (matches[0] if first else matches[-1]).group(1).lower()
    value = _NUMBER_WORDS.get(token)
    return str(value) if value is not None else str(int(token))


def _extract_for_spec(text: str, spec: CategorySpec, *, first: bool) -> Optional[str]:
    if spec.kind == "integer":
        return _extract_integer(text, first=first)
    if spec.labels == ("valid", "invalid"):
        return _extract_valid(text, first=first)
    return _extract_yes_no(text, first=first)


def extract_answer(category: str, reply: str) -> tuple[Optional[str], Optional[ExtractionMethod]]:
    """Pull a normalized answer token out of *reply* for *category*.

    Returns ``(answer, method)``; both are ``None`` when nothing gradable is
    present. Unknown categories yield ``(None, None)`` rather than a guess.
    """
    spec = CATEGORY_SPECS.get(category)
    if spec is None or not reply or not reply.strip():
        return None, None

    tail = _marker_tail(reply)
    if tail is not None:
        answer = _extract_for_spec(tail, spec, first=True)
        if answer is not None:
            return answer, "marker"

    answer = _extract_for_spec(reply, spec, first=False)
    return (answer, "scan") if answer is not None else (None, None)


def normalize_official(category: str, official_answer: str) -> str:
    """Normalize a dataset answer into the same token space as :func:`extract_answer`."""
    spec = CATEGORY_SPECS.get(category)
    text = official_answer.strip()
    if spec is not None and spec.kind == "integer":
        return str(int(text))
    return text.lower()


def grade(category: str, official_answer: str, reply: str) -> Grade:
    """Grade one assistant *reply* against the dataset's *official_answer*."""
    extracted, method = extract_answer(category, reply)
    if extracted is None:
        return Grade(extracted=None, correct=None, method=None)
    return Grade(
        extracted=extracted,
        correct=extracted == normalize_official(category, official_answer),
        method=method,
    )
