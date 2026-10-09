"""Spoken replies grade to the right answer token, and ambiguous ones grade to none."""

import pytest

from speech_to_speech.evals.big_bench_audio.scoring import (
    extract_answer,
    grade,
    normalize_official,
)


@pytest.mark.parametrize(
    ("category", "reply", "expected"),
    [
        ("web_of_lies", "Let me trace it. Final answer: No.", "no"),
        ("web_of_lies", "Osvaldo tells the truth, so my answer is Yes", "yes"),
        ("navigate", "You circle back. Final answer: Yes", "yes"),
        ("formal_fallacies", "Final answer: invalid", "invalid"),
        ("formal_fallacies", "So the argument is deductively valid.", "valid"),
        ("object_counting", "Final answer: 12", "12"),
        ("object_counting", "That gives seven instruments.", "7"),
    ],
)
def test_reply_yields_its_answer_token(category, reply, expected):
    assert extract_answer(category, reply)[0] == expected


def test_marker_beats_the_reasoning_that_precedes_it():
    reply = "First it looks valid, and valid again on the second premise. Final answer: invalid"
    assert extract_answer("formal_fallacies", reply) == ("invalid", "marker")


def test_scan_without_a_marker_takes_the_concluding_claim():
    """These questions need several steps, so the conclusion trails the reasoning."""
    reply = "At first it looked like no, but retracing the turns you do land back at the start. Yes."
    assert extract_answer("navigate", reply) == ("yes", "scan")


@pytest.mark.parametrize(
    "reply",
    ["The argument is not valid.", "That is not deductively valid.", "This isn't valid."],
)
def test_negated_valid_reads_as_invalid(reply):
    assert extract_answer("formal_fallacies", reply)[0] == "invalid"


def test_hedging_without_a_negation_keeps_the_stated_answer():
    assert extract_answer("formal_fallacies", "I am not sure, but the argument is valid.")[0] == "valid"


@pytest.mark.parametrize(
    ("category", "reply"),
    [
        # "correct" and "negative" are ordinary words here, not yes/no answers.
        ("navigate", "The correct approach is to track each turn."),
        ("navigate", "You end up facing the negative x direction."),
        ("web_of_lies", ""),
        ("object_counting", "I could not follow the list."),
    ],
)
def test_replies_without_an_answer_stay_unparsed(category, reply):
    assert extract_answer(category, reply) == (None, None)


def test_unknown_category_is_never_guessed():
    assert extract_answer("sports_trivia", "Final answer: Yes") == (None, None)


def test_unparsed_reply_grades_as_neither_right_nor_wrong():
    verdict = grade("web_of_lies", "Yes", "Hmm, hard to say.")
    assert verdict.correct is None
    assert verdict.parsed is False


def test_grade_compares_against_the_normalized_official_answer():
    assert grade("navigate", "No", "Final answer: no").correct is True
    assert grade("navigate", "No", "Final answer: yes").correct is False
    assert grade("object_counting", "07", "Final answer: seven").correct is True


def test_normalize_official_matches_the_extractor_token_space():
    assert normalize_official("formal_fallacies", "Invalid") == "invalid"
    assert normalize_official("object_counting", "07") == "7"
