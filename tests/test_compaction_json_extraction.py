"""Compaction responses tolerate trailing prose after the JSON object."""

import pytest

from speech_to_speech.LLM.compaction_prompt import _extract_json


def test_extract_json_ignores_trailing_prose_with_braces():
    text = '{"summary": "done"}\nNote: the {old} topic was dropped.'
    assert _extract_json(text) == {"summary": "done"}


def test_extract_json_still_rejects_no_object():
    with pytest.raises(ValueError, match="No JSON object found"):
        _extract_json("no object here")
