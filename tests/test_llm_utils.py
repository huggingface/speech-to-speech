from speech_to_speech.LLM.utils import remove_unspeechable


def test_remove_unspeechable_normalizes_smart_apostrophes() -> None:
    assert remove_unspeechable("I’ll reply if here’s the plan.") == "I'll reply if here's the plan."


def test_remove_unspeechable_keeps_text_and_drops_emoji() -> None:
    assert remove_unspeechable("Hello 👋 lobster 🦞") == "Hello  lobster "


def test_remove_unspeechable_strips_bold():
    assert remove_unspeechable("**bold**") == "bold"


def test_remove_unspeechable_strips_italic():
    assert remove_unspeechable("*italic*") == "italic"


def test_remove_unspeechable_strips_inline_code():
    assert remove_unspeechable("`code`") == "code"


def test_remove_unspeechable_strips_heading():
    assert remove_unspeechable("# Heading") == "Heading"


def test_remove_unspeechable_strips_bullets():
    assert remove_unspeechable("- item one\n- item two") == "item one\nitem two"


def test_remove_unspeechable_mixed_markdown():
    assert remove_unspeechable("**Hello** world and `code` here") == "Hello world and code here"


def test_remove_unspeechable_strips_fenced_code():
    assert remove_unspeechable("```python\nprint(1)\n```") == ""
