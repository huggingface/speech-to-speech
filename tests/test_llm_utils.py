from speech_to_speech.LLM.utils import remove_markdown, remove_unspeechable


def test_remove_unspeechable_normalizes_smart_apostrophes() -> None:
    assert remove_unspeechable("I’ll reply if here’s the plan.") == "I'll reply if here's the plan."


def test_remove_unspeechable_keeps_text_and_drops_emoji() -> None:
    assert remove_unspeechable("Hello 👋 lobster 🦞") == "Hello  lobster "


def test_remove_markdown_strips_bold_and_italic() -> None:
    assert remove_markdown("**bold** and *italic* text") == "bold and italic text"


def test_remove_markdown_keeps_snake_case_identifiers() -> None:
    assert remove_markdown("function_call_output") == "function_call_output"


def test_remove_markdown_strips_bullets_without_eating_following_lines() -> None:
    assert remove_markdown("* first\n* second\n* third") == "first\nsecond\nthird"
    assert remove_markdown("- one\n- two") == "one\ntwo"


def test_remove_markdown_strips_headings() -> None:
    assert remove_markdown("# Title\nsome text") == "Title\nsome text"
    assert remove_markdown("### Subheading") == "Subheading"


def test_remove_markdown_does_not_eat_multiplication() -> None:
    assert remove_markdown("2 * 3 * 4") == "2 * 3 * 4"


def test_remove_markdown_strips_delimiter_glued_to_punctuation() -> None:
    """A closing '**' butted against punctuation, not a space, still leaked
    before: `(?!\\s)` only checked for a following space, not for a following
    non-word character in general."""
    assert remove_markdown("Do you mean snake case**?") == "Do you mean snake case?"
    assert remove_markdown("a theme/topic**.") == "a theme/topic."


def test_remove_markdown_strips_nested_bold_and_code() -> None:
    assert remove_markdown("**bold with `code`**") == "bold with code"


def test_remove_markdown_strips_inline_code() -> None:
    assert remove_markdown("`inline`") == "inline"


def test_remove_markdown_strips_fenced_code_block_and_language_tag() -> None:
    assert remove_markdown("```python\nname = 'Alice'\n```") == "name = 'Alice'\n"
    assert remove_markdown("```\ncode\n```") == "code\n"


def test_remove_markdown_rewrites_links_to_visible_text_only() -> None:
    assert remove_markdown("Check the [docs](https://example.com/en/x) for details.") == "Check the docs for details."
    assert remove_markdown("[text](url)") == "text"


def test_remove_markdown_is_streaming_safe_across_split_deltas() -> None:
    """remove_markdown must be applied to complete text, not per-delta: a
    delimiter pair split across two deltas (*ita / lic*) has nothing to match
    on its own, so callers accumulate first and strip once, as tested here."""
    deltas = ["*ita", "lic* is a word."]
    accumulated = "".join(deltas)
    assert remove_markdown(accumulated) == "italic is a word."
