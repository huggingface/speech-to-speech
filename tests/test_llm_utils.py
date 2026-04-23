from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from LLM.utils import remove_unspeechable


def test_remove_unspeechable_normalizes_smart_apostrophes() -> None:
    assert (
        remove_unspeechable("I’ll reply if here’s the plan.")
        == "I'll reply if here's the plan."
    )


def test_remove_unspeechable_keeps_text_and_drops_emoji() -> None:
    assert remove_unspeechable("Hello 👋 lobster 🦞") == "Hello  lobster "
