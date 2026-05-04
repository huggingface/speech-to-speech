import builtins
import importlib
import sys

import pytest

from speech_to_speech.LLM.utils import remove_unspeechable


def test_remove_unspeechable_normalizes_smart_apostrophes() -> None:
    assert remove_unspeechable("I’ll reply if here’s the plan.") == "I'll reply if here's the plan."


def test_remove_unspeechable_keeps_text_and_drops_emoji() -> None:
    assert remove_unspeechable("Hello 👋 lobster 🦞") == "Hello  lobster "


def test_llm_utils_import_does_not_require_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    sys.modules.pop("speech_to_speech.LLM.utils", None)
    original_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("blocked pillow import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    module = importlib.import_module("speech_to_speech.LLM.utils")

    assert module.remove_unspeechable("hello") == "hello"


def test_image_url_to_pil_explains_missing_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    from speech_to_speech.LLM import utils

    original_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("blocked pillow import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match="Pillow is required"):
        utils.image_url_to_pil("data:image/png;base64,")
