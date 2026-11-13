"""Every TTS handler must honor the speculative turn gate the same way.

Stale work is gated before any model is touched, so handlers are built with
``__new__`` and need no model, device or network.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import threading

import pytest

import speech_to_speech.TTS as tts_package
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.TTS.base_tts_handler import BaseTTSHandler
from tests.turns import reopen


def _discover() -> tuple[dict[str, type[BaseTTSHandler]], dict[str, str]]:
    """Find concrete handlers and retain import failures as explicit skips."""
    found: dict[str, type[BaseTTSHandler]] = {}
    unavailable: dict[str, str] = {}
    for module_info in pkgutil.iter_modules(tts_package.__path__):
        try:
            module = importlib.import_module(f"{tts_package.__name__}.{module_info.name}")
        except ImportError as exc:
            unavailable[module_info.name] = f"{module_info.name} unavailable: {type(exc).__name__}: {exc}"
            continue
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, BaseTTSHandler) and cls is not BaseTTSHandler and cls.__module__ == module.__name__:
                found[name] = cls
    return found, unavailable


HANDLERS, UNAVAILABLE = _discover()
HANDLER_CASES = [pytest.param(name, id=name) for name in sorted(HANDLERS)] + [
    pytest.param(name, id=name, marks=pytest.mark.skip(reason=reason)) for name, reason in sorted(UNAVAILABLE.items())
]


def _stale_handler(cls: type[BaseTTSHandler]) -> tuple[BaseTTSHandler, str, int]:
    tracker = SpeculativeTurnTracker()
    turn_id, revision = tracker.start_turn()
    reopen(tracker)  # the turn is now superseded
    handler = cls.__new__(cls)
    handler.speculative_turns = tracker
    handler.cancel_scope = None
    handler._failed_responses = set()  # type: ignore[attr-defined]
    handler._operation_lock = threading.Lock()  # type: ignore[attr-defined]
    return handler, turn_id, revision


def test_tts_handler_discovery_finds_handlers() -> None:
    """Fail loudly if discovery silently finds nothing (for example after a rename)."""
    assert len(HANDLERS) >= 8, sorted(HANDLERS)


@pytest.mark.parametrize("name", HANDLER_CASES)
def test_stale_text_is_dropped_without_synthesis(name: str) -> None:
    handler, turn_id, revision = _stale_handler(HANDLERS[name])
    assert list(handler.process(TTSInput(text="Old", turn_id=turn_id, turn_revision=revision))) == []


@pytest.mark.parametrize("name", HANDLER_CASES)
def test_stale_keyed_end_of_response_is_forwarded_for_cleanup(name: str) -> None:
    handler, turn_id, revision = _stale_handler(HANDLERS[name])
    end = EndOfResponse(response_key="response_1", turn_id=turn_id, turn_revision=revision)
    assert list(handler.process(end)) == [AUDIO_RESPONSE_DONE]
    assert end.cleanup_only is True


@pytest.mark.parametrize("name", HANDLER_CASES)
def test_stale_unkeyed_end_of_response_is_dropped(name: str) -> None:
    handler, turn_id, revision = _stale_handler(HANDLERS[name])
    assert list(handler.process(EndOfResponse(turn_id=turn_id, turn_revision=revision))) == []
