from queue import Queue
from threading import Event

import pytest
import torch
from openai.types.realtime import RealtimeSessionCreateRequest

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.VAD.vad_handler import VADHandler
from tests.test_vad_iterator import _FakeVADModel

SAMPLE_RATE = 16000


@pytest.fixture
def handler(monkeypatch: pytest.MonkeyPatch) -> VADHandler:
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: (_FakeVADModel([]), None))
    return VADHandler(
        Event(),
        Queue(),
        Queue(),
        setup_kwargs={
            "should_listen": Event(),
            "speculative_turns": SpeculativeTurnTracker(),
            "thresh": 0.6,
            "min_silence_ms": 64,
            "smart_turn": False,
        },
    )


def _session(turn_detection: dict | None = None) -> RuntimeConfig:
    audio_input = {"turn_detection": turn_detection} if turn_detection is not None else {}
    return RuntimeConfig(
        session=RealtimeSessionCreateRequest.model_validate({"type": "realtime", "audio": {"input": audio_input}})
    )


def _server_vad(**fields) -> dict:
    return {"type": "server_vad", "interrupt_response": True, **fields}


def _settings(handler: VADHandler) -> tuple[float, float]:
    return handler.iterator.threshold, handler.iterator.min_silence_samples


DEFAULTS = (0.6, SAMPLE_RATE * 64 / 1000)


def _override_and_end_session(handler: VADHandler) -> None:
    handler._apply_runtime_turn_detection(_session(_server_vad(silence_duration_ms=800, threshold=0.7)))
    assert _settings(handler) == (0.7, SAMPLE_RATE * 800 / 1000)
    handler.on_session_end()


def test_next_session_without_turn_detection_fields_uses_configured_defaults(handler: VADHandler) -> None:
    _override_and_end_session(handler)

    handler._apply_runtime_turn_detection(_session(_server_vad()))

    assert _settings(handler) == DEFAULTS


def test_next_session_without_turn_detection_uses_configured_defaults(handler: VADHandler) -> None:
    _override_and_end_session(handler)

    handler._apply_runtime_turn_detection(_session())

    assert _settings(handler) == DEFAULTS


def test_next_session_repeating_previous_overrides_applies_them(handler: VADHandler) -> None:
    _override_and_end_session(handler)

    handler._apply_runtime_turn_detection(_session(_server_vad(silence_duration_ms=800, threshold=0.7)))

    assert _settings(handler) == (0.7, SAMPLE_RATE * 800 / 1000)


def test_next_session_overriding_one_field_keeps_the_other_default(handler: VADHandler) -> None:
    _override_and_end_session(handler)

    handler._apply_runtime_turn_detection(_session(_server_vad(silence_duration_ms=300)))

    assert _settings(handler) == (0.6, SAMPLE_RATE * 300 / 1000)


def test_partial_update_within_session_keeps_earlier_overrides(handler: VADHandler) -> None:
    config = _session(_server_vad(silence_duration_ms=800, threshold=0.7))
    handler._apply_runtime_turn_detection(config)

    config.apply_session_update(
        RealtimeSessionCreateRequest.model_validate(
            {"type": "realtime", "audio": {"input": {"turn_detection": _server_vad(threshold=0.4)}}}
        )
    )
    handler._apply_runtime_turn_detection(config)

    assert _settings(handler) == (0.4, SAMPLE_RATE * 800 / 1000)
