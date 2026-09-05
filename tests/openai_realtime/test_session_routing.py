from queue import Queue
from threading import Event

import numpy as np
import pytest
from openai.types.realtime import RealtimeSessionCreateRequest, SessionUpdateEvent

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.session_routing import SessionRouting
from speech_to_speech.STT.openai_compatible_handler import OpenAICompatibleSTTHandler


def routing():
    return SessionRouting.model_validate({
        "id": "allocated-session", "pipeline": "qwen-gemma-openai",
        "routes": {
            "stt": {"model": "qwen-asr", "provider": "vllm", "protocol": "transcriptions"},
            "llm": {"model": "gemma", "provider": "vllm", "protocol": "chat_completions"},
            "tts": {"model": "mini-tts", "provider": "openai", "protocol": "speech", "voice": "alloy"},
        },
    })


def test_admitted_model_update_is_atomic_and_unconfigured_sessions_keep_existing_behavior():
    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    sid = service.register(routing=routing())
    cfg = service._state(sid).runtime_config
    event = SessionUpdateEvent(type="session.update", session=RealtimeSessionCreateRequest(type="realtime", model="other", instructions="changed"))
    before = cfg.session.model_dump()
    assert service.session.handle_session_update(sid, event) is not None
    assert cfg.session.model_dump() == before
    assert service.build_session_created(sid).session.model == "gemma"
    plain = service.register()
    assert service.session.handle_session_update(plain, event) is None
    assert service._state(plain).runtime_config.session.model == "other"
    assert cfg.routing.id == "allocated-session"


def test_stt_operation_captures_the_admitted_route_without_mutating_handler_defaults(monkeypatch):
    monkeypatch.setattr(OpenAICompatibleSTTHandler, "warmup", lambda self: None)
    handler = OpenAICompatibleSTTHandler(Event(), queue_in=Queue(), queue_out=Queue(), setup_kwargs={"base_url": "http://gateway/v1", "model": "default", "api_key": "gateway-key"})
    cfg = RuntimeConfig(routing=routing())
    first = handler._make_operation(np.zeros(160), runtime_config=cfg)
    second = handler._make_operation(np.zeros(160))
    assert first.model == "qwen-asr"
    assert first.extra_headers == {"X-Speech-Provider": "vllm", "X-Speech-Session-Id": "allocated-session"}
    assert first.api_key == "gateway-key"
    assert first.endpoint_url == "http://gateway/v1/audio/transcriptions"
    assert second.model == "default"
    assert second.extra_headers is None
    with pytest.raises(ValueError):
        cfg.routing.routes.stt.model = "changed"
