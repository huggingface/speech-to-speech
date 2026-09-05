from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from openai.types.realtime import RealtimeSessionCreateRequest, SessionUpdateEvent
from starlette.testclient import TestClient

from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.session_routing import SessionRouting
from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.LLM.chat import make_user_message
from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler
from speech_to_speech.LLM.responses_api_language_model import ResponsesApiModelHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import GenerateResponseRequest
from speech_to_speech.STT.openai_compatible_handler import OpenAICompatibleSTTHandler
from speech_to_speech.TTS.openai_compatible_handler import OpenAICompatibleTTSHandler


def routing():
    return SessionRouting.model_validate(
        {
            "id": "allocated-session",
            "pipeline": "qwen-gemma-openai",
            "routes": {
                "stt": {"model": "qwen-asr", "provider": "vllm", "protocol": "transcriptions"},
                "llm": {"model": "gemma", "provider": "vllm", "protocol": "chat_completions"},
                "tts": {"model": "mini-tts", "provider": "openai", "protocol": "speech", "voice": "alloy"},
            },
        }
    )


def test_allocator_ids_accept_urlsafe_leading_punctuation():
    for session_id in ("-allocated", "_allocated"):
        payload = routing().model_dump()
        payload["id"] = session_id
        assert SessionRouting.model_validate(payload).id == session_id


def test_admitted_model_update_is_atomic_and_unconfigured_sessions_keep_existing_behavior():
    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    sid = service.register(routing=routing())
    cfg = service._state(sid).runtime_config
    event = SessionUpdateEvent(
        type="session.update",
        session=RealtimeSessionCreateRequest(type="realtime", model="other", instructions="changed"),
    )
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
    handler = OpenAICompatibleSTTHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_kwargs={"base_url": "http://gateway/v1", "model": "default", "api_key": "gateway-key"},
    )
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


@pytest.mark.parametrize("handler_type", [ChatCompletionsApiModelHandler, ResponsesApiModelHandler])
def test_llm_requests_and_compaction_capture_routes_and_preserve_context(monkeypatch, handler_type):
    monkeypatch.setattr(handler_type, "warmup", lambda self: None)
    handler = handler_type(
        Event(),
        Queue(),
        Queue(),
        setup_kwargs={"model_name": "default", "api_key": "gateway-key", "stream": False, "disable_thinking": False},
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hello.", tool_calls=[]))],
        output=[],
        output_text="summary",
        usage=None,
    )
    create = Mock(return_value=response)
    handler.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)), responses=SimpleNamespace(create=create)
    )
    admitted = routing()
    if handler_type is ResponsesApiModelHandler:
        data = admitted.model_dump()
        data["routes"]["llm"]["protocol"] = "responses"
        admitted = SessionRouting.model_validate(data)
    cfg = RuntimeConfig(routing=admitted)
    cfg.chat.add_item(make_user_message("Remember the blue bicycle."))
    cfg.chat.add_item(make_user_message("What color was it?"))
    list(handler.process(GenerateResponseRequest(runtime_config=cfg)))
    assert create.call_args.kwargs["model"] == "gemma"
    assert create.call_args.kwargs["extra_headers"] == admitted.headers("llm")
    assert "blue bicycle" in str(create.call_args.kwargs.get("messages", create.call_args.kwargs.get("input")))
    assert handler.model_name == "default"
    generate = handler._build_compaction_generate_fn({"model": "gemma", "extra_headers": admitted.headers("llm")})
    generate("summarize", "history")
    assert create.call_args.kwargs["model"] == "gemma"
    assert create.call_args.kwargs["extra_headers"] == admitted.headers("llm")


def test_tts_routing_keeps_transport_and_voice_updates(monkeypatch):
    monkeypatch.setattr(OpenAICompatibleTTSHandler, "warmup", lambda self: None)
    handler = OpenAICompatibleTTSHandler(
        Event(),
        Queue(),
        Queue(),
        setup_kwargs={
            "base_url": "http://gateway/v1",
            "api_key": "gateway-key",
            "model": "default",
            "stream": False,
            "should_listen": Event(),
        },
    )
    cfg = RuntimeConfig(routing=routing())
    operation = handler._make_operation(text="Hello.", voice="coral", runtime_config=cfg)
    assert operation.payload["model"] == "mini-tts"
    assert operation.payload["voice"] == "coral"
    assert operation.extra_headers == routing().headers("tts")
    assert operation.api_key == "gateway-key"
    assert operation.endpoint_url == "http://gateway/v1/audio/speech"
    assert handler._make_operation(text="Hello.", voice="alloy").payload["model"] == "default"


def unit():
    return PipelineUnit(
        index=0,
        service=RealtimeService(text_prompt_queue=Queue(), should_listen=Event()),
        cancel_scope=CancelScope(),
        should_listen=Event(),
        response_playing=Event(),
        input_queue=Queue(),
        output_queue=Queue(),
        text_output_queue=Queue(),
        text_prompt_queue=Queue(),
        handlers=[
            object.__new__(OpenAICompatibleSTTHandler),
            object.__new__(OpenAICompatibleTTSHandler),
            object.__new__(ChatCompletionsApiModelHandler),
        ],
    )


@pytest.mark.parametrize("enabled,invalid", [(False, False), (True, True)])
def test_untrusted_or_wrong_protocol_handoff_is_rejected_before_claim(enabled, invalid):
    pipeline = unit()
    payload = routing().model_dump()
    if invalid:
        payload["routes"]["llm"]["protocol"] = "responses"
    import json

    with TestClient(create_app([pipeline], Event(), session_routing_enabled=enabled)) as client:
        with client.websocket_connect("/v1/realtime", headers={"X-Speech-Session-Routing": json.dumps(payload)}) as ws:
            assert ws.receive_json()["type"] == "error"
            assert pipeline.session is None
            assert pipeline.service.total_usage.connections == 0


def test_valid_handoff_sets_effective_session_before_created_event():
    pipeline = unit()
    with TestClient(create_app([pipeline], Event(), session_routing_enabled=True)) as client:
        with client.websocket_connect(
            "/v1/realtime", headers={"X-Speech-Session-Routing": routing().model_dump_json()}
        ) as ws:
            event = ws.receive_json()
            assert event["session"]["model"] == "gemma"
            assert event["session"]["audio"]["output"]["voice"] == "alloy"
            cfg = pipeline.service._state(event["session"]["id"]).runtime_config
            assert cfg.routing == routing()


def test_routing_rejects_local_handlers_at_startup():
    pipeline = unit()
    pipeline.handlers = []
    with pytest.raises(ValueError, match="remote"):
        create_app([pipeline], Event(), session_routing_enabled=True)
