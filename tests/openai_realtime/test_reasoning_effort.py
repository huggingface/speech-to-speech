"""Client-selected reasoning effort via Realtime ``session.update`` / ``response.create``."""

from types import SimpleNamespace

import pytest
from openai.types.realtime import (
    RealtimeErrorEvent,
    RealtimeSessionCreateRequest,
    ResponseCreateEvent,
    SessionUpdateEvent,
)
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig, resolve_reasoning_effort
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.pipeline.messages import GenerateResponseRequest
from tests.test_chat_completions_backend import _chunk, _FakeStream
from tests.test_chat_completions_backend import _make_handler as _make_chat_handler
from tests.test_responses_api_language_model import (
    _make_audio_request,
    _make_output_item_done_event,
    _make_stream,
    _make_text_delta_event,
)
from tests.test_responses_api_language_model import _make_handler as _make_responses_handler


def _session_update(session: dict) -> SessionUpdateEvent:
    return SessionUpdateEvent.model_validate({"type": "session.update", "session": {"type": "realtime", **session}})


def _runtime_config(session_effort: str | None = None) -> RuntimeConfig:
    session = {"type": "realtime", "instructions": "Be brief."}
    if session_effort is not None:
        session["reasoning"] = {"effort": session_effort}
    cfg = RuntimeConfig(chat=Chat(5), session=RealtimeSessionCreateRequest.model_validate(session))
    cfg.chat.add_item(make_user_message("Hi"))
    return cfg


def _response(effort: str) -> RealtimeResponseCreateParams:
    return RealtimeResponseCreateParams.model_validate({"reasoning": {"effort": effort}})


# ── Protocol: session.update / session.updated / response.create ─────────────


def test_session_update_stores_and_echoes_reasoning(service, conn_id, runtime_config):
    assert service.handle_session_update(conn_id, _session_update({"reasoning": {"effort": "high"}})) is None
    assert resolve_reasoning_effort(runtime_config) == "high"

    updated = service.build_session_updated(conn_id).model_dump(exclude_none=True)
    assert updated["session"]["reasoning"] == {"effort": "high"}


def test_partial_session_update_preserves_reasoning(service, conn_id, runtime_config):
    service.handle_session_update(conn_id, _session_update({"reasoning": {"effort": "low"}}))
    service.handle_session_update(conn_id, _session_update({"instructions": "Be concise"}))
    assert resolve_reasoning_effort(runtime_config) == "low"

    service.handle_session_update(conn_id, _session_update({"reasoning": {"effort": "xhigh"}}))
    assert resolve_reasoning_effort(runtime_config) == "xhigh"


def test_session_update_null_reasoning_restores_server_default(service, conn_id, runtime_config):
    service.handle_session_update(conn_id, _session_update({"reasoning": {"effort": "high"}}))
    service.handle_session_update(conn_id, _session_update({"reasoning": None}))
    assert resolve_reasoning_effort(runtime_config) is None


def test_session_update_empty_reasoning_object_keeps_effort(service, conn_id, runtime_config):
    service.handle_session_update(conn_id, _session_update({"reasoning": {"effort": "high"}}))
    assert service.handle_session_update(conn_id, _session_update({"reasoning": {}})) is None
    assert resolve_reasoning_effort(runtime_config) == "high"
    updated = service.build_session_updated(conn_id).model_dump(exclude_none=True)
    assert updated["session"]["reasoning"] == {"effort": "high"}


@pytest.mark.parametrize("reasoning", [{"effort": "maximum"}, {"effort": 3}, "high"])
def test_session_update_rejects_invalid_reasoning(service, conn_id, runtime_config, reasoning):
    error = service.handle_session_update(conn_id, _session_update({"reasoning": reasoning}))
    assert isinstance(error, RealtimeErrorEvent)
    assert error.error.type == "invalid_request_error"
    assert resolve_reasoning_effort(runtime_config) is None


def test_response_create_rejects_invalid_reasoning(service, conn_id):
    event = ResponseCreateEvent.model_validate(
        {"type": "response.create", "response": {"reasoning": {"effort": "turbo"}}}
    )
    error = service.handle_response_create(conn_id, event)
    assert isinstance(error, RealtimeErrorEvent)
    assert error.error.type == "invalid_request_error"


def test_response_effort_overrides_session_effort():
    cfg = _runtime_config("low")
    assert resolve_reasoning_effort(cfg) == "low"
    assert resolve_reasoning_effort(cfg, _response("high")) == "high"
    assert resolve_reasoning_effort(cfg, RealtimeResponseCreateParams()) == "low"


# ── Responses API backend ────────────────────────────────────────────────────


def _drive_responses(handler, request) -> dict:
    captured: dict = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _make_stream([_make_text_delta_event("Ok"), _make_output_item_done_event(content="Ok")])

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    list(handler.process(request))
    return captured


@pytest.mark.parametrize(
    ("session_effort", "response", "expected"),
    [
        (None, None, {"effort": "none"}),  # startup default
        ("high", None, {"effort": "high"}),
        ("high", "minimal", {"effort": "minimal"}),
    ],
)
def test_responses_backend_sends_client_effort(session_effort, response, expected):
    handler = _make_responses_handler(reasoning_effort="none")
    request = GenerateResponseRequest(
        runtime_config=_runtime_config(session_effort),
        response=_response(response) if response else None,
    )
    assert _drive_responses(handler, request)["reasoning"] == expected


@pytest.mark.parametrize(("session_effort", "response"), [("high", None), (None, "high")])
def test_responses_backend_client_effort_drops_startup_disable_thinking(session_effort, response):
    handler = _make_responses_handler(disable_thinking=True, reasoning_effort=None)
    request = GenerateResponseRequest(
        runtime_config=_runtime_config(session_effort),
        response=_response(response) if response else None,
    )
    captured = _drive_responses(handler, request)
    assert captured["reasoning"] == {"effort": "high"}
    assert captured["extra_body"] is None
    # Startup defaults are untouched; turns without a client effort keep them.
    assert handler._extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    captured = _drive_responses(handler, GenerateResponseRequest(runtime_config=_runtime_config()))
    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "reasoning" not in captured


def test_responses_backend_client_effort_without_startup_default():
    handler = _make_responses_handler(reasoning_effort=None)
    assert "reasoning" not in _drive_responses(handler, GenerateResponseRequest(runtime_config=_runtime_config()))
    request = GenerateResponseRequest(runtime_config=_runtime_config("medium"))
    assert _drive_responses(handler, request)["reasoning"] == {"effort": "medium"}


def test_responses_backend_audio_turn_sends_client_effort_as_chat_reasoning_effort():
    handler = _make_responses_handler(stream=False, disable_thinking=True, reasoning_effort=None)
    captured: dict = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Ok.", refusal=None, tool_calls=[]))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )

    handler.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    cfg = _runtime_config("high")
    list(handler.process(_make_audio_request(cfg)))
    assert captured["extra_body"] == {"reasoning_effort": "high"}


# ── Chat Completions backend ─────────────────────────────────────────────────


def _drive_chat(handler, request) -> dict:
    captured: dict = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _FakeStream([_chunk(content="ok.")])

    handler.client.chat.completions.create = create
    list(handler.process(request))
    return captured


def test_chat_backend_keeps_startup_extra_body_without_client_effort():
    handler = _make_chat_handler()
    captured = _drive_chat(handler, GenerateResponseRequest(runtime_config=_runtime_config()))
    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "reasoning_effort" not in captured


@pytest.mark.parametrize(
    ("session_effort", "response", "expected"),
    [("high", None, "high"), ("high", "low", "low"), (None, "medium", "medium")],
)
def test_chat_backend_client_effort_replaces_startup_extra_body(session_effort, response, expected):
    handler = _make_chat_handler()
    request = GenerateResponseRequest(
        runtime_config=_runtime_config(session_effort),
        response=_response(response) if response else None,
    )
    captured = _drive_chat(handler, request)
    assert captured["extra_body"] == {"reasoning_effort": expected}
    # Startup kwargs on the handler are untouched for later turns.
    assert handler._extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
