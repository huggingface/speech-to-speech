from queue import Queue
from threading import Event

import pytest
from openai.types.realtime import RealtimeSessionCreateRequest, SessionUpdateEvent

from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.session_routing import SessionRouting
from speech_to_speech.LLM.chat import make_user_message


def route(model="first", *, context=32768, tools=True, images=True, audio=True):
    return SessionRouting.model_validate({
        "id": "allocated-session", "pipeline": model, "updates_enabled": True,
        "routes": {"stt": None, "tts": None, "llm": {
            "model": model, "provider": "hf", "protocol": "chat_completions",
            "capabilities": {"context_window": context, "tools": tools, "images": images, "audio_input": audio},
        }},
    })


def event(**fields):
    return SessionUpdateEvent(type="session.update", session=RealtimeSessionCreateRequest(type="realtime", **fields))


def test_switch_preserves_identity_and_context_and_reports_effective_selection():
    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    sid = service.register(routing=route())
    state = service._state(sid)
    old = state.runtime_config
    old.chat.add_item(make_user_message("Remember the blue bicycle."))
    conversation_id = state.conversation_id
    assert service.handle_session_update(sid, event(model="second", instructions="new instructions"), routing=route("second")) is None
    assert state.runtime_config.routing.id == old.routing.id
    assert state.runtime_config.chat is old.chat
    assert state.conversation_id == conversation_id
    assert "blue bicycle" in str(state.runtime_config.chat.to_transformers_chat())
    updated = service.build_session_updated(sid).model_dump()["session"]
    assert updated["id"] == sid
    assert updated["model"] == "second"
    assert updated["models"]["llm"] == {"model": "second", "provider": "hf"}
    assert updated["instructions"] == "new instructions"


@pytest.mark.parametrize("busy", ["in_response", "response_pending"])
def test_busy_combined_update_is_atomic(busy):
    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    sid = service.register(routing=route())
    state = service._state(sid)
    setattr(state, busy, True)
    before = state.runtime_config
    error = service.handle_session_update(sid, event(model="second", instructions="changed"), routing=route("second"))
    assert error.type == "error"
    assert state.runtime_config is before


def test_smaller_context_and_unresolved_tools_are_rejected_without_changes():
    from openai.types.realtime import RealtimeConversationItemFunctionCall

    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    sid = service.register(routing=route())
    state = service._state(sid)
    before = state.runtime_config
    assert service.handle_session_update(sid, event(model="second"), routing=route("second", context=8192)) is not None
    assert state.runtime_config is before
    before.chat.add_item(RealtimeConversationItemFunctionCall(type="function_call", name="look", call_id="pending", arguments="{}"))
    assert service.handle_session_update(sid, event(model="second"), routing=route("second")) is not None
    assert state.runtime_config is before


def test_public_models_cannot_change_or_fake_the_trusted_selection():
    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    sid = service.register(routing=route())
    before = service._state(sid).runtime_config
    assert service.handle_session_update(sid, event(models={"llm": "second"}, instructions="changed")) is not None
    assert service._state(sid).runtime_config is before
