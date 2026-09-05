from queue import Queue
from threading import Event

import pytest
from openai.types.realtime import RealtimeSessionCreateRequest, SessionUpdateEvent

from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.session_routing import SessionRouting
from speech_to_speech.LLM.chat import make_user_message


def route(model="first", *, context=32768, tools=True, images=True, audio=True):
    return SessionRouting.model_validate(
        {
            "id": "allocated-session",
            "pipeline": model,
            "updates_enabled": True,
            "routes": {
                "stt": None,
                "tts": None,
                "llm": {
                    "model": model,
                    "provider": "hf",
                    "protocol": "chat_completions",
                    "capabilities": {"context_window": context, "tools": tools, "images": images, "audio_input": audio},
                },
            },
        }
    )


def event(**fields):
    return SessionUpdateEvent(type="session.update", session=RealtimeSessionCreateRequest(type="realtime", **fields))


def test_switch_preserves_identity_and_context_and_reports_effective_selection():
    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    sid = service.register(routing=route())
    state = service._state(sid)
    old = state.runtime_config
    old.chat.add_item(make_user_message("Remember the blue bicycle."))
    conversation_id = state.conversation_id
    assert (
        service.handle_session_update(
            sid, event(model="second", instructions="new instructions"), routing=route("second")
        )
        is None
    )
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
    before.chat.add_item(
        RealtimeConversationItemFunctionCall(type="function_call", name="look", call_id="call_pending", arguments="{}")
    )
    assert service.handle_session_update(sid, event(model="second"), routing=route("second")) is not None
    assert state.runtime_config is before


@pytest.mark.parametrize("context, accepted", [(8192, False), (32768, True), (65536, True)])
def test_llm_removal_preserves_context_floor_until_session_ends(context, accepted):
    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    sid = service.register(routing=route())
    state = service._state(sid)
    chat = state.runtime_config.chat
    chat.add_item(make_user_message("Keep this conversation while the LLM is disabled."))
    disabled = route().model_copy(update={"routes": route().routes.model_copy(update={"llm": None})})
    assert service.handle_session_update(sid, event(models={"llm": None}), routing=disabled) is None
    before = state.runtime_config
    error = service.handle_session_update(
        sid, event(model="second", instructions="new instructions"), routing=route("second", context=context)
    )
    assert (error is None) == accepted
    assert state.runtime_config.chat is chat
    assert "Keep this conversation" in str(chat.to_transformers_chat())
    if not accepted:
        assert state.runtime_config is before
        assert state.runtime_config.routing.routes.llm is None
        assert state.runtime_config.session.instructions != "new instructions"

    # The constraint belongs to this conversation, never to a reused CPU slot.
    service.unregister(sid)
    new_sid = service.register(routing=disabled)
    assert service.handle_session_update(new_sid, event(model="small"), routing=route("small", context=8192)) is None


def test_public_models_cannot_change_or_fake_the_trusted_selection():
    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    sid = service.register(routing=route())
    before = service._state(sid).runtime_config
    assert service.handle_session_update(sid, event(models={"llm": "second"}, instructions="changed")) is not None
    assert service._state(sid).runtime_config is before


def test_removed_stages_can_be_added_again_and_bad_voice_updates_are_atomic():
    from speech_to_speech.api.openai_realtime.session_routing import SpeechRoute

    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    sid = service.register(routing=route())
    with_tts = route().model_copy(
        update={
            "routes": route().routes.model_copy(
                update={
                    "tts": SpeechRoute(
                        model="tts", provider="hf", protocol="speech", voice="aiden", voices=["aiden", "alloy"]
                    )
                }
            )
        }
    )
    assert service.handle_session_update(sid, event(models={"tts": "tts"}), routing=with_tts) is None
    assert service.build_session_updated(sid).session.output_modalities == ["audio"]
    before = service._state(sid).runtime_config
    error = service.handle_session_update(
        sid, event(instructions="must not apply", audio={"output": {"voice": "missing"}}), routing=with_tts
    )
    assert error is not None
    assert service._state(sid).runtime_config is before
    assert service.handle_session_update(sid, event(models={"tts": None}), routing=route()) is None
    assert service.build_session_updated(sid).session.output_modalities == ["text"]


def test_media_and_completed_tool_history_require_destination_capabilities():
    from openai.types.realtime import RealtimeConversationItemFunctionCall, RealtimeConversationItemFunctionCallOutput

    from speech_to_speech.LLM.chat import UserContent, make_user_audio_message

    image_message = make_user_message("")
    image_message.content = [UserContent(type="input_image", image_url="https://example.com/image.png")]

    for item, selected in [
        (image_message, route("second", images=False)),
        (make_user_audio_message("AA=="), route("second", audio=False)),
    ]:
        service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
        sid = service.register(routing=route())
        cfg = service._state(sid).runtime_config
        cfg.chat.add_item(item)
        assert service.handle_session_update(sid, event(model="second"), routing=selected) is not None
        assert service._state(sid).runtime_config is cfg
    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    sid = service.register(routing=route())
    cfg = service._state(sid).runtime_config
    cfg.chat.add_item(
        RealtimeConversationItemFunctionCall(type="function_call", name="look", call_id="call_done", arguments="{}")
    )
    cfg.chat.add_item(
        RealtimeConversationItemFunctionCallOutput(type="function_call_output", call_id="call_done", output="result")
    )
    assert not cfg.chat.has_pending_tool_calls()
    assert service.handle_session_update(sid, event(model="second"), routing=route("second", tools=False)) is not None
    assert service._state(sid).runtime_config is cfg


@pytest.fixture
def running_unit():
    from threading import Thread
    from types import SimpleNamespace
    from unittest.mock import Mock

    from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
    from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler
    from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
    from speech_to_speech.pipeline.cancel_scope import CancelScope
    from speech_to_speech.STT.openai_compatible_handler import OpenAICompatibleSTTHandler
    from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier
    from speech_to_speech.TTS.openai_compatible_handler import OpenAICompatibleTTSHandler

    stop, listen, cancel = Event(), Event(), CancelScope()
    queues = [Queue() for _ in range(7)]
    incoming, transcription, prompts, lm_output, speech, outgoing, events = queues
    stt = OpenAICompatibleSTTHandler(stop, incoming, transcription, setup_kwargs={"warmup_enabled": False})
    notifier = TranscriptionNotifier(stop, transcription, prompts, setup_kwargs={"text_output_queue": events})
    llm = ChatCompletionsApiModelHandler(
        stop,
        prompts,
        lm_output,
        setup_kwargs={"api_key": "test", "stream": False, "warmup_enabled": False, "cancel_scope": cancel},
    )
    create = Mock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="It is blue.", tool_calls=[]))], usage=None
        )
    )
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    processor = LMOutputProcessor(stop, lm_output, speech, setup_kwargs={"text_output_queue": events})
    tts = OpenAICompatibleTTSHandler(
        stop, speech, outgoing, setup_kwargs={"warmup_enabled": False, "should_listen": listen, "cancel_scope": cancel}
    )
    unit = PipelineUnit(
        index=0,
        service=RealtimeService(text_prompt_queue=prompts, should_listen=listen),
        cancel_scope=cancel,
        should_listen=listen,
        response_playing=Event(),
        input_queue=incoming,
        output_queue=outgoing,
        text_output_queue=events,
        text_prompt_queue=prompts,
        handlers=[stt, notifier, llm, processor, tts],
    )
    workers = [Thread(target=handler.run, daemon=True) for handler in unit.handlers]
    for worker in workers:
        worker.start()
    yield unit, stop, create
    stop.set()
    for worker in workers:
        worker.join(timeout=2)
        assert not worker.is_alive()


def send_switch(ws, selected, *, update_id="update", **session):
    import json

    from openai.resources.realtime.realtime import RealtimeConnection

    class AdmissionProxy:
        def send(self, message):
            raw = json.loads(message)
            assert "_session_routing" not in raw
            raw["_session_routing"] = {"update_id": update_id, "routing": selected.model_dump()}
            ws.send_json(raw)

    # Use the installed official SDK serializer for the public event. Only the
    # admission proxy attaches the already-reserved private route proposal.
    RealtimeConnection(connection=AdmissionProxy()).session.update(
        session={"type": "realtime", **session}, event_id="client-update"
    )
    return ws.receive_json()


def test_websocket_switch_drains_handlers_and_next_request_uses_retained_context(running_unit):
    import json

    from starlette.testclient import TestClient

    from speech_to_speech.api.openai_realtime.websocket_router import create_app

    unit, stop, create = running_unit
    empty = route().model_copy(update={"routes": route().routes.model_copy(update={"llm": None})})
    with TestClient(create_app([unit], stop, session_routing_enabled=True)) as client:
        with client.websocket_connect(
            "/v1/realtime", headers={"X-Speech-Session-Routing": empty.model_dump_json()}
        ) as ws:
            created = ws.receive_json()
            sid = created["session"]["id"]
            for index, model in enumerate(("first", "second")):
                updated = send_switch(ws, route(model), model=model)
                assert updated["type"] == "session.updated", updated
                assert updated["_session_routing"] == "update"
                assert updated["session"]["id"] == sid
                ws.send_json(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "Remember the blue bicycle." if index == 0 else "Which color?",
                                }
                            ],
                        },
                    }
                )
                assert ws.receive_json()["type"] == "conversation.item.created"
                ws.send_json({"type": "response.create"})
                received = []
                while not received or received[-1]["type"] != "response.done":
                    received.append(ws.receive_json())
                assert received[-1]["response"]["status"] == "completed"
                assert any(event["type"] == "response.output_text.delta" for event in received)
                assert not any(event["type"] == "response.output_audio.delta" for event in received)
                assert create.call_args.kwargs["model"] == model
                assert "blue bicycle" in json.dumps(create.call_args.kwargs["messages"])
                assert create.call_args.kwargs["extra_headers"]["X-Speech-Session-Id"] == "allocated-session"


def test_cancelled_generation_must_drain_before_switching(running_unit):
    from starlette.testclient import TestClient

    from speech_to_speech.api.openai_realtime.websocket_router import create_app

    unit, stop, create = running_unit
    entered, release = Event(), Event()
    response = create.return_value

    def blocked(**kwargs):
        entered.set()
        assert release.wait(5)
        return response

    create.side_effect = blocked
    try:
        with TestClient(create_app([unit], stop, session_routing_enabled=True)) as client:
            with client.websocket_connect(
                "/v1/realtime", headers={"X-Speech-Session-Routing": route().model_dump_json()}
            ) as ws:
                ws.receive_json()
                ws.send_json(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Hello"}],
                        },
                    }
                )
                ws.receive_json()
                ws.send_json({"type": "response.create"})
                assert ws.receive_json()["type"] == "response.created"
                assert entered.wait(1)
                ws.send_json({"type": "response.cancel"})
                assert ws.receive_json()["type"] == "response.done"
                rejected = send_switch(ws, route("second"), model="second", instructions="must not apply")
                assert rejected["type"] == "error"
                assert rejected["error"]["event_id"] == "client-update"
                assert unit.service._state(unit.session.session_id).runtime_config.session.model == "first"
                release.set()
    finally:
        release.set()
