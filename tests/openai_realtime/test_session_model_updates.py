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
                "stt": {"model": "asr-default", "provider": "hf", "protocol": "transcriptions"},
                "tts": {
                    "model": "tts-default",
                    "provider": "hf",
                    "protocol": "speech",
                    "voice": "aiden",
                    "voices": ["aiden", "alloy"],
                },
                "llm": {
                    "model": model,
                    "provider": "hf",
                    "protocol": "chat_completions",
                    "capabilities": {"context_window": context, "tools": tools, "images": images, "audio_input": audio},
                },
            },
        }
    )


def speech_route(model):
    from speech_to_speech.api.openai_realtime.session_routing import SpeechRoute, TranscriptionRoute

    selected = route(model)
    provider = f"{model}-provider"
    return selected.model_copy(
        update={
            "routes": selected.routes.model_copy(
                update={
                    "stt": TranscriptionRoute(model=f"asr-{model}", provider=provider, protocol="transcriptions"),
                    "llm": selected.routes.llm.model_copy(update={"provider": provider}),
                    "tts": SpeechRoute(
                        model=f"tts-{model}",
                        provider=provider,
                        protocol="speech",
                        voice="aiden" if model == "first" else "alloy",
                        voices=["aiden", "alloy"],
                    ),
                }
            )
        }
    )


def event(**fields):
    return SessionUpdateEvent(type="session.update", session=RealtimeSessionCreateRequest(type="realtime", **fields))


@pytest.mark.parametrize("stage", ["stt", "llm", "tts"])
def test_routes_require_all_three_stages(stage):
    from pydantic import ValidationError

    payload = route().model_dump()
    payload["routes"][stage] = None
    with pytest.raises(ValidationError):
        SessionRouting.model_validate(payload)


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


def test_smaller_context_window_does_not_block_a_switch():
    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    sid = service.register(routing=route(context=1_000_000))
    service._state(sid).runtime_config.chat.add_item(make_user_message("Remember the blue bicycle."))
    assert service.handle_session_update(sid, event(model="second"), routing=route("second", context=512_000)) is None
    cfg = service._state(sid).runtime_config
    assert cfg.routing.routes.llm.model == "second"
    assert "blue bicycle" in str(cfg.chat.to_transformers_chat())


def test_public_models_cannot_change_or_fake_the_trusted_selection():
    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    sid = service.register(routing=route())
    before = service._state(sid).runtime_config
    assert service.handle_session_update(sid, event(models={"llm": "second"}, instructions="changed")) is not None
    assert service._state(sid).runtime_config is before


def test_bad_voice_update_is_atomic():
    from speech_to_speech.api.openai_realtime.session_routing import SpeechRoute

    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    sid = service.register(routing=route())
    with_tts = route().model_copy(
        update={
            "routes": route().routes.model_copy(
                update={
                    "tts": SpeechRoute(
                        model="tts-alternative",
                        provider="hf",
                        protocol="speech",
                        voice="alloy",
                        voices=["aiden", "alloy"],
                    )
                }
            )
        }
    )
    before = service._state(sid).runtime_config
    error = service.handle_session_update(
        sid,
        event(models={"tts": "tts-alternative"}, instructions="must not apply", audio={"output": {"voice": "missing"}}),
        routing=with_tts,
    )
    assert error is not None
    assert service._state(sid).runtime_config is before
    assert service.handle_session_update(sid, event(models={"tts": "tts-alternative"}), routing=with_tts) is None
    assert service.build_session_updated(sid).session.audio.output.voice == "aiden"


def test_stt_switch_does_not_require_unchanged_llm_capabilities():
    from speech_to_speech.api.openai_realtime.session_routing import TranscriptionRoute

    initial_data = route().model_dump()
    initial_data["routes"]["llm"].pop("capabilities")
    initial = SessionRouting.model_validate(initial_data)
    destination = initial.model_copy(
        update={
            "routes": initial.routes.model_copy(
                update={"stt": TranscriptionRoute(model="asr-second", provider="other", protocol="transcriptions")}
            )
        }
    )
    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    sid = service.register(routing=initial)
    before = service._state(sid).runtime_config
    before.chat.add_item(make_user_message("Remember the blue bicycle."))
    assert service.handle_session_update(sid, event(tools=[{"type": "function", "name": "look"}])) is None

    assert service.handle_session_update(sid, event(models={"stt": "asr-second"}), routing=destination) is None
    after = service._state(sid).runtime_config
    assert after.chat is before.chat
    assert after.routing.routes.llm == initial.routes.llm
    assert after.session.tools == before.session.tools
    assert service.build_session_updated(sid).session.model_dump()["models"]["stt"] == {
        "model": "asr-second",
        "provider": "other",
    }


@pytest.mark.parametrize("supports_tools", [False, True])
def test_stt_switch_validates_new_tools_against_unchanged_llm(supports_tools):
    from speech_to_speech.api.openai_realtime.session_routing import TranscriptionRoute

    initial = route(tools=supports_tools)
    destination = initial.model_copy(
        update={
            "routes": initial.routes.model_copy(
                update={"stt": TranscriptionRoute(model="asr-second", provider="hf", protocol="transcriptions")}
            )
        }
    )
    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    sid = service.register(routing=initial)
    before = service._state(sid).runtime_config
    before_session = before.session.model_dump()
    tools = [{"type": "function", "name": "look", "parameters": {"type": "object", "properties": {}}}]

    error = service.handle_session_update(
        sid,
        event(models={"stt": "asr-second"}, tools=tools, instructions="changed"),
        routing=destination,
    )

    if supports_tools:
        assert error is None
        updated = service.build_session_updated(sid).session.model_dump()
        assert updated["models"]["stt"]["model"] == "asr-second"
        assert updated["tools"][0]["name"] == "look"
        assert updated["instructions"] == "changed"
    else:
        assert error is not None
        assert "does not support the session's tools" in error.error.message
        assert service._state(sid).runtime_config is before
        assert before.session.model_dump() == before_session


def test_managed_voice_update_fails_before_changing_any_session_setting():
    from speech_to_speech.api.openai_realtime.session_routing import TranscriptionRoute

    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    sid = service.register(routing=route())
    cfg = service._state(sid).runtime_config
    before = cfg.session.model_dump()

    error = service.handle_session_update(
        sid, event(audio={"output": {"voice": "missing"}}, instructions="must not apply")
    )
    assert error is not None
    assert "does not support this voice" in error.error.message
    assert cfg.session.model_dump() == before

    assert service.handle_session_update(sid, event(audio={"output": {"voice": "alloy"}})) is None
    assert cfg.session.audio.output.voice == "alloy"
    assert service.handle_session_update(sid, event(audio={"output": {"voice": "aiden"}})) is None
    assert cfg.session.audio.output.voice == "aiden"

    selected = route()
    selected = selected.model_copy(
        update={
            "routes": selected.routes.model_copy(
                update={"stt": TranscriptionRoute(model="asr-second", provider="hf", protocol="transcriptions")}
            )
        }
    )
    assert service.handle_session_update(sid, event(models={"stt": "asr-second"}), routing=selected) is None

    plain = service.register()
    assert service.handle_session_update(plain, event(audio={"output": {"voice": "missing"}})) is None
    assert service._state(plain).runtime_config.session.audio.output.voice == "missing"

    fixed_route = service.register(routing=route().model_copy(update={"updates_enabled": False}))
    assert service.handle_session_update(fixed_route, event(audio={"output": {"voice": "missing"}})) is None
    assert service._state(fixed_route).runtime_config.session.audio.output.voice == "missing"


def test_llm_switch_keeps_history_and_drops_what_the_destination_cannot_use():
    from openai.types.realtime import RealtimeConversationItemFunctionCall, RealtimeConversationItemFunctionCallOutput
    from openai.types.responses import ResponseReasoningItem

    from speech_to_speech.api.openai_realtime.session_routing import TranscriptionRoute
    from speech_to_speech.LLM.chat import AUDIO_INPUT_HISTORY_PLACEHOLDER, UserContent, make_user_audio_message

    image_message = make_user_message("Look at this.")
    image_message.content.append(UserContent(type="input_image", image_url="https://example.com/image.png"))
    initial = route(images=False, audio=False)
    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    sid = service.register(routing=initial)
    chat = service._state(sid).runtime_config.chat
    for item in (
        image_message,
        make_user_audio_message("AA=="),
        ResponseReasoningItem(id="rs_first", type="reasoning", summary=[], encrypted_content="first-provider"),
        RealtimeConversationItemFunctionCall(type="function_call", name="look", call_id="call_done", arguments="{}"),
        RealtimeConversationItemFunctionCallOutput(type="function_call_output", call_id="call_done", output="result"),
    ):
        chat.add_item(item)
    retained = [item.model_copy(deep=True) for item in chat.buffer]

    # A speech-only switch leaves the LLM's retained history untouched.
    stt_only = initial.model_copy(
        update={
            "routes": initial.routes.model_copy(
                update={"stt": TranscriptionRoute(model="asr-second", provider="hf", protocol="transcriptions")}
            )
        }
    )
    assert service.handle_session_update(sid, event(models={"stt": "asr-second"}), routing=stt_only) is None
    assert chat.buffer == retained

    selected = route("second", images=False, audio=False, tools=False)
    assert service.handle_session_update(sid, event(model="second"), routing=selected) is None
    cfg = service._state(sid).runtime_config
    assert cfg.chat is chat
    assert cfg.routing.routes.llm.model == "second"
    assert not any(isinstance(item, ResponseReasoningItem) for item in chat.buffer)
    parts = [part for item in chat.buffer if item.type == "message" for part in item.content]
    assert [(part.type, part.text) for part in parts] == [
        ("input_text", "Look at this."),
        ("input_text", AUDIO_INPUT_HISTORY_PLACEHOLDER),
    ]
    assert [item.type for item in chat.buffer if item.type != "message"] == ["function_call", "function_call_output"]
    assert "call_done" in str(chat.to_transformers_chat())


def test_tts_switch_keeps_a_supported_voice_and_otherwise_uses_the_route_default():
    from speech_to_speech.api.openai_realtime.session_routing import SpeechRoute

    for voices, expected in ((["aiden", "alloy"], "aiden"), ([], "alloy")):
        initial = route()
        selected = initial.model_copy(
            update={
                "routes": initial.routes.model_copy(
                    update={
                        "tts": SpeechRoute(
                            model="tts-alternative", provider="hf", protocol="speech", voice="alloy", voices=voices
                        )
                    }
                )
            }
        )
        service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
        sid = service.register(routing=initial)
        assert service.build_session_updated(sid).session.audio.output.voice == "aiden"
        assert service.handle_session_update(sid, event(models={"tts": "tts-alternative"}), routing=selected) is None
        assert service.build_session_updated(sid).session.audio.output.voice == expected


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


def send_switch(ws, selected, *, update_id="update", receive=True, **session):
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
    return receive_settlement(ws) if receive else None


def receive_settlement(ws):
    """Skip other session events until a routed update is acknowledged."""
    while True:
        received = ws.receive_json()
        if "_session_routing" in received:
            return received


@pytest.mark.parametrize("changed_stages", [("stt",), ("llm",), ("tts",), ("stt", "llm", "tts")])
def test_websocket_switch_routes_next_spoken_turn_through_selected_models(running_unit, monkeypatch, changed_stages):
    import numpy as np
    from starlette.testclient import TestClient

    from speech_to_speech.api.openai_realtime.websocket_router import create_app
    from speech_to_speech.pipeline.messages import VADAudio
    from speech_to_speech.STT.openai_compatible_handler import HttpTranscriptionOperation, HttpTranscriptionResult
    from speech_to_speech.TTS.openai_compatible_handler import HttpSpeechOperation

    unit, stop, create = running_unit
    transcriptions, speech_requests = [], []

    def transcribe(operation, cancel_check):
        transcriptions.append(operation)
        return HttpTranscriptionResult(
            text="Remember the blue bicycle." if len(transcriptions) == 1 else "Which color?"
        )

    def speak(operation, cancel_check):
        speech_requests.append(operation)
        return iter([np.zeros(2400, dtype="<i2").tobytes()])

    monkeypatch.setattr(HttpTranscriptionOperation, "run", transcribe)
    monkeypatch.setattr(HttpSpeechOperation, "iter_bytes", speak)

    initial = speech_route("first")
    destination = speech_route("second")
    selected = initial.model_copy(
        update={
            "routes": initial.routes.model_copy(
                update={stage: getattr(destination.routes, stage) for stage in changed_stages}
            )
        }
    )
    with TestClient(create_app([unit], stop, session_routing_enabled=True)) as client:
        with client.websocket_connect(
            "/v1/realtime", headers={"X-Speech-Session-Routing": initial.model_dump_json()}
        ) as ws:
            created = ws.receive_json()
            sid = created["session"]["id"]
            assert created["session"]["models"] == initial.models()
            for turn in (0, 1):
                if turn:
                    if changed_stages == ("llm",):
                        updated = send_switch(ws, selected, model="second")
                    else:
                        updated = send_switch(
                            ws,
                            selected,
                            models={
                                stage: {
                                    "model": getattr(selected.routes, stage).model,
                                    "provider": getattr(selected.routes, stage).provider,
                                }
                                for stage in changed_stages
                            },
                        )
                    assert updated["type"] == "session.updated", updated
                    assert updated["session"]["id"] == sid
                    assert updated["session"]["models"] == selected.models()
                    assert updated["session"]["model"] == selected.routes.llm.model
                cfg = unit.service._state(sid).runtime_config
                unit.input_queue.put(VADAudio(audio=np.zeros(160, dtype=np.float32), runtime_config=cfg))
                events = []
                while not events or events[-1]["type"] != "response.done":
                    events.append(ws.receive_json())
                assert events[-1]["response"]["status"] == "completed"
                assert any(event["type"] == "conversation.item.input_audio_transcription.completed" for event in events)
                assert any(event["type"] == "response.output_audio.delta" for event in events)
                assert len(transcriptions) == len(speech_requests) == turn + 1
                expected = {
                    stage: "second" if turn and stage in changed_stages else "first" for stage in ("stt", "llm", "tts")
                }
                assert transcriptions[-1].model == f"asr-{expected['stt']}"
                assert transcriptions[-1].extra_headers == {
                    "X-Speech-Provider": f"{expected['stt']}-provider",
                    "X-Speech-Session-Id": "allocated-session",
                }
                assert create.call_args.kwargs["model"] == expected["llm"]
                assert create.call_args.kwargs["extra_headers"] == {
                    "X-Speech-Provider": f"{expected['llm']}-provider",
                    "X-Speech-Session-Id": "allocated-session",
                }
                assert speech_requests[-1].payload["model"] == f"tts-{expected['tts']}"
                # Both TTS routes support the session's voice, so a switch keeps it.
                assert speech_requests[-1].payload["voice"] == "aiden"
                assert speech_requests[-1].extra_headers == {
                    "X-Speech-Provider": f"{expected['tts']}-provider",
                    "X-Speech-Session-Id": "allocated-session",
                }
                if turn:
                    assert "blue bicycle" in str(create.call_args.kwargs["messages"])


def test_switch_during_cancelled_generation_applies_after_cleanup(running_unit):
    from time import sleep

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
                sid = ws.receive_json()["session"]["id"]
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
                send_switch(ws, route("second"), receive=False, model="second", instructions="applies later")
                sleep(0.3)
                assert unit.service._state(sid).runtime_config.session.model == "first"

                release.set()
                updated = receive_settlement(ws)
                assert updated["type"] == "session.updated", updated
                assert updated["session"]["model"] == "second"
                assert updated["session"]["instructions"] == "applies later"
    finally:
        release.set()


def test_switch_during_cancelled_speech_synthesis_applies_after_cleanup(running_unit, monkeypatch):
    from time import sleep

    import numpy as np
    from starlette.testclient import TestClient

    from speech_to_speech.api.openai_realtime.websocket_router import create_app
    from speech_to_speech.pipeline.messages import VADAudio
    from speech_to_speech.STT.openai_compatible_handler import HttpTranscriptionOperation, HttpTranscriptionResult
    from speech_to_speech.TTS.openai_compatible_handler import HttpSpeechOperation

    unit, stop, _ = running_unit
    entered, release = Event(), Event()
    destination = speech_route("second")
    monkeypatch.setattr(
        HttpTranscriptionOperation, "run", lambda operation, cancel_check: HttpTranscriptionResult(text="Hello")
    )

    def blocked_speech(operation, cancel_check):
        entered.set()
        assert release.wait(5)
        return iter([np.zeros(2400, dtype="<i2").tobytes()])

    monkeypatch.setattr(HttpSpeechOperation, "iter_bytes", blocked_speech)
    try:
        with TestClient(create_app([unit], stop, session_routing_enabled=True)) as client:
            with client.websocket_connect(
                "/v1/realtime", headers={"X-Speech-Session-Routing": speech_route("first").model_dump_json()}
            ) as ws:
                sid = ws.receive_json()["session"]["id"]
                before = unit.service._state(sid).runtime_config
                unit.input_queue.put(VADAudio(audio=np.zeros(160, dtype=np.float32), runtime_config=before))
                assert entered.wait(1)
                ws.send_json({"type": "response.cancel"})
                send_switch(ws, destination, receive=False, models=destination.models())
                sleep(0.3)
                assert unit.service._state(sid).runtime_config is before

                release.set()
                updated = receive_settlement(ws)
                assert updated["type"] == "session.updated", updated
                assert updated["session"]["models"] == destination.models()
    finally:
        release.set()


def test_switch_during_background_final_stt_applies_after_the_turn(running_unit, monkeypatch):
    from time import sleep

    import numpy as np
    from starlette.testclient import TestClient

    from speech_to_speech.api.openai_realtime.websocket_router import create_app
    from speech_to_speech.pipeline.messages import VADAudio
    from speech_to_speech.STT.openai_compatible_handler import HttpTranscriptionResult
    from speech_to_speech.TTS.openai_compatible_handler import HttpSpeechOperation

    unit, stop, create = running_unit
    entered, release = Event(), Event()

    class BlockingTranscription:
        def run(self, cancel_check):
            entered.set()
            assert release.wait(5)
            return HttpTranscriptionResult(text="Hello")

        def cancel(self, reason):
            release.set()

    monkeypatch.setattr(unit.handlers[0], "_make_operation", lambda *args, **kwargs: BlockingTranscription())
    monkeypatch.setattr(
        HttpSpeechOperation, "iter_bytes", lambda operation, cancel_check: iter([np.zeros(2400, dtype="<i2").tobytes()])
    )
    try:
        with TestClient(create_app([unit], stop, session_routing_enabled=True)) as client:
            with client.websocket_connect(
                "/v1/realtime", headers={"X-Speech-Session-Routing": route().model_dump_json()}
            ) as ws:
                sid = ws.receive_json()["session"]["id"]
                before = unit.service._state(sid).runtime_config
                unit.input_queue.put(VADAudio(audio=np.zeros(160, dtype=np.float32), runtime_config=before))
                assert entered.wait(1)
                send_switch(ws, route("second"), receive=False, model="second")
                sleep(0.3)
                assert unit.service._state(sid).runtime_config is before

                # The spoken turn completes with the old routes; the switch follows it.
                release.set()
                updated = receive_settlement(ws)
                assert updated["type"] == "session.updated", updated
                assert updated["session"]["model"] == "second"
                assert create.call_args.kwargs["model"] == "first"
    finally:
        release.set()


def test_pending_switch_waits_while_the_client_resolves_tool_calls(running_unit):
    from time import sleep

    from openai.types.realtime import RealtimeConversationItemFunctionCall
    from starlette.testclient import TestClient

    from speech_to_speech.api.openai_realtime.websocket_router import create_app

    unit, stop, _ = running_unit
    with TestClient(create_app([unit], stop, session_routing_enabled=True)) as client:
        with client.websocket_connect(
            "/v1/realtime", headers={"X-Speech-Session-Routing": route().model_dump_json()}
        ) as ws:
            sid = ws.receive_json()["session"]["id"]
            before = unit.service._state(sid).runtime_config
            before.chat.add_item(
                RealtimeConversationItemFunctionCall(
                    type="function_call", name="look", call_id="call_pending", arguments="{}"
                )
            )
            send_switch(ws, route("second"), receive=False, model="second")
            sleep(0.3)
            assert unit.service._state(sid).runtime_config is before

            # Client events keep flowing while the switch waits.
            ws.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "function_call_output", "call_id": "call_pending", "output": "done"},
                }
            )
            updated = receive_settlement(ws)
            assert updated["type"] == "session.updated", updated
            assert updated["session"]["model"] == "second"


def busy_unit(running_unit):
    from speech_to_speech.baseHandler import BaseHandler

    class BackgroundHandler(BaseHandler):
        busy = True

        def has_pending_session_work(self):
            return self.busy

    unit, stop, _ = running_unit
    handler = BackgroundHandler(stop, Queue(), Queue())
    unit.handlers.append(handler)
    return unit, stop, handler


def test_switch_waits_for_handler_reported_work(running_unit):
    from time import sleep

    from starlette.testclient import TestClient

    from speech_to_speech.api.openai_realtime.websocket_router import create_app

    unit, stop, handler = busy_unit(running_unit)
    with TestClient(create_app([unit], stop, session_routing_enabled=True)) as client:
        with client.websocket_connect(
            "/v1/realtime", headers={"X-Speech-Session-Routing": route().model_dump_json()}
        ) as ws:
            ws.receive_json()
            before = unit.service._state(unit.session.session_id).runtime_config
            send_switch(ws, route("second"), receive=False, model="second")
            sleep(0.3)
            assert unit.service._state(unit.session.session_id).runtime_config is before
            handler.busy = False
            assert receive_settlement(ws)["type"] == "session.updated"


def test_later_switch_supersedes_a_pending_one(running_unit):
    from starlette.testclient import TestClient

    from speech_to_speech.api.openai_realtime.websocket_router import create_app

    unit, stop, handler = busy_unit(running_unit)
    with TestClient(create_app([unit], stop, session_routing_enabled=True)) as client:
        with client.websocket_connect(
            "/v1/realtime", headers={"X-Speech-Session-Routing": route().model_dump_json()}
        ) as ws:
            ws.receive_json()
            send_switch(ws, route("second"), update_id="earlier", receive=False, model="second")
            superseded = send_switch(ws, route("third"), update_id="later", model="third")
            assert superseded["type"] == "error"
            assert superseded["_session_routing"] == "earlier"
            assert superseded["error"]["event_id"] == "client-update"
            handler.busy = False
            updated = receive_settlement(ws)
            assert updated["_session_routing"] == "later"
            assert updated["session"]["model"] == "third"


def test_pending_switch_times_out_without_changes(running_unit, monkeypatch):
    from starlette.testclient import TestClient

    from speech_to_speech.api.openai_realtime import websocket_router

    monkeypatch.setattr(websocket_router, "ROUTING_UPDATE_TIMEOUT_S", 0.2)
    unit, stop, _ = busy_unit(running_unit)
    with TestClient(websocket_router.create_app([unit], stop, session_routing_enabled=True)) as client:
        with client.websocket_connect(
            "/v1/realtime", headers={"X-Speech-Session-Routing": route().model_dump_json()}
        ) as ws:
            ws.receive_json()
            before = unit.service._state(unit.session.session_id).runtime_config
            rejected = send_switch(ws, route("second"), model="second", instructions="must not apply")
            assert rejected["type"] == "error"
            assert rejected["error"]["event_id"] == "client-update"
            assert "did not finish in time" in rejected["error"]["message"]
            assert unit.service._state(unit.session.session_id).runtime_config is before
