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
    assert service.build_session_updated(sid).session.audio.output.voice == "alloy"


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
                assert speech_requests[-1].payload["voice"] == ("aiden" if expected["tts"] == "first" else "alloy")
                assert speech_requests[-1].extra_headers == {
                    "X-Speech-Provider": f"{expected['tts']}-provider",
                    "X-Speech-Session-Id": "allocated-session",
                }
                if turn:
                    assert "blue bicycle" in str(create.call_args.kwargs["messages"])


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


def test_cancelled_speech_synthesis_must_drain_before_switching(running_unit, monkeypatch):
    from time import monotonic, sleep

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
                assert ws.receive_json()["type"] == "conversation.item.input_audio_transcription.completed"
                ws.send_json({"type": "response.cancel"})
                rejected = send_switch(ws, destination, models=destination.models())
                assert rejected["type"] == "error", rejected
                assert rejected["error"]["event_id"] == "client-update"
                assert "Pipeline work is still draining" in rejected["error"]["message"]
                assert unit.service._state(sid).runtime_config is before

                release.set()
                deadline = monotonic() + 2
                while unit.handlers[-1]._active_operation is not None and monotonic() < deadline:
                    sleep(0.01)
                assert unit.handlers[-1]._active_operation is None
                updated = send_switch(ws, destination, models=destination.models())
                assert updated["type"] == "session.updated", updated
                assert updated["session"]["models"] == destination.models()
    finally:
        release.set()


def test_background_final_stt_must_drain_before_switching(running_unit, monkeypatch):
    import numpy as np
    from starlette.testclient import TestClient

    from speech_to_speech.api.openai_realtime.websocket_router import create_app
    from speech_to_speech.pipeline.messages import VADAudio
    from speech_to_speech.STT.openai_compatible_handler import HttpTranscriptionResult

    unit, stop, _ = running_unit
    entered, release = Event(), Event()

    class BlockingTranscription:
        def run(self, cancel_check):
            entered.set()
            assert release.wait(5)
            return HttpTranscriptionResult(text="Hello")

        def cancel(self, reason):
            release.set()

    monkeypatch.setattr(unit.handlers[0], "_make_operation", lambda *args, **kwargs: BlockingTranscription())
    initial = route()
    try:
        with TestClient(create_app([unit], stop, session_routing_enabled=True)) as client:
            with client.websocket_connect(
                "/v1/realtime", headers={"X-Speech-Session-Routing": initial.model_dump_json()}
            ) as ws:
                ws.receive_json()
                before = unit.service._state(unit.session.session_id).runtime_config
                unit.input_queue.put(VADAudio(audio=np.zeros(160, dtype=np.float32), runtime_config=before))
                assert entered.wait(1)
                rejected = send_switch(ws, route("second"), model="second", instructions="must not apply")
                assert rejected["type"] == "error"
                assert rejected["error"]["event_id"] == "client-update"
                assert unit.service._state(unit.session.session_id).runtime_config is before
    finally:
        release.set()


def test_switch_asks_handler_to_report_pending_work(running_unit):
    from starlette.testclient import TestClient

    from speech_to_speech.api.openai_realtime.websocket_router import create_app
    from speech_to_speech.baseHandler import BaseHandler

    class BackgroundHandler(BaseHandler):
        busy = True

        def has_pending_session_work(self):
            return self.busy

    unit, stop, _ = running_unit
    handler = BackgroundHandler(stop, Queue(), Queue())
    unit.handlers.append(handler)
    with TestClient(create_app([unit], stop, session_routing_enabled=True)) as client:
        with client.websocket_connect(
            "/v1/realtime", headers={"X-Speech-Session-Routing": route().model_dump_json()}
        ) as ws:
            ws.receive_json()
            before = unit.service._state(unit.session.session_id).runtime_config
            rejected = send_switch(ws, route("second"), model="second")
            assert rejected["type"] == "error"
            assert unit.service._state(unit.session.session_id).runtime_config is before
            handler.busy = False
            assert send_switch(ws, route("second"), model="second")["type"] == "session.updated"
