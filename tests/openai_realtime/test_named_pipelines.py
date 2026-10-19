"""Named admission keeps capacity and session state in its selected pool."""

from queue import Queue
from threading import Event

import pytest
from starlette.testclient import TestClient

from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.pipeline.cancel_scope import CancelScope


def make_unit(index):
    prompts = Queue()
    listening = Event()
    return PipelineUnit(
        index=index,
        service=RealtimeService(text_prompt_queue=prompts, should_listen=listening),
        cancel_scope=CancelScope(),
        should_listen=listening,
        response_playing=Event(),
        input_queue=Queue(),
        output_queue=Queue(),
        text_output_queue=Queue(),
        text_prompt_queue=prompts,
        handlers=[],
    )


def test_named_capacity_and_selection():
    units = [make_unit(i) for i in range(3)]
    app = create_app(units, Event(), pipeline_pools={"alpha": units[:2], "beta": units[2:]})
    with TestClient(app) as client:
        for path, error in [
            ("/v1/realtime", "pipeline_selection_required"),
            ("/v1/pipelines/ALPHA/realtime", "unknown_pipeline"),
        ]:
            with client.websocket_connect(path) as rejected:
                assert rejected.receive_json()["error"]["type"] == error
        with client.websocket_connect("/v1/pipelines/beta/realtime") as beta:
            beta.receive_json()
            assert units[2].session is not None
            with client.websocket_connect("/v1/pipelines/beta/realtime") as rejected:
                assert rejected.receive_json()["error"]["type"] == "session_limit_reached"
            with client.websocket_connect("/v1/pipelines/alpha/realtime") as alpha:
                alpha.receive_json()
                assert units[0].session is not None
                summary = client.get("/v1/pool").json()
                assert summary["available"] == 1
                assert summary["pipelines"]["beta"]["in_use"] == 1
        usage = client.get("/v1/usage").json()
        assert "beta" in usage["pipelines"]


def test_named_mapping_must_match_aggregate():
    unit = make_unit(0)
    with pytest.raises(ValueError):
        create_app([unit], Event(), pipeline_pools={"alpha": []})


def test_single_pipeline_alias_and_shutdown_rejection():
    unit = make_unit(0)
    stop = Event()
    app = create_app([unit], stop, pipeline_pools={"alpha": [unit]})
    with TestClient(app) as client:
        stop.set()
        with client.websocket_connect("/v1/realtime") as rejected:
            assert rejected.receive_json()["error"]["type"] == "session_limit_reached"
        assert unit.session is None


def test_failed_replica_is_not_reused():
    units = [make_unit(i) for i in range(3)]
    units[0].mark_failed()
    app = create_app(units, Event(), pipeline_pools={"alpha": units[:2], "beta": units[2:]})
    with TestClient(app) as client:
        with client.websocket_connect("/v1/pipelines/alpha/realtime") as ws:
            ws.receive_json()
            assert units[1].session is not None
        units[1].mark_failed()
        with client.websocket_connect("/v1/pipelines/alpha/realtime") as rejected:
            assert rejected.receive_json()["error"]["type"] == "pipeline_unavailable"
        with client.websocket_connect("/v1/pipelines/beta/realtime") as ws:
            ws.receive_json()
        assert client.get("/v1/pool").json()["pipelines"]["alpha"]["failed"] == 2


def test_webrtc_named_location_and_hangup_owner(monkeypatch):
    import speech_to_speech.api.openai_realtime.websocket_router as router
    from speech_to_speech.api.openai_realtime.transports import SessionTransport

    class Peer(SessionTransport):
        kind = "webrtc"

        def __init__(self, pc, **callbacks):
            self.callbacks = callbacks

        def setup(self):
            pass

        async def negotiate(self, offer):
            return "answer"

        async def send_events(self, events):
            pass

        async def send_audio_chunk(self, *args):
            pass

        def discard_pending_audio(self):
            pass

        async def close(self):
            self.callbacks["on_closed"]()

    monkeypatch.setattr(router, "WEBRTC_AVAILABLE", True)
    monkeypatch.setattr(router, "RTCPeerConnection", lambda: None, raising=False)
    monkeypatch.setattr(router, "rtc_configuration_from_env", lambda: None, raising=False)
    monkeypatch.setattr(router, "WebRTCSession", Peer, raising=False)
    units = [make_unit(0), make_unit(1)]
    app = create_app(units, Event(), pipeline_pools={"alpha": units[:1], "beta": units[1:]})
    with TestClient(app) as client:
        headers = {"content-type": "application/sdp"}
        assert client.post("/v1/realtime/calls", content="offer", headers=headers).status_code == 400
        assert client.post("/v1/pipelines/missing/realtime/calls", content="offer", headers=headers).status_code == 404
        first = client.post("/v1/pipelines/alpha/realtime/calls?name=beta", content="offer", headers=headers)
        assert first.status_code == 201
        location = first.headers["location"]
        assert location.startswith("/v1/pipelines/alpha/realtime/calls/")
        assert client.delete(location.replace("alpha", "beta")).status_code == 404
        assert units[0].session.released_at is None
        assert client.post("/v1/pipelines/alpha/realtime/calls", content="offer", headers=headers).status_code == 503
        assert client.post("/v1/pipelines/beta/realtime/calls", content="offer", headers=headers).status_code == 201
        assert client.delete(location.replace("/pipelines/alpha", "") + "?name=beta").status_code == 200


def test_unknown_and_ambiguous_proxy_requests_have_no_usage():
    units = [make_unit(0), make_unit(1)]
    app = create_app(units, Event(), pipeline_pools={"alpha": units[:1], "beta": units[1:]})
    with TestClient(app) as client:
        for suffix in ("responses", "chat/completions"):
            assert client.post(f"/v1/{suffix}", json={}).json()["error"]["type"] == "pipeline_selection_required"
            assert client.post(f"/v1/pipelines/missing/{suffix}", json={}).status_code == 404
            assert client.post(f"/v1/pipelines/alpha/{suffix}", json={}).status_code == 501
        usage = client.get("/v1/usage").json()
        assert usage["llm_proxy"]["requests"] == 0
        assert usage["pipelines"]["alpha"]["llm_proxy"]["requests"] == 0


@pytest.mark.asyncio
async def test_late_release_cannot_clear_another_session():
    from speech_to_speech.api.openai_realtime.pipeline_unit import SessionState
    from speech_to_speech.api.openai_realtime.websocket_router import _release_unit_after_drain

    unit = make_unit(0)
    old_id = unit.service.register()
    old = SessionState(session_id=old_id)
    old.drained.set()
    current = SessionState(session_id=unit.service.register())
    unit.session = current
    await _release_unit_after_drain(unit, old, old_id)
    assert unit.session is current


def test_worker_failure_closes_only_its_session():
    from starlette.websockets import WebSocketDisconnect

    units = [make_unit(0), make_unit(1)]
    app = create_app(units, Event(), pipeline_pools={"alpha": units[:1], "beta": units[1:]})
    with TestClient(app) as client:
        with client.websocket_connect("/v1/pipelines/alpha/realtime") as alpha:
            alpha.receive_json()
            with client.websocket_connect("/v1/pipelines/beta/realtime") as beta:
                beta.receive_json()
                units[0].mark_failed()
                with pytest.raises(WebSocketDisconnect):
                    alpha.receive_json()
                beta.send_json({"type": "session.update", "session": {"type": "realtime", "instructions": "beta"}})
                assert beta.receive_json()["type"] == "session.updated"


@pytest.mark.asyncio
async def test_named_loopback_sessions_keep_instance_affinity():
    from contextlib import AsyncExitStack

    from openai import AsyncOpenAI

    from .test_openai_client import _recv, _ServerEnv

    units = [make_unit(i) for i in range(3)]
    env = _ServerEnv()
    env.app = create_app(units, env.stop_event, pipeline_pools={"alpha": units[:2], "beta": units[2:]})
    env.start()
    try:
        async with AsyncExitStack() as stack:
            connections = []
            for name in ("alpha", "alpha", "beta"):
                client = await stack.enter_async_context(
                    AsyncOpenAI(api_key="test", websocket_base_url=f"ws://127.0.0.1:{env.port}/v1/pipelines/{name}")
                )
                connection = await stack.enter_async_context(client.realtime.connect(model="identical-model"))
                connections.append(connection)
                assert (await _recv(connection)).type == "session.created"
            for index, connection in enumerate(connections):
                await connection.send(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "instructions": f"instance {index}",
                            "audio": {
                                "input": {"transcription": {"model": "same", "language": ("en", "fr", "de")[index]}},
                                "output": {"voice": ("alloy", "echo", "shimmer")[index]},
                            },
                            "tools": [{"type": "function", "name": f"tool_{index}"}],
                        },
                    }
                )
                assert (await _recv(connection)).type == "session.updated"
                session_id = units[index].session.session_id
                configured = units[index].service._state(session_id).runtime_config.session
                assert configured.instructions == f"instance {index}"
                assert configured.audio.output.voice == ("alloy", "echo", "shimmer")[index]
                assert configured.audio.input.transcription.language == ("en", "fr", "de")[index]
                assert configured.tools[0].name == f"tool_{index}"
                await connection.send(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": f"history {index}"}],
                        },
                    }
                )
                assert (await _recv(connection)).item.content[0].text == f"history {index}"
                assert all(
                    other.service._state(other.session.session_id).runtime_config.session.instructions
                    == f"instance {position}"
                    for position, other in enumerate(units[: index + 1])
                )
            from speech_to_speech.pipeline.events import AssistantOutputEvent
            from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, AssistantToolCallPart

            for unit, connection in zip(units, connections, strict=True):
                unit.output_queue.put(b"\x00" * 512)
                events = [await _recv(connection) for _ in range(4)]
                assert events[-1].type == "response.output_audio.delta"
            untouched_generations = [unit.cancel_scope.generation for unit in units[1:]]
            await connections[0].send({"type": "response.cancel"})
            cancelled = [await _recv(connections[0]) for _ in range(4)]
            assert cancelled[-1].response.status == "cancelled"
            assert [unit.cancel_scope.generation for unit in units[1:]] == untouched_generations
            for index in (1, 2):
                units[index].output_queue.put(AssistantOutputEvent(text=f"output {index}"))
                units[index].output_queue.put(
                    AssistantOutputEvent(
                        parts=[
                            AssistantToolCallPart(
                                tool={
                                    "type": "function_call",
                                    "call_id": f"call_{index}",
                                    "name": f"tool_{index}",
                                    "arguments": "{}",
                                }
                            )
                        ]
                    )
                )
                units[index].output_queue.put(AUDIO_RESPONSE_DONE)
                texts = []
                tools = []
                while True:
                    event = await _recv(connections[index])
                    if event.type == "response.output_audio_transcript.delta":
                        texts.append(event.delta)
                    if event.type == "response.function_call_arguments.done":
                        tools.append(event.name)
                    if event.type == "response.done":
                        assert event.response.status == "completed"
                        break
                assert texts == [f"output {index}"]
                assert tools == [f"tool_{index}"]
            client = await stack.enter_async_context(
                AsyncOpenAI(api_key="test", websocket_base_url=f"ws://127.0.0.1:{env.port}/v1/pipelines/alpha")
            )
            rejected = await stack.enter_async_context(client.realtime.connect(model="identical-model"))
            assert (await _recv(rejected)).error.type == "session_limit_reached"
    finally:
        env.stop()


@pytest.mark.asyncio
async def test_named_webrtc_loopback_capacity_affinity_and_reuse():
    import asyncio
    import json

    import httpx

    aiortc = pytest.importorskip("aiortc")
    from speech_to_speech.pipeline.control import SESSION_END, PipelineControlMessage

    from .test_openai_client import _ServerEnv
    from .test_webrtc import _DataChannelInbox

    units = [make_unit(i) for i in range(3)]
    env = _ServerEnv()
    env.app = create_app(units, env.stop_event, pipeline_pools={"alpha": units[:2], "beta": units[2:]})
    env.start()
    peers = []
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{env.port}", timeout=20) as client:
            calls = []
            for index, name in enumerate(("alpha", "alpha", "beta")):
                peer = aiortc.RTCPeerConnection()
                peers.append(peer)
                channel = peer.createDataChannel("oai-events")
                peer.addTrack(aiortc.AudioStreamTrack())
                inbox = _DataChannelInbox(channel)
                await peer.setLocalDescription(await peer.createOffer())
                response = await client.post(
                    f"/v1/pipelines/{name}/realtime/calls?model=identical-model",
                    content=peer.localDescription.sdp,
                    headers={"content-type": "application/sdp"},
                )
                assert response.status_code == 201
                calls.append(response.headers["location"])
                await peer.setRemoteDescription(aiortc.RTCSessionDescription(sdp=response.text, type="answer"))
                created = await inbox.wait_for("session.created", timeout=10)
                assert created["session"]["id"] == units[index].session.session_id
                channel.send(
                    json.dumps(
                        {"type": "session.update", "session": {"type": "realtime", "instructions": f"rtc {index}"}}
                    )
                )
                updated = await inbox.wait_for("session.updated")
                assert updated["session"]["instructions"] == f"rtc {index}"
            rejected = await client.post(
                "/v1/pipelines/alpha/realtime/calls",
                content=peers[0].localDescription.sdp,
                headers={"content-type": "application/sdp"},
            )
            assert rejected.json()["error"]["type"] == "session_limit_reached"
            assert (await client.delete(calls[0].replace("alpha", "beta"))).status_code == 404
            assert (await client.delete(calls[0])).status_code == 200
            old_id = calls[0].rsplit("/", 1)[1]
            units[0].output_queue.put(PipelineControlMessage(SESSION_END.kind, session_id=old_id))
            for _ in range(100):
                if units[0].session is None:
                    break
                await asyncio.sleep(0.01)
            assert units[0].session is None
            assert units[1].service._state(units[1].session.session_id).runtime_config.session.instructions == "rtc 1"
            assert units[2].service._state(units[2].session.session_id).runtime_config.session.instructions == "rtc 2"
            peer = aiortc.RTCPeerConnection()
            peers.append(peer)
            peer.createDataChannel("oai-events")
            peer.addTrack(aiortc.AudioStreamTrack())
            await peer.setLocalDescription(await peer.createOffer())
            reused = await client.post(
                "/v1/pipelines/alpha/realtime/calls",
                content=peer.localDescription.sdp,
                headers={"content-type": "application/sdp"},
            )
            assert reused.status_code == 201
            assert units[0].session.session_id != old_id
            assert units[0].service._state(units[0].session.session_id).runtime_config.session.instructions != "rtc 0"
    finally:
        await asyncio.gather(*(peer.close() for peer in peers))
        env.stop()


@pytest.mark.parametrize("name", ["", 1, "a/b", "{name}", "a" * 65])
def test_named_routes_require_config_names(name):
    unit = make_unit(0)
    with pytest.raises(ValueError):
        create_app([unit], Event(), pipeline_pools={name: [unit]})


def test_named_websocket_send_failure_disables_its_unit(monkeypatch, caplog):
    from fastapi import WebSocket
    from starlette.websockets import WebSocketDisconnect

    original = WebSocket.send_json

    async def broken_send(self, data, *args, **kwargs):
        if data["type"] == "response.created":
            raise OSError("private-transport-marker")
        await original(self, data, *args, **kwargs)

    monkeypatch.setattr(WebSocket, "send_json", broken_send)
    units = [make_unit(0), make_unit(1)]
    app = create_app(units, Event(), pipeline_pools={"alpha": units[:1], "beta": units[1:]})
    with TestClient(app) as client:
        with client.websocket_connect("/v1/pipelines/alpha/realtime") as ws:
            ws.receive_json()
            units[0].output_queue.put(b"\x00" * 512)
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()
        assert units[0].lifecycle_state == "failed"
        assert units[1].healthy
        assert "private-transport-marker" not in caplog.text


def test_query_name_cannot_select_or_override_a_pipeline():
    units = [make_unit(0), make_unit(1)]
    app = create_app(units, Event(), pipeline_pools={"alpha": units[:1], "beta": units[1:]})
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?name=alpha") as rejected:
            assert rejected.receive_json()["error"]["type"] == "pipeline_selection_required"
        with client.websocket_connect("/v1/pipelines/alpha/realtime?name=beta") as ws:
            ws.receive_json()
            assert units[0].session is not None
            assert units[1].session is None
        response = client.post(
            "/v1/realtime/calls?name=alpha", content="offer", headers={"content-type": "application/sdp"}
        )
        assert response.json()["error"]["type"] == "pipeline_selection_required"


@pytest.mark.asyncio
async def test_delayed_worker_start_keeps_the_output_loop_alive():
    import asyncio

    from openai import AsyncOpenAI

    from .test_openai_client import _recv, _ServerEnv

    unit = make_unit(0)
    unit.lifecycle_state = "constructed"
    unit.stop_event = Event()
    env = _ServerEnv()
    env.app = create_app([unit], env.stop_event, pipeline_pools={"alpha": [unit]})
    env.start()
    try:
        await asyncio.sleep(0.05)
        async with AsyncOpenAI(
            api_key="test", websocket_base_url=f"ws://127.0.0.1:{env.port}/v1/pipelines/alpha"
        ) as client:
            async with client.realtime.connect(model="same") as rejected:
                assert (await _recv(rejected)).error.type == "pipeline_unavailable"
            with unit.lock:
                unit.lifecycle_state = "ready"
            async with client.realtime.connect(model="same") as connection:
                assert (await _recv(connection)).type == "session.created"
                unit.output_queue.put(b"\x00" * 512)
                events = [await _recv(connection, timeout=0.5) for _ in range(4)]
                assert events[-1].type == "response.output_audio.delta"
    finally:
        env.stop()
