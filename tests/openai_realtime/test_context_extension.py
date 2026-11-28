import asyncio
import json
from contextlib import asynccontextmanager
from threading import Event
from time import perf_counter
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.context_extension import (
    ContextDispatcher,
    ContextIntegration,
    configured_integration,
    render_records,
)
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.pipeline.events import TranscriptionCompletedEvent
from speech_to_speech.pipeline.messages import GenerateResponseRequest
from tests.openai_realtime.conftest import make_pipeline_unit


class Recorder:
    def __init__(self):
        self.requests = []

    async def prepare_context(self, query):
        self.requests.append(query)
        return ("amber",)


def test_host_recall_default_and_ceiling():
    for constructor in (ContextIntegration, ContextDispatcher):
        assert constructor(Recorder()).wait_seconds == 0.4
        for requested, effective in ((0, 0), (0.15, 0.15), (0.4, 0.4), (1, 0.4), (5, 0.4)):
            assert constructor(Recorder(), wait_seconds=requested).wait_seconds == effective
        for invalid in (-1, float("nan"), float("inf"), True, "5"):
            with pytest.raises(ValueError, match="budget"):
                constructor(Recorder(), wait_seconds=invalid)


def test_ws_manual_request_scoped_copy():
    unit = make_pipeline_unit()
    provider = Recorder()
    lifecycle = []

    @asynccontextmanager
    async def open_hooks():
        lifecycle.append(asyncio.get_running_loop())
        yield provider
        lifecycle.append(asyncio.get_running_loop())

    app = create_app([unit], Event(), context_integration=ContextIntegration(open_hooks))
    with TestClient(app) as client, client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "response.create",
                "response": {
                    "input": [
                        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "espectros"}]}
                    ]
                },
            }
        )
        assert ws.receive_json()["type"] == "response.created"
        request = unit.text_prompt_queue.get(timeout=2)
        assert [q.text for q in provider.requests] == ["espectros"]
        view = request.runtime_config.chat.copy(deep=True)
        request.prepared_context.project(view)
        assert "amber" in view.to_transformers_chat()[-1]["content"]
        assert "amber" not in str(request.runtime_config.chat.to_responses_api_chat())
        # Same text, same item ID, new host mutation generation.
        request.runtime_config.chat.replace_user_message_text(view.buffer[-1].id, "espectros")
        view = request.runtime_config.chat.copy(deep=True)
        request.prepared_context.project(view)
        assert "amber" not in str(view.to_responses_api_chat())
    assert len(lifecycle) == 2 and lifecycle[0] is lifecycle[1]


@pytest.mark.asyncio
async def test_late_audio_lookup_proceeds_once_and_holds_capacity(service, conn_id, text_prompt_queue):
    release = asyncio.Event()

    class Slow:
        async def prepare_context(self, query):
            await release.wait()
            return ("late",)

    dispatcher = ContextDispatcher(Slow(), wait_seconds=0.005)
    service.context_dispatcher = dispatcher
    service.dispatch_pipeline_event(conn_id, TranscriptionCompletedEvent(transcript="espectros", language_code="es"))
    await asyncio.sleep(0.03)
    assert text_prompt_queue.get_nowait().prepared_context is None
    assert conn_id in dispatcher.active
    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not dispatcher.active and text_prompt_queue.empty()
    await dispatcher.close()


@pytest.mark.asyncio
async def test_disconnected_lookup_never_reaches_reused_slot(service, text_prompt_queue):
    session = service.register()
    dispatcher = ContextDispatcher(Recorder())
    service.context_dispatcher = dispatcher
    service.dispatch_pipeline_event(session, TranscriptionCompletedEvent(transcript="hello"))
    service.unregister(session)
    second = service.register()
    await asyncio.sleep(0.02)
    assert text_prompt_queue.empty()
    service.unregister(second)
    await dispatcher.close()


@pytest.mark.asyncio
async def test_overlapping_transcripts_preserve_model_queue_order(service, conn_id, text_prompt_queue):
    release = asyncio.Event()
    received = []

    class Slow:
        async def prepare_context(self, query):
            received.append(query)
            await release.wait()
            return ("late",)

    dispatcher = ContextDispatcher(Slow(), wait_seconds=1)
    service.context_dispatcher = dispatcher
    service.dispatch_pipeline_event(conn_id, TranscriptionCompletedEvent(transcript="first"))
    await asyncio.sleep(0)
    key = received[0].response_key
    service.dispatch_pipeline_event(conn_id, TranscriptionCompletedEvent(transcript="second"))
    first, second = text_prompt_queue.get_nowait(), text_prompt_queue.get_nowait()
    assert first.response_key == key != second.response_key
    assert first.prepared_context is second.prepared_context is None
    release.set()
    await asyncio.sleep(0)
    await dispatcher.close()
    assert text_prompt_queue.empty() and len(received) == 1


def test_record_budget_includes_framing():
    assert render_records(("x" * 1024,), 1024) is None
    assert render_records(("a",), 20) == 'Reference data ["a"]'
    assert render_records("wrong shape", 1024) is None


@pytest.mark.parametrize(
    "record", [[], {"format": 1}, {"format": 1, "distribution": "sample", "entry": "v1", "config": 3}]
)
def test_invalid_binding_reports_configuration_error(tmp_path, monkeypatch, record):
    path = tmp_path / "binding.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setenv("S2S_CONTEXT_BINDING", str(path))
    with pytest.raises(ValueError, match="context binding"):
        configured_integration()


def pending_text_request(text):
    chat = Chat(4)
    chat.add_item(make_user_message(text))
    return GenerateResponseRequest(runtime_config=RuntimeConfig(chat=chat))


@pytest.mark.asyncio
async def test_five_second_extension_cannot_extend_recall_window():
    deadline = asyncio.get_running_loop().create_future()
    release = asyncio.Event()

    class Slow:
        async def prepare_context(self, query):
            deadline.set_result(query.deadline)
            await release.wait()
            return ("late record",)

    dispatcher = ContextDispatcher(Slow(), wait_seconds=5)
    queue = asyncio.Queue()
    request = pending_text_request("coffee")
    try:
        before = perf_counter()
        dispatcher.submit("session", request, SimpleNamespace(put=queue.put_nowait), None, lambda: True)
        after = perf_counter()
        cutoff = await asyncio.wait_for(deadline, 1)
        assert before + 0.4 <= cutoff <= after + 0.4
        assert await asyncio.wait_for(queue.get(), 2) is request
        assert perf_counter() >= cutoff
        assert request.prepared_context is None
        release.set()
        await dispatcher.active["session"]
        assert request.prepared_context is None and queue.empty()
    finally:
        await dispatcher.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["exception", "malformed", "empty"])
async def test_provider_errors_and_misses_submit_one_plain_model_request(outcome):
    class Provider:
        async def prepare_context(self, query):
            if outcome == "exception":
                raise RuntimeError("offline")
            return ["invalid record container"] if outcome == "malformed" else ()

    queue = asyncio.Queue()
    dispatcher = ContextDispatcher(Provider(), wait_seconds=1)
    request = pending_text_request("question")
    try:
        dispatcher.submit("session", request, SimpleNamespace(put=queue.put_nowait), None, lambda: True)
        assert await asyncio.wait_for(queue.get(), 2) is request
        assert request.prepared_context is None
        assert queue.empty() and not dispatcher.active
    finally:
        await dispatcher.close()


@pytest.mark.asyncio
async def test_global_capacity_bypasses_recall_and_recovers_after_work_finishes():
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    class Provider:
        async def prepare_context(self, query):
            calls.append(query.session_id)
            started.set()
            await release.wait()
            return ("record",)

    queue = asyncio.Queue()
    output = SimpleNamespace(put=queue.put_nowait)
    dispatcher = ContextDispatcher(Provider(), wait_seconds=1, capacity=1)
    first, second, third = map(pending_text_request, ("first", "second", "third"))
    try:
        dispatcher.submit("one", first, output, None, lambda: True)
        await asyncio.wait_for(started.wait(), 2)
        dispatcher.submit("two", second, output, None, lambda: True)
        assert queue.get_nowait() is second and second.prepared_context is None
        assert calls == ["one"]

        release.set()
        assert await asyncio.wait_for(queue.get(), 2) is first
        assert first.prepared_context is not None
        dispatcher.submit("two", third, output, None, lambda: True)
        assert await asyncio.wait_for(queue.get(), 2) is third
        assert third.prepared_context is not None
        assert calls == ["one", "two"] and queue.empty()
    finally:
        await dispatcher.close()
