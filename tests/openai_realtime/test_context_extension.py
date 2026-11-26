import asyncio
from contextlib import asynccontextmanager
from threading import Event

import pytest
from fastapi.testclient import TestClient

from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.context_extension import ContextDispatcher, ContextIntegration, render_records
from speech_to_speech.pipeline.events import TranscriptionCompletedEvent
from tests.openai_realtime.conftest import make_pipeline_unit


class Recorder:
    def __init__(self):
        self.requests = []

    async def prepare_context(self, query):
        self.requests.append(query)
        return ("amber",)


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
