import asyncio
from contextlib import asynccontextmanager
from threading import Event
from time import monotonic, sleep

import pytest
from fastapi.testclient import TestClient

from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.context_extension import STATE_KEY, ContextIntegration, ContextScope
from tests.openai_realtime.conftest import make_pipeline_unit
from tests.openai_realtime.test_websocket_router import _simulate_session_end_drain


def scoped_app(unit, hooks, scope):
    def scope_middleware(app):
        async def middleware(asgi_scope, receive, send):
            if asgi_scope["type"] in {"http", "websocket"}:
                asgi_scope.setdefault("state", {})[STATE_KEY] = scope
            await app(asgi_scope, receive, send)

        return middleware

    integration = ContextIntegration(hooks, wait_seconds=1, scope_middleware=scope_middleware)
    return create_app([unit], Event(), context_integration=integration)


def test_release_with_inflight_recall_drains_revoked_lease():
    unit = make_pipeline_unit()
    lease = ContextScope("tenant", "issuer", "first", monotonic() + 30)
    started = Event()
    release = asyncio.Event()

    class SlowProvider:
        async def prepare_context(self, request):
            started.set()
            await release.wait()
            return ("private first owner",)

    @asynccontextmanager
    async def hooks():
        yield SlowProvider()

    with TestClient(scoped_app(unit, hooks, lease)) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()
            ws.send_json(
                {
                    "type": "response.create",
                    "response": {
                        "input": [
                            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}
                        ]
                    },
                }
            )
            assert ws.receive_json()["type"] == "response.created"
            assert started.wait(1)
        lease.revoked.set()
        _simulate_session_end_drain(unit.input_queue, unit.output_queue)
        client.portal.call(release.set)
        deadline = monotonic() + 2
        while unit.session is not None and monotonic() < deadline:
            sleep(0.01)
        assert unit.session is None
        assert not unit.service._conns
        assert unit.text_prompt_queue.empty()


def test_rejected_webrtc_scope_releases_registration():
    pytest.importorskip("aiortc")
    unit = make_pipeline_unit()
    lease = ContextScope("tenant", "issuer", "expired", monotonic() - 1)

    @asynccontextmanager
    async def hooks():
        yield object()

    with TestClient(scoped_app(unit, hooks, lease)) as client:
        response = client.post("/v1/realtime/calls", headers={"Content-Type": "application/sdp"}, content="v=0")
        assert response.status_code == 500
        assert unit.session is None
        assert not unit.service._conns
