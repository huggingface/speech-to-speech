"""The optional memory's server surface: nothing when it is off, its status and erase route when it is on."""

from threading import Event as ThreadingEvent

import pytest
from starlette.testclient import TestClient

from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.memory import reset_memory_provider
from tests.openai_realtime.conftest import make_pipeline_unit


class FakeProvider:
    """Stands in for the AgentMemory provider; needs no native library."""

    writable = True

    def __init__(self):
        self.connected_sessions: list[str] = []
        self.forgotten: list[str] = []

    def status(self):
        return {"enabled": True, "profile": "explicit_write", "state": "ready", "deadlineMs": 450}

    def connected(self, conn_id):
        self.connected_sessions.append(conn_id)

    def forget_connection(self, conn_id):
        self.forgotten.append(conn_id)


@pytest.fixture
def memory_off():
    reset_memory_provider(None)
    yield
    reset_memory_provider(None)


@pytest.fixture
def memory_on():
    provider = FakeProvider()
    reset_memory_provider(provider)
    yield provider
    reset_memory_provider(None)


def test_with_memory_off_the_capabilities_and_routes_are_unchanged(memory_off):
    app = create_app(pool=[make_pipeline_unit(index=0)], stop_event=ThreadingEvent())
    with TestClient(app) as client:
        capabilities = client.get("/v1/realtime/capabilities").json()
        assert set(capabilities) == {"pipelines"}
        assert client.post("/v1/memory/erase", json={"confirm": True}).status_code == 404
    assert not any(getattr(route, "path", None) == "/v1/memory/erase" for route in app.routes)


def test_with_memory_on_the_capabilities_report_its_status(memory_on):
    app = create_app(pool=[make_pipeline_unit(index=0)], stop_event=ThreadingEvent())
    with TestClient(app) as client:
        capabilities = client.get("/v1/realtime/capabilities").json()
        assert capabilities["memory"] == memory_on.status()


def test_with_memory_on_the_erase_route_stays_closed_without_a_token_file(memory_on, monkeypatch):
    monkeypatch.delenv("S2S_MEMORY_CONTROL_TOKEN_FILE", raising=False)
    app = create_app(pool=[make_pipeline_unit(index=0)], stop_event=ThreadingEvent())
    with TestClient(app) as client:
        response = client.post("/v1/memory/erase", json={"confirm": True})
        assert response.status_code == 404
        assert response.json() == {"error": "memory_control_off"}


def test_a_session_tells_memory_when_it_opens_and_closes(memory_on):
    service = make_pipeline_unit(index=0).service
    session_id = service.register()
    assert memory_on.connected_sessions == [session_id]
    service.unregister(session_id)
    assert memory_on.forgotten == [session_id]
