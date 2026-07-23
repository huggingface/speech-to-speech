"""Integration tests for the session-gated LLM proxy.

Drives the full FastAPI app produced by ``create_app`` through its public
HTTP surface (Starlette TestClient), exactly like the websocket lifecycle
tests. The upstream provider is a real local HTTP server on an ephemeral
port that the proxy reaches through its configured base URL — no injection
hooks in production code.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
from threading import Event as ThreadingEvent
from typing import Any

import pytest
from starlette.testclient import TestClient

from speech_to_speech.api.openai_realtime.llm_proxy import LLMProxyConfig
from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.pipeline.cancel_scope import CancelScope

# ---------------------------------------------------------------------------
# Fake upstream provider
# ---------------------------------------------------------------------------


class FakeUpstream:
    """A real OpenAI-shaped HTTP server on an ephemeral port.

    Records every request (path, headers, parsed JSON body) and answers with
    whatever ``responder`` returns: ``(status_code, json_payload)``.
    """

    def __init__(self):
        self.requests: list[dict[str, Any]] = []
        self.responder = lambda request: (
            200,
            {"id": "chatcmpl-1", "object": "chat.completion", "choices": []},
        )

        upstream = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                request = {
                    "path": self.path,
                    "headers": dict(self.headers),
                    "body": json.loads(raw) if raw else None,
                }
                upstream.requests.append(request)
                status, payload = upstream.responder(request)
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def upstream():
    server = FakeUpstream()
    yield server
    server.close()


# ---------------------------------------------------------------------------
# App fixtures
# ---------------------------------------------------------------------------


def _make_unit(index: int = 0) -> PipelineUnit:
    text_prompt_queue: Queue = Queue()
    should_listen = ThreadingEvent()
    should_listen.set()
    return PipelineUnit(
        index=index,
        service=RealtimeService(text_prompt_queue=text_prompt_queue, should_listen=should_listen),
        cancel_scope=CancelScope(),
        should_listen=should_listen,
        response_playing=ThreadingEvent(),
        input_queue=Queue(),
        output_queue=Queue(),
        text_output_queue=Queue(),
        text_prompt_queue=text_prompt_queue,
        handlers=[],
    )


def _make_app(config: LLMProxyConfig | None):
    return create_app(pool=[_make_unit()], stop_event=ThreadingEvent(), llm_proxy_config=config)


def _proxy_config(upstream: FakeUpstream, **overrides: Any) -> LLMProxyConfig:
    defaults: dict[str, Any] = dict(
        enabled=True,
        llm_backend="chat-completions",
        upstream_base_url=upstream.base_url,
        upstream_api_key="sk-server-secret",
        model_name="server-model",
    )
    defaults.update(overrides)
    return LLMProxyConfig(**defaults)


CHAT_BODY = {
    "model": "client-model",
    "messages": [{"role": "user", "content": "hello"}],
}


# ---------------------------------------------------------------------------
# Chat completions passthrough (non-streaming)
# ---------------------------------------------------------------------------


class TestChatCompletionsPassthrough:
    def test_valid_session_gets_upstream_response_verbatim(self, upstream):
        app = _make_app(_proxy_config(upstream))
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                session_id = ws.receive_json()["session"]["id"]
                r = client.post(
                    "/v1/chat/completions",
                    json=CHAT_BODY,
                    headers={"Authorization": f"Bearer {session_id}"},
                )
        assert r.status_code == 200
        assert r.json() == {"id": "chatcmpl-1", "object": "chat.completion", "choices": []}

    def test_upstream_receives_forced_model_and_server_key(self, upstream):
        app = _make_app(_proxy_config(upstream))
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                session_id = ws.receive_json()["session"]["id"]
                client.post(
                    "/v1/chat/completions",
                    json=CHAT_BODY,
                    headers={"Authorization": f"Bearer {session_id}"},
                )
        (request,) = upstream.requests
        assert request["path"] == "/v1/chat/completions"
        assert request["body"]["model"] == "server-model"
        assert request["body"]["messages"] == CHAT_BODY["messages"]
        assert request["headers"]["Authorization"] == "Bearer sk-server-secret"
        assert session_id not in json.dumps(request["headers"])


class TestAuthentication:
    def test_missing_bearer_is_401(self, upstream):
        app = _make_app(_proxy_config(upstream))
        with TestClient(app) as client:
            r = client.post("/v1/chat/completions", json=CHAT_BODY)
        assert r.status_code == 401
        assert "error" in r.json()
        assert upstream.requests == []

    def test_unknown_session_id_is_401(self, upstream):
        app = _make_app(_proxy_config(upstream))
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                r = client.post(
                    "/v1/chat/completions",
                    json=CHAT_BODY,
                    headers={"Authorization": "Bearer sess_deadbeef"},
                )
        assert r.status_code == 401
        assert upstream.requests == []

    def test_closed_session_is_401(self, upstream):
        app = _make_app(_proxy_config(upstream))
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                session_id = ws.receive_json()["session"]["id"]
            r = client.post(
                "/v1/chat/completions",
                json=CHAT_BODY,
                headers={"Authorization": f"Bearer {session_id}"},
            )
        assert r.status_code == 401
        assert upstream.requests == []


class TestUnavailableContract:
    def _post_with_session(self, app, path: str):
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                session_id = ws.receive_json()["session"]["id"]
                return client.post(path, json=CHAT_BODY, headers={"Authorization": f"Bearer {session_id}"})

    def test_flag_off_is_501(self, upstream):
        app = _make_app(_proxy_config(upstream, enabled=False))
        r = self._post_with_session(app, "/v1/chat/completions")
        assert r.status_code == 501
        assert "disabled" in r.json()["error"]["message"]

    def test_no_config_defaults_to_disabled(self):
        app = create_app(pool=[_make_unit()], stop_event=ThreadingEvent())
        with TestClient(app) as client:
            r = client.post("/v1/chat/completions", json=CHAT_BODY)
        assert r.status_code == 501

    @pytest.mark.parametrize("backend", ["transformers", "mlx-lm"])
    def test_local_backend_is_501_naming_remote_backends(self, upstream, backend):
        app = _make_app(_proxy_config(upstream, llm_backend=backend))
        r = self._post_with_session(app, "/v1/chat/completions")
        assert r.status_code == 501
        message = r.json()["error"]["message"]
        assert "chat-completions" in message
        assert "responses-api" in message

    def test_responses_path_is_501_under_chat_completions_backend(self, upstream):
        app = _make_app(_proxy_config(upstream))
        r = self._post_with_session(app, "/v1/responses")
        assert r.status_code == 501
        assert "/v1/chat/completions" in r.json()["error"]["message"]


class TestUpstreamErrorPassthrough:
    @pytest.mark.parametrize("status", [400, 429, 500, 503])
    def test_upstream_errors_pass_through_verbatim(self, upstream, status):
        upstream.responder = lambda request: (status, {"error": {"message": "upstream says no", "type": "boom"}})
        app = _make_app(_proxy_config(upstream))
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                session_id = ws.receive_json()["session"]["id"]
                r = client.post(
                    "/v1/chat/completions",
                    json=CHAT_BODY,
                    headers={"Authorization": f"Bearer {session_id}"},
                )
        assert r.status_code == status
        assert r.json() == {"error": {"message": "upstream says no", "type": "boom"}}
