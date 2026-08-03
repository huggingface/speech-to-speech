"""Integration tests for the LLM proxy.

Drives the full FastAPI app produced by ``create_app`` through its public
HTTP surface (Starlette TestClient), exactly like the websocket lifecycle
tests. The upstream provider is a real local HTTP server on an ephemeral
port that the proxy reaches through its configured base URL — no injection
hooks in production code.
"""

import gzip
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
from threading import Event as ThreadingEvent
from typing import Any

import httpx
import pytest
import uvicorn
from pydantic import BaseModel
from starlette.testclient import TestClient

from speech_to_speech.api.openai_realtime.llm_proxy import LLMProxyConfig
from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.pipeline.cancel_scope import CancelScope

# ---------------------------------------------------------------------------
# Fake upstream provider
# ---------------------------------------------------------------------------


class SSEScript(BaseModel):
    """Streamed answer for the fake upstream: raw frames with a delay before each.

    ``content_encoding`` labels pre-encoded frames (e.g. gzip): the handler
    advertises it so the proxy's HTTP client decodes what it reads.
    """

    frames: list[tuple[float, bytes]]
    content_encoding: str | None = None


class FakeUpstream:
    """A real OpenAI-shaped HTTP server on an ephemeral port.

    Records every request (path, headers, parsed JSON body) and answers with
    whatever ``responder`` returns: ``(status_code, json_payload)`` for a
    buffered JSON answer, or an ``SSEScript`` to stream raw frames with
    delays between them.
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
                answer = upstream.responder(request)
                if isinstance(answer, SSEScript):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    if answer.content_encoding:
                        self.send_header("Content-Encoding", answer.content_encoding)
                    self.end_headers()
                    for delay, frame in answer.frames:
                        time.sleep(delay)
                        self.wfile.write(frame)
                        self.wfile.flush()
                    return
                status, payload = answer
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


class LiveApp:
    """Run the app under a real uvicorn server on an ephemeral port.

    Needed only where TestClient cannot observe the behavior under test:
    it buffers the whole ASGI response, hiding streaming timing and
    mid-stream lifecycle.
    """

    def __init__(self, app):
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
        self._server = uvicorn.Server(config)
        self._server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> "LiveApp":
        self._thread.start()
        deadline = time.monotonic() + 5
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("uvicorn did not start in time")
            time.sleep(0.01)
        self.port = self._server.servers[0].sockets[0].getsockname()[1]
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)


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
    def test_upstream_response_arrives_verbatim(self, upstream):
        app = _make_app(_proxy_config(upstream))
        with TestClient(app) as client:
            r = client.post("/v1/chat/completions", json=CHAT_BODY)
        assert r.status_code == 200
        assert r.json() == {"id": "chatcmpl-1", "object": "chat.completion", "choices": []}

    def test_upstream_receives_forced_model_and_server_key(self, upstream):
        app = _make_app(_proxy_config(upstream))
        with TestClient(app) as client:
            client.post("/v1/chat/completions", json=CHAT_BODY)
        (request,) = upstream.requests
        assert request["path"] == "/v1/chat/completions"
        assert request["body"]["model"] == "server-model"
        assert request["body"]["messages"] == CHAT_BODY["messages"]
        assert request["headers"]["Authorization"] == "Bearer sk-server-secret"

    @pytest.mark.parametrize("raw_body", [b"[1, 2]", b'"a string"', b"42", b"null", b"true"])
    def test_valid_json_that_is_not_an_object_is_400(self, upstream, raw_body):
        # request.json() accepts any JSON value; only objects can carry the
        # forced fields, and the contract allows 400, never 500.
        app = _make_app(_proxy_config(upstream))
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                content=raw_body,
                headers={"Content-Type": "application/json"},
            )
        assert r.status_code == 400
        assert r.json()["error"]["type"] == "invalid_request_error"
        assert upstream.requests == []

    def test_non_dict_stream_options_passes_through_for_upstream_to_reject(self, upstream):
        upstream.responder = lambda request: (
            400,
            {"error": {"message": "bad stream_options", "type": "invalid_request_error"}},
        )
        app = _make_app(_proxy_config(upstream))
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={**STREAM_BODY, "stream_options": [1, 2]},
            )
        assert r.status_code == 400  # upstream's answer, not a crash
        (request,) = upstream.requests
        assert request["body"]["stream_options"] == [1, 2]  # untouched

    def test_client_bearer_is_ignored_and_never_forwarded(self, upstream):
        # The server does no authentication: whatever api_key the SDK sends
        # is accepted and replaced by the server-held upstream key.
        app = _make_app(_proxy_config(upstream))
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json=CHAT_BODY,
                headers={"Authorization": "Bearer client-credential"},
            )
        assert r.status_code == 200
        (request,) = upstream.requests
        assert request["headers"]["Authorization"] == "Bearer sk-server-secret"
        assert "client-credential" not in json.dumps(request["headers"])


class TestUnavailableContract:
    def _post(self, app, path: str):
        with TestClient(app) as client:
            return client.post(path, json=CHAT_BODY)

    def test_flag_off_is_501(self, upstream):
        app = _make_app(_proxy_config(upstream, enabled=False))
        r = self._post(app, "/v1/chat/completions")
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
        r = self._post(app, "/v1/chat/completions")
        assert r.status_code == 501
        message = r.json()["error"]["message"]
        assert "chat-completions" in message
        assert "responses-api" in message

    def test_responses_path_is_501_under_chat_completions_backend(self, upstream):
        app = _make_app(_proxy_config(upstream))
        r = self._post(app, "/v1/responses")
        assert r.status_code == 501
        assert "/v1/chat/completions" in r.json()["error"]["message"]


SSE_FRAMES = [
    b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Hel"}}]}\n\n',
    b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"lo"}}]}\n\n',
    b"data: [DONE]\n\n",
]

STREAM_BODY = {**CHAT_BODY, "stream": True}


class TestStreamingPassthrough:
    def test_streamed_bytes_arrive_verbatim_in_order(self, upstream):
        upstream.responder = lambda request: SSEScript(frames=[(0.0, f) for f in SSE_FRAMES])
        app = _make_app(_proxy_config(upstream))
        with TestClient(app) as client:
            with client.stream("POST", "/v1/chat/completions", json=STREAM_BODY) as r:
                assert r.status_code == 200
                received = b"".join(r.iter_raw())
        assert received == b"".join(SSE_FRAMES)

    def test_frames_forward_as_they_arrive_not_buffered(self, upstream):
        # Starlette's TestClient buffers the whole ASGI response, so timing
        # is only observable over a real server.
        delay = 0.4
        upstream.responder = lambda request: SSEScript(
            frames=[(0.0, SSE_FRAMES[0]), (delay, SSE_FRAMES[1]), (0.0, SSE_FRAMES[2])]
        )
        with LiveApp(_make_app(_proxy_config(upstream))) as live:
            arrivals: list[float] = []
            with httpx.stream(
                "POST",
                f"http://127.0.0.1:{live.port}/v1/chat/completions",
                json=STREAM_BODY,
            ) as r:
                for _ in r.iter_raw():
                    arrivals.append(time.monotonic())
        # If the proxy buffered the whole upstream response, every chunk
        # would land at once and the spread would be ~0.
        assert arrivals[-1] - arrivals[0] >= delay * 0.5

    def test_unreachable_upstream_fails_cleanly_within_connect_timeout(self):
        config = LLMProxyConfig(
            enabled=True,
            llm_backend="chat-completions",
            upstream_base_url="http://10.255.255.1:9/v1",  # blackhole address
            upstream_api_key="sk-server-secret",
            model_name="server-model",
            connect_timeout_s=0.2,
        )
        app = _make_app(config)
        with TestClient(app) as client:
            start = time.monotonic()
            r = client.post("/v1/chat/completions", json=STREAM_BODY)
            elapsed = time.monotonic() - start
        assert r.status_code == 502
        assert "error" in r.json()
        assert elapsed < 3

    def test_gzip_compressed_upstream_stream_is_decoded_for_the_client(self, upstream):
        # The upstream may honor the proxy's Accept-Encoding and compress the
        # stream. The forwarded response carries no Content-Encoding header,
        # so the proxy must forward decoded bytes — and account tokens on the
        # decoded stream too.
        frames = SSE_FRAMES[:2] + [
            b'data: {"id":"chatcmpl-1","choices":[],"usage":{"prompt_tokens":11,"completion_tokens":5}}\n\n',
            b"data: [DONE]\n\n",
        ]
        compressed = gzip.compress(b"".join(frames))
        upstream.responder = lambda request: SSEScript(frames=[(0.0, compressed)], content_encoding="gzip")
        app = _make_app(_proxy_config(upstream))
        with TestClient(app) as client:
            with client.stream("POST", "/v1/chat/completions", json=STREAM_BODY) as r:
                assert r.status_code == 200
                assert "content-encoding" not in r.headers
                received = b"".join(r.iter_raw())
            usage = client.get("/v1/usage").json()["llm_proxy"]
        assert received == b"".join(frames)
        assert usage["input_tokens"] == 11
        assert usage["output_tokens"] == 5

    def test_upstream_error_before_stream_passes_through(self, upstream):
        upstream.responder = lambda request: (429, {"error": {"message": "quota", "type": "rate_limit"}})
        app = _make_app(_proxy_config(upstream))
        with TestClient(app) as client:
            r = client.post("/v1/chat/completions", json=STREAM_BODY)
        assert r.status_code == 429
        assert r.json() == {"error": {"message": "quota", "type": "rate_limit"}}


RESPONSES_BODY = {
    "model": "client-model",
    "input": [{"role": "user", "content": "hello"}],
    "store": True,
}

RESPONSES_SSE_FRAMES = [
    b'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_1"}}\n\n',
    b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"Hi"}\n\n',
    b'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_1"}}\n\n',
]


class TestResponsesPassthrough:
    def _config(self, upstream, **overrides):
        return _proxy_config(upstream, llm_backend="responses-api", **overrides)

    def test_non_streaming_passes_through_verbatim(self, upstream):
        upstream.responder = lambda request: (200, {"id": "resp_1", "object": "response", "output": []})
        app = _make_app(self._config(upstream))
        with TestClient(app) as client:
            r = client.post("/v1/responses", json=RESPONSES_BODY)
        assert r.status_code == 200
        assert r.json() == {"id": "resp_1", "object": "response", "output": []}

    def test_upstream_receives_store_false_and_forced_model(self, upstream):
        app = _make_app(self._config(upstream))
        with TestClient(app) as client:
            client.post("/v1/responses", json=RESPONSES_BODY)
        (request,) = upstream.requests
        assert request["path"] == "/v1/responses"
        assert request["body"]["store"] is False
        assert request["body"]["model"] == "server-model"
        assert request["body"]["input"] == RESPONSES_BODY["input"]

    def test_streaming_responses_grammar_passes_through_verbatim(self, upstream):
        upstream.responder = lambda request: SSEScript(frames=[(0.0, f) for f in RESPONSES_SSE_FRAMES])
        app = _make_app(self._config(upstream))
        with TestClient(app) as client:
            with client.stream("POST", "/v1/responses", json={**RESPONSES_BODY, "stream": True}) as r:
                assert r.status_code == 200
                received = b"".join(r.iter_raw())
        assert received == b"".join(RESPONSES_SSE_FRAMES)
        (request,) = upstream.requests
        assert request["body"]["store"] is False

    def test_chat_completions_path_is_501_under_responses_backend(self, upstream):
        app = _make_app(self._config(upstream))
        with TestClient(app) as client:
            r = client.post("/v1/chat/completions", json=CHAT_BODY)
        assert r.status_code == 501
        assert "/v1/responses" in r.json()["error"]["message"]

    @pytest.mark.parametrize("backend", ["transformers", "mlx-lm"])
    def test_flag_off_and_local_backend_are_501_on_responses_path(self, upstream, backend):
        for config in (
            self._config(upstream, enabled=False),
            _proxy_config(upstream, llm_backend=backend),
        ):
            app = _make_app(config)
            with TestClient(app) as client:
                r = client.post("/v1/responses", json=RESPONSES_BODY)
            assert r.status_code == 501


class TestUsageSection:
    def _post(self, client, body=None, path="/v1/chat/completions"):
        return client.post(path, json=body or CHAT_BODY)

    def test_counters_after_mixed_traffic(self, upstream):
        answers = iter(
            [
                (200, {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}),
                (200, {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}),
                (500, {"error": {"message": "boom", "type": "server_error"}}),
                (429, {"error": {"message": "quota", "type": "rate_limit"}}),
                (400, {"error": {"message": "bad", "type": "invalid_request_error"}}),
            ]
        )
        upstream.responder = lambda request: next(answers)
        app = _make_app(_proxy_config(upstream))
        with TestClient(app) as client:
            assert self._post(client).status_code == 200
            assert self._post(client).status_code == 200
            assert self._post(client).status_code == 500
            assert self._post(client).status_code == 429
            assert self._post(client).status_code == 400
            usage = client.get("/v1/usage").json()["llm_proxy"]
        assert usage["requests"] == 5
        assert usage["responses_2xx"] == 2
        assert usage["responses_5xx"] == 1
        assert usage["responses_429"] == 1
        assert usage["responses_4xx"] == 1
        assert usage["input_tokens"] == 14
        assert usage["output_tokens"] == 6

    def test_streamed_chat_completions_get_include_usage_injected_and_tokens_counted(self, upstream):
        frames = SSE_FRAMES[:2] + [
            b'data: {"id":"chatcmpl-1","choices":[],"usage":{"prompt_tokens":11,"completion_tokens":5}}\n\n',
            b"data: [DONE]\n\n",
        ]
        upstream.responder = lambda request: SSEScript(frames=[(0.0, f) for f in frames])
        app = _make_app(_proxy_config(upstream))
        with TestClient(app) as client:
            with client.stream("POST", "/v1/chat/completions", json=STREAM_BODY) as r:
                received = b"".join(r.iter_raw())
            usage = client.get("/v1/usage").json()["llm_proxy"]
        # The client did not ask for usage, the server injected it upstream,
        # and the byte stream still arrives verbatim including the usage frame.
        (request,) = upstream.requests
        assert request["body"]["stream_options"]["include_usage"] is True
        assert received == b"".join(frames)
        assert usage["input_tokens"] == 11
        assert usage["output_tokens"] == 5

    def test_streamed_responses_tokens_come_from_completed_event_without_mutation(self, upstream):
        frames = RESPONSES_SSE_FRAMES[:2] + [
            b'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_1",'
            b'"usage":{"input_tokens":9,"output_tokens":4}}}\n\n',
        ]
        upstream.responder = lambda request: SSEScript(frames=[(0.0, f) for f in frames])
        app = _make_app(_proxy_config(upstream, llm_backend="responses-api"))
        with TestClient(app) as client:
            with client.stream("POST", "/v1/responses", json={**RESPONSES_BODY, "stream": True}) as r:
                received = b"".join(r.iter_raw())
            usage = client.get("/v1/usage").json()["llm_proxy"]
        (request,) = upstream.requests
        assert "stream_options" not in request["body"]  # no request mutation on Responses
        assert received == b"".join(frames)
        assert usage["input_tokens"] == 9
        assert usage["output_tokens"] == 4

    def test_tokens_counted_from_crlf_delimited_sse(self, upstream):
        # The SSE spec allows CRLF event delimiters; accounting must not
        # depend on LF-only upstreams (the bytes pass through verbatim
        # either way).
        frames = [
            b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Hi"}}]}\r\n\r\n',
            b'data: {"id":"chatcmpl-1","choices":[],"usage":{"prompt_tokens":6,"completion_tokens":2}}\r\n\r\n',
            b"data: [DONE]\r\n\r\n",
        ]
        upstream.responder = lambda request: SSEScript(frames=[(0.0, f) for f in frames])
        app = _make_app(_proxy_config(upstream))
        with TestClient(app) as client:
            with client.stream("POST", "/v1/chat/completions", json=STREAM_BODY) as r:
                received = b"".join(r.iter_raw())
            usage = client.get("/v1/usage").json()["llm_proxy"]
        assert received == b"".join(frames)
        assert usage["input_tokens"] == 6
        assert usage["output_tokens"] == 2

    def test_non_streaming_responses_tokens_come_from_body(self, upstream):
        upstream.responder = lambda request: (
            200,
            {"id": "resp_1", "output": [], "usage": {"input_tokens": 21, "output_tokens": 8}},
        )
        app = _make_app(_proxy_config(upstream, llm_backend="responses-api"))
        with TestClient(app) as client:
            assert self._post(client, RESPONSES_BODY, "/v1/responses").status_code == 200
            usage = client.get("/v1/usage").json()["llm_proxy"]
        assert usage["input_tokens"] == 21
        assert usage["output_tokens"] == 8

    def test_existing_usage_shape_keeps_its_keys(self, upstream):
        app = _make_app(_proxy_config(upstream))
        with TestClient(app) as client:
            data = client.get("/v1/usage").json()
        # Pre-existing aggregate keys survive next to the additive section.
        assert "connections" in data
        assert "llm_proxy" in data


class TestProxyConfigFollowsBackendSettings:
    def test_config_reads_the_renamed_connection_fields(self):
        # Regression: main() runs prepare_all_args() before the proxy config
        # is built, and rename_args() pops responses_api_base_url/api_key
        # into base_url/api_key. Reading the raw names afterwards silently
        # returned the dataclass class defaults (None), so the proxy hit the
        # OpenAI default upstream with no key regardless of the flags while
        # the pipeline LM used the configured provider.
        from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
        from speech_to_speech.arguments_classes.responses_api_language_model_arguments import (
            ResponsesApiLanguageModelHandlerArguments,
        )
        from speech_to_speech.s2s_pipeline import build_llm_proxy_config, rename_args

        module_kwargs = ModuleArguments(enable_llm_proxy=True, llm_backend="chat-completions")
        lm_kwargs = ResponsesApiLanguageModelHandlerArguments(
            model_name="google/gemma-test",
            responses_api_base_url="https://router.huggingface.co/v1",
            responses_api_api_key="hf_secret",
        )
        rename_args(lm_kwargs, "responses_api")  # what prepare_all_args() does before the config is built

        config = build_llm_proxy_config(module_kwargs, lm_kwargs)

        assert config.enabled is True
        assert config.llm_backend == "chat-completions"
        assert config.upstream_base_url == "https://router.huggingface.co/v1"
        assert config.upstream_api_key == "hf_secret"
        assert config.model_name == "google/gemma-test"


class TestUpstreamErrorPassthrough:
    @pytest.mark.parametrize("status", [400, 429, 500, 503])
    def test_upstream_errors_pass_through_verbatim(self, upstream, status):
        upstream.responder = lambda request: (status, {"error": {"message": "upstream says no", "type": "boom"}})
        app = _make_app(_proxy_config(upstream))
        with TestClient(app) as client:
            r = client.post("/v1/chat/completions", json=CHAT_BODY)
        assert r.status_code == status
        assert r.json() == {"error": {"message": "upstream says no", "type": "boom"}}
