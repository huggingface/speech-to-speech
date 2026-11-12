"""Streamed LLM turns against a real HTTP/1.1 provider.

The provider is a real server so the tests see what the HTTP client does with
the connection and what the SDK raises. openai before 3.30 stopped reading at
``data: [DONE]``, before the end of the chunked body, and the connection was
dropped instead of reused; every streamed turn then paid a new handshake.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx2
import pytest
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from openai.types.realtime.realtime_session_create_request import RealtimeSessionCreateRequest

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler
from speech_to_speech.LLM.responses_api_language_model import ResponsesApiModelHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import EndOfResponse, GenerateResponseRequest, LLMResponseChunk
from tests.llm_history import drive_llm

REPLY = "Hello there."


def _sse(payload: dict, *, event: str | None = None) -> bytes:
    prefix = f"event: {event}\n".encode() if event else b""
    return prefix + b"data: " + json.dumps(payload).encode() + b"\n\n"


def _chat_events() -> list[bytes]:
    def chunk(delta: dict, finish_reason: str | None = None) -> bytes:
        return _sse(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "fake",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
        )

    return [chunk({"role": "assistant", "content": REPLY}), chunk({}, "stop")]


def _responses_events() -> list[bytes]:
    item = {"id": "msg_1", "type": "message", "role": "assistant", "status": "in_progress", "content": []}
    done_item = {**item, "status": "completed", "content": [{"type": "output_text", "text": REPLY, "annotations": []}]}
    response = {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "fake",
        "status": "in_progress",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    events = [
        {"type": "response.created", "response": response},
        {"type": "response.output_item.added", "output_index": 0, "item": item},
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": REPLY,
            "logprobs": [],
        },
        {"type": "response.output_item.done", "output_index": 0, "item": done_item},
        {"type": "response.completed", "response": {**response, "status": "completed", "output": [done_item]}},
    ]
    return [_sse({**event, "sequence_number": i}, event=event["type"]) for i, event in enumerate(events)]


BACKENDS = [
    pytest.param(ChatCompletionsApiModelHandler, _chat_events(), id="chat-completions"),
    pytest.param(ResponsesApiModelHandler, _responses_events(), id="responses"),
]


class _Provider:
    """Stream ``events`` as a chunked SSE body that ends with ``[DONE]``."""

    def __init__(self, events: list[bytes], *, stall_before_done: threading.Event | None = None) -> None:
        self.connections: list[tuple[str, int]] = []
        provider = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _write(self, data: bytes) -> None:
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                self.wfile.flush()

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                provider.connections.append(self.client_address)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    for event in events:
                        self._write(event)
                    if stall_before_done is not None:
                        stall_before_done.wait(10)
                    self._write(b"data: [DONE]\n\n")
                    # Like real servers, send the end of the body as its own write.
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever)
        self._thread.start()
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()


def _handler(handler_cls, provider: _Provider, monkeypatch):
    monkeypatch.setattr(handler_cls, "warmup", lambda self: None)
    return handler_cls(
        threading.Event(),
        queue.Queue(),
        queue.Queue(),
        setup_kwargs={"base_url": provider.base_url, "api_key": "none", "cancel_scope": CancelScope()},
    )


def _request(turn: int) -> GenerateResponseRequest:
    chat = Chat(10)
    chat.add_item(make_user_message("Hello"))
    session = RealtimeSessionCreateRequest(type="realtime", instructions="You are helpful.")
    return GenerateResponseRequest(
        runtime_config=RuntimeConfig(chat=chat, session=session),
        response=RealtimeResponseCreateParams(output_modalities=["text"]),
        language_code="en",
        turn_id=f"turn_{turn}",
        turn_revision=0,
    )


@pytest.mark.parametrize(("handler_cls", "events"), BACKENDS)
def test_streamed_turns_reuse_one_connection(handler_cls, events, monkeypatch):
    provider = _Provider(events)
    handler = _handler(handler_cls, provider, monkeypatch)
    try:
        for turn in range(4):
            outputs = list(drive_llm(handler, _request(turn)))
            assert "".join(o.text for o in outputs if isinstance(o, LLMResponseChunk)) == REPLY
            assert outputs[-1].error is None
        assert len(provider.connections) == 4
        assert len(set(provider.connections)) == 1, "each streamed turn opened a new connection"
    finally:
        handler.cleanup()
        provider.close()


@pytest.mark.parametrize(("handler_cls", "events"), BACKENDS)
def test_interrupted_stream_does_not_wait_for_the_rest_of_the_body(handler_cls, events, monkeypatch):
    stall = threading.Event()
    provider = _Provider(events, stall_before_done=stall)
    handler = _handler(handler_cls, provider, monkeypatch)
    try:
        started = time.monotonic()
        outputs = []
        for output in drive_llm(handler, _request(0)):
            outputs.append(output)
            if isinstance(output, LLMResponseChunk):
                # The provider is silent from here on; the turn is barged in on.
                handler.cancel_scope.cancel()
        assert isinstance(outputs[-1], EndOfResponse)
        assert time.monotonic() - started < 5, "the interrupted turn waited for the provider"
    finally:
        stall.set()
        handler.cleanup()
        provider.close()


@pytest.mark.parametrize(("handler_cls", "events"), BACKENDS)
def test_stalled_stream_ends_the_turn_with_the_read_timeout(handler_cls, events, monkeypatch):
    stall = threading.Event()
    provider = _Provider(events, stall_before_done=stall)
    handler = _handler(handler_cls, provider, monkeypatch)
    handler.request_timeout_s = 0.3
    handler.request_timeout = httpx2.Timeout(0.3)
    try:
        outputs = list(drive_llm(handler, _request(0)))
        assert isinstance(outputs[-1], EndOfResponse)
        assert outputs[-1].error == "Language model generation timed out after 0.3s."
    finally:
        stall.set()
        handler.cleanup()
        provider.close()
