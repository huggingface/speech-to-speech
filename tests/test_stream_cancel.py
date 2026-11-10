"""A cancelled streaming turn must not wait for the provider to speak again.

The provider socket can stay busy without producing an event: SSE comment
keep-alives keep the read alive while the SDK ignores them, so neither the
cancellation check nor the read timeout in the turn gets a chance to run.
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from openai.types.realtime.realtime_session_create_request import RealtimeSessionCreateRequest

import speech_to_speech.LLM.base_openai_compatible_language_model as base_mod
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler
from speech_to_speech.LLM.responses_api_language_model import ResponsesApiModelHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import EndOfResponse, GenerateResponseRequest, LLMResponseChunk, TokenUsage
from tests.llm_history import drive_llm

# A cancelled turn that still waits on the provider is what this file is about;
# give the regression plenty of room, but never let a broken build hang CI.
CANCEL_DEADLINE_S = 5.0


class _KeepAliveThenSilentStream:
    """One chunk, then keep-alive traffic that never becomes an event.

    ``close`` releases the pending read, as the real SDK does; nothing else
    does, so a turn that never closes the response cannot finish.
    """

    def __init__(self):
        self.released = threading.Event()
        self.closed = threading.Event()
        self.reading = threading.Event()

    def __iter__(self):
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hallo.", tool_calls=None), finish_reason=None)],
            usage=None,
        )
        # Reached only when the consumer asks for the next event, i.e. while the
        # real read sits waiting for another SSE line.
        self.reading.set()
        self.released.wait(60)

    def close(self):
        self.closed.set()
        self.released.set()


class _FakeCompletions:
    def __init__(self, stream):
        self._stream = stream

    def create(self, **kwargs):
        return self._stream


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=None)

    def with_options(self, **kwargs):
        return self

    def close(self):
        pass


def _make_handler(stream):
    original = base_mod.OpenAI

    def build(*args, **kwargs):
        client = _FakeClient()
        client.chat.completions = _FakeCompletions(stream)
        return client

    base_mod.OpenAI = build
    try:
        return ChatCompletionsApiModelHandler(
            threading.Event(),
            queue.Queue(),
            queue.Queue(),
            setup_kwargs=dict(
                model_name="test-model",
                base_url="http://fake/v1",
                api_key="k",
                stream=True,
                disable_thinking=True,
                compact_history=False,
            ),
        )
    finally:
        base_mod.OpenAI = original


def _request(text="Hallo"):
    chat = Chat(10)
    chat.add_item(make_user_message(text))
    session = RealtimeSessionCreateRequest(type="realtime", instructions="Du bist ein Roboter.")
    return GenerateResponseRequest(
        runtime_config=RuntimeConfig(chat=chat, session=session),
        response=None,
        language_code="de",
        turn_id="turn_1",
        turn_revision=0,
    )


def _drive_in_background(handler, request, *, first_output=None):
    """Run one turn on its own thread, as the pipeline does, and record it."""
    result: dict[str, object] = {}
    finished = threading.Event()

    def run():
        outputs = []
        try:
            for output in drive_llm(handler, request):
                outputs.append(output)
                if first_output is not None and isinstance(output, LLMResponseChunk):
                    first_output.set()
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertions
            result["error"] = exc
        else:
            result["outputs"] = outputs
        finally:
            finished.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, finished, result


def test_cancel_releases_a_stream_reading_keep_alives():
    stream = _KeepAliveThenSilentStream()

    scope = CancelScope()
    handler = _make_handler(stream)
    handler.cancel_scope = scope
    request = _request()

    thread, finished, result = _drive_in_background(handler, request)
    assert stream.reading.wait(10), "the turn never started waiting for the next provider event"

    scope.cancel()
    assert finished.wait(CANCEL_DEADLINE_S), "a cancelled turn stayed blocked on the provider read"
    thread.join(1)
    assert "error" not in result, result.get("error")

    outputs = result["outputs"]
    assert isinstance(outputs[-1], EndOfResponse)
    assert outputs[-1].error is None
    assert stream.closed.is_set(), "the response must be released when the turn is cancelled"
    assert handler._stream_worker is None
    assert [getattr(item, "role", None) for item in request.runtime_config.chat.buffer] == ["user"]


def test_completed_stream_still_delivers_every_event():
    class _Chunks:
        def __init__(self):
            self.closed = threading.Event()

        def __iter__(self):
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Hallo.", tool_calls=None), finish_reason=None)],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=" Wie geht es dir?", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5),
            )

        def close(self):
            self.closed.set()

    stream = _Chunks()
    handler = _make_handler(stream)
    text = ""
    end = None
    for output in drive_llm(handler, _request()):
        if isinstance(output, LLMResponseChunk):
            text += output.text
        elif isinstance(output, EndOfResponse):
            end = output
    assert "Hallo" in text and "Wie geht es dir" in text
    assert end is not None and end.error is None
    assert stream.closed.is_set(), "the response is released once the turn completes"


def _tls_contexts(tmp_path):
    """A locally trusted certificate; no external server or openssl executable."""
    import datetime
    import ipaddress
    import ssl

    pytest.importorskip("cryptography")
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
        .sign(key, hashes.SHA256())
    )
    certificate_path, key_path = tmp_path / "server.crt", tmp_path / "server.key"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(certificate_path, key_path)
    return server_context, ssl.create_default_context(cafile=str(certificate_path))


@pytest.mark.parametrize("handler_class", [ChatCompletionsApiModelHandler, ResponsesApiModelHandler])
@pytest.mark.parametrize("use_tls", [False, True], ids=["http", "https"])
@pytest.mark.parametrize("comment_interval", [0.05, 15.0], ids=["frequent-comments", "blocked-read"])
def test_real_sdk_cancel_reaps_reader_and_next_turn_completes(
    monkeypatch, tmp_path, handler_class, use_tls, comment_interval
):
    release = threading.Event()
    server_context, verify = _tls_contexts(tmp_path) if use_tls else (None, True)
    completed_connections = []

    class Provider(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not body.get("stream"):
                content = json.dumps({"id": "warmup", "choices": [], "output": []}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def send(data):
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            def event(data):
                send(b"data: " + json.dumps(data).encode() + b"\n\n")

            responses_api = self.path.endswith("/responses")

            def chat_chunk(delta, finish_reason=None):
                return {
                    "id": "local",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "local",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                }

            try:
                if responses_api:
                    event(
                        {
                            "type": "response.output_text.delta",
                            "delta": "Ready.",
                            "item_id": "msg_local",
                            "output_index": 0,
                            "content_index": 0,
                            "sequence_number": 0,
                        }
                    )
                else:
                    event(chat_chunk({"content": "Ready."}))

                if "COMPLETE" not in json.dumps(body):
                    # Both frequent comments (no normalized event or timeout)
                    # and long gaps must be interruptible by the same path.
                    send(b": ping\n\n")
                    while not release.wait(comment_interval):
                        send(b": ping\n\n")
                else:
                    completed_connections.append(self.connection)
                    if responses_api:
                        event(
                            {
                                "type": "response.output_item.done",
                                "sequence_number": 1,
                                "output_index": 0,
                                "item": {
                                    "id": "msg_local",
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "completed",
                                    "content": [{"type": "output_text", "text": "Ready.", "annotations": []}],
                                },
                            }
                        )
                        event(
                            {
                                "type": "response.completed",
                                "sequence_number": 2,
                                "response": {"id": "resp_local", "status": "completed", "output": []},
                            }
                        )
                    else:
                        event(chat_chunk({}, finish_reason="stop"))
                    # Exhaust the HTTP body, so HTTPX can pool the connection.
                    # The SDK stops early on [DONE] and closes that connection.
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    if server_context is not None:
        server.socket = server_context.wrap_socket(server.socket, server_side=True)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    monkeypatch.setattr(
        base_mod,
        "OpenAI",
        lambda **kwargs: OpenAI(http_client=httpx.Client(verify=verify, trust_env=False), **kwargs),
    )
    scope = CancelScope()
    handler = None
    readers = []
    consumers = []
    try:
        handler = handler_class(
            threading.Event(),
            queue.Queue(),
            queue.Queue(),
            setup_kwargs={
                "base_url": f"{'https' if use_tls else 'http'}://127.0.0.1:{server.server_port}/v1",
                "api_key": "none",
                "model_name": "local",
                "cancel_scope": scope,
                "request_timeout_s": 20,
            },
        )
        for i in range(3):
            request = _request(f"CANCEL {i}")
            request.response = RealtimeResponseCreateParams(output_modalities=["text"])
            first_output = threading.Event()
            consumer, finished, result = _drive_in_background(handler, request, first_output=first_output)
            consumers.append(consumer)
            assert first_output.wait(5), "provider never delivered its first delta"
            reader = handler._stream_worker
            assert reader is not None
            readers.append(reader)
            scope.cancel()
            assert finished.wait(2), "cancellation waited on provider traffic or its 20s read timeout"
            assert "error" not in result
            assert not reader.is_alive(), "a cancelled SDK reader was abandoned"
            assert handler._stream_worker is None
            assert isinstance(result["outputs"][-1], EndOfResponse)
            assert result["outputs"][-1].error is None
            assert [item.role for item in request.runtime_config.chat.buffer] == ["user"]

        # The same handler must accept the next turn. Normal completion must
        # retain connection pooling rather than shutting down every response.
        for _ in range(2):
            request = _request("COMPLETE")
            request.response = RealtimeResponseCreateParams(output_modalities=["text"])
            outputs = list(drive_llm(handler, request))
            assert "".join(output.text for output in outputs if isinstance(output, LLMResponseChunk)) == "Ready."
            assert isinstance(outputs[-1], EndOfResponse) and outputs[-1].error is None
            assert [item.role for item in request.runtime_config.chat.buffer] == ["user", "assistant"]
            assert handler._stream_worker is None
        assert completed_connections[0] is completed_connections[1]
    finally:
        release.set()
        if handler is not None:
            handler.cleanup()
        for thread in consumers + readers:
            thread.join(2)
        server.shutdown()
        server.server_close()
        server_thread.join(2)


def test_uncooperative_stream_does_not_accumulate_readers():
    class UncooperativeStream(_KeepAliveThenSilentStream):
        def close(self):
            self.closed.set()  # Deliberately does not release its read.

    stream = UncooperativeStream()
    handler = _make_handler(stream)
    scope = handler.cancel_scope = CancelScope()
    consumer, finished, result = _drive_in_background(handler, _request())
    reader = None
    try:
        assert stream.reading.wait(5)
        reader = handler._stream_worker
        scope.cancel()
        assert finished.wait(2)
        assert result["outputs"][-1].error is None
        assert reader.is_alive()
        outputs = list(drive_llm(handler, _request()))
        assert "previous language model stream reader has not stopped" in outputs[-1].error
        assert handler._stream_worker is reader
    finally:
        stream.released.set()
        consumer.join(2)
        if reader is not None:
            reader.join(2)
        handler.cleanup()
    assert handler._stream_worker is None


def test_usage_survives_cancel_between_queue_read_and_consumer(monkeypatch):
    stream = _KeepAliveThenSilentStream()
    handler = _make_handler(stream)
    scope = handler.cancel_scope = CancelScope()
    handler._iter_stream_events = lambda _: iter(
        [base_mod.TextDelta(text="unseen"), base_mod.Usage(input_tokens=11, output_tokens=7)]
    )

    class CancellingQueue(queue.Queue):
        def get(self, *args, **kwargs):
            item = super().get(*args, **kwargs)
            scope.cancel()
            return item

    monkeypatch.setattr(base_mod, "Queue", CancellingQueue)
    request = _request()
    outputs = list(drive_llm(handler, request))

    usage = next(output for output in outputs if isinstance(output, TokenUsage))
    assert (usage.input_tokens, usage.output_tokens) == (11, 7)
    assert not any(isinstance(output, LLMResponseChunk) for output in outputs)
    assert outputs[-1].error is None
    assert [item.role for item in request.runtime_config.chat.buffer] == ["user"]
    assert handler._stream_worker is None


def test_reader_start_failure_closes_response(monkeypatch):
    stream = _KeepAliveThenSilentStream()
    handler = _make_handler(stream)

    class FailingThread(threading.Thread):
        def start(self):
            raise RuntimeError("cannot start reader")

    monkeypatch.setattr(base_mod, "Thread", FailingThread)
    outputs = list(drive_llm(handler, _request()))

    assert "cannot start reader" in outputs[-1].error
    assert stream.closed.is_set()
    assert handler._stream_worker is None
