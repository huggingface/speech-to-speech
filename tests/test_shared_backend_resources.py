"""Exercise shared ownership through the real LLM factories and SDK requests."""

import json
from dataclasses import replace
from threading import Event, Thread
from types import SimpleNamespace

import httpx
import pytest
from openai import AuthenticationError, OpenAI

import speech_to_speech.LLM.base_openai_compatible_language_model as llm_module
from speech_to_speech import s2s_pipeline
from speech_to_speech.backend_registry import create_backend_handler
from speech_to_speech.LLM.shared_client import SharedOpenAIClient
from speech_to_speech.pipeline.messages import EndOfResponse, ResponsePrefetchTransaction
from speech_to_speech.pipeline.runtime import PipelineRuntime
from tests.test_backend_registry import _context
from tests.test_responses_api_language_model import _make_request


class TrackedHttpClient(httpx.Client):
    def __init__(self, respond):
        super().__init__(transport=httpx.MockTransport(respond))
        self.close_count = 0

    def close(self):
        self.close_count += 1
        super().close()


def _reply(request, text="Hello."):
    if request.url.path.endswith("chat/completions"):
        body = {
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
        }
    else:
        body = {
            "id": "resp_test",
            "object": "response",
            "created_at": 0,
            "model": "test",
            "status": "completed",
            "output": [
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
        }
    return httpx.Response(200, json=body)


def _build(monkeypatch, backend="responses-api", respond=_reply, *, compact=False, http_client=None):
    if http_client is None:
        http_client = TrackedHttpClient(respond)
    created = []

    def create_client(**kwargs):
        client = OpenAI(**kwargs, http_client=http_client, max_retries=0)
        created.append(client)
        return client

    monkeypatch.setattr("openai.OpenAI", create_client)
    args = s2s_pipeline.parse_arguments(
        [
            "--llm_backend",
            backend,
            "--responses_api_base_url",
            "http://localhost:1234/v1",
            "--responses_api_api_key",
            "test",
            "--num_pipelines",
            "2",
        ]
    )
    args.llm_backend.config["compact_history"] = compact

    def build_unit(**kwargs):
        context = replace(
            _context(),
            stop_event=kwargs["stop_event"],
            pipeline_index=kwargs["index"],
            openai_client_resource=kwargs["openai_client_resource"],
        )
        handler = create_backend_handler(kwargs["llm_backend"], context)
        return SimpleNamespace(handlers=[handler])

    monkeypatch.setattr(s2s_pipeline, "_build_pipeline_unit", build_unit)
    runtime = s2s_pipeline.build_pipeline(args, Event())
    # Exercise the real LLM worker threads, without starting an HTTP server.
    runtime._manager.handlers = runtime.handlers[:-1]
    return runtime, http_client, created


@pytest.mark.parametrize("backend", ["responses-api", "chat-completions"])
def test_selected_backend_shares_client_but_keeps_conversations_separate(monkeypatch, backend):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return _reply(request)

    runtime, http_client, created = _build(monkeypatch, backend, respond)
    first, second = runtime.handlers
    assert len(created) == 1
    assert first.client is second.client is created[0]
    assert first.cancel_scope is not second.cancel_scope
    assert first.speculative_turns is not second.speculative_turns
    runtime.start()
    try:
        for index, handler in enumerate(runtime.handlers):
            request = _make_request(f"Question {index}")
            request.turn_id = request.turn_revision = None
            handler.queue_in.put(request)
            while not isinstance(handler.queue_out.get(timeout=2), EndOfResponse):
                pass
        assert len(requests) == 4  # two warmups and two conversations
        assert "Question 0" in str(requests[2]) and "Question 1" not in str(requests[2])
        assert "Question 1" in str(requests[3]) and "Question 0" not in str(requests[3])
    finally:
        runtime.stop()
    runtime.stop()
    runtime.wait()
    assert http_client.is_closed
    assert http_client.close_count == 1


@pytest.mark.parametrize("work", ["prefetch", "compaction"])
def test_shutdown_waits_for_real_background_provider_work(monkeypatch, work):
    entered, release, block = Event(), Event(), Event()

    def respond(request):
        if block.is_set():
            entered.set()
            assert release.wait(3)
        return _reply(request, '{"user_summary":"question","assistant_summary":"answer"}')

    runtime, http_client, _ = _build(monkeypatch, respond=respond, compact=True)
    runtime.CLEANUP_WAIT_S = 0.01
    handler = runtime.handlers[0]
    block.set()
    if work == "prefetch":
        worker = handler._start_prefetch_worker(
            lambda: handler.client.responses.create(model="test", input="hello"), name="test-prefetch"
        )
    else:
        worker = Thread(target=lambda: handler.compactor([{"role": "user", "content": "question"}]))
        worker.start()
    assert entered.wait(2)
    try:
        runtime.stop()
        assert not http_client.is_closed
    finally:
        release.set()
        worker.join(timeout=2)
        runtime.stop()
        runtime.wait()
    assert http_client.close_count == 1


def test_cancelled_prefetch_does_not_close_other_pipeline_client(monkeypatch):
    entered, release = Event(), Event()

    def respond(request):
        if "blocked question" in str(json.loads(request.content)):
            entered.set()
            assert release.wait(3)
        return _reply(request)

    runtime, http_client, _ = _build(monkeypatch, respond=respond)
    first, second = runtime.handlers
    request = _make_request("blocked question")
    request.turn_id = request.turn_revision = None
    request.prefetch_transaction = ResponsePrefetchTransaction()
    runtime.start()
    try:
        first.queue_in.put(request)
        assert entered.wait(2)
        first.cancel_scope.cancel()
        while not isinstance(first.queue_out.get(timeout=2), EndOfResponse):
            pass
        assert not http_client.is_closed
        other_request = _make_request("other pipeline question")
        other_request.turn_id = other_request.turn_revision = None
        second.queue_in.put(other_request)
        while True:
            output = second.queue_out.get(timeout=2)
            if isinstance(output, EndOfResponse):
                assert output.error is None
                break
    finally:
        release.set()
        runtime.stop()
    assert http_client.close_count == 1


def test_construction_failure_closes_the_shared_client(monkeypatch):
    http_client = TrackedHttpClient(_reply)
    monkeypatch.setattr("openai.OpenAI", lambda **kw: OpenAI(**kw, http_client=http_client))
    args = s2s_pipeline.parse_arguments(["--responses_api_api_key", "test", "--num_pipelines", "2"])

    def build_unit(**kwargs):
        if kwargs["index"] == 1:
            raise RuntimeError("second pipeline failed")
        context = replace(_context(), openai_client_resource=kwargs["openai_client_resource"])
        handler = create_backend_handler(kwargs["llm_backend"], context)
        return SimpleNamespace(handlers=[handler])

    monkeypatch.setattr(s2s_pipeline, "_build_pipeline_unit", build_unit)
    with pytest.raises(RuntimeError, match="second pipeline failed"):
        s2s_pipeline.build_pipeline(args, Event())
    assert http_client.close_count == 1


def test_warmup_failure_closes_the_shared_client(monkeypatch):
    def fail(request):
        return httpx.Response(401, json={"error": {"message": "provider startup failed"}})

    http_client = TrackedHttpClient(fail)
    with pytest.raises(AuthenticationError, match="provider startup failed"):
        _build(monkeypatch, http_client=http_client)
    assert http_client.close_count == 1


def test_cleanup_failure_preserves_backend_startup_error(monkeypatch, caplog):
    def fail(request):
        return httpx.Response(401, json={"error": {"message": "original startup failure"}})

    http_client = TrackedHttpClient(fail)
    close = http_client.close

    def broken_close():
        close()
        raise RuntimeError("cleanup also failed")

    monkeypatch.setattr(http_client, "close", broken_close)
    with pytest.raises(AuthenticationError, match="original startup failure"):
        _build(monkeypatch, http_client=http_client)
    assert http_client.close_count == 1
    assert "Shared LLM client cleanup failed during construction" in caplog.text


@pytest.mark.parametrize("failure", ["construction", "start"])
def test_prefetch_worker_failure_releases_client_and_admission_slot(monkeypatch, failure):
    runtime, http_client, _ = _build(monkeypatch)
    handler = runtime.handlers[0]
    create_thread = llm_module.Thread

    def fail_start():
        raise RuntimeError("worker initialization failed")

    def broken_thread(*args, **kwargs):
        if failure == "construction":
            raise RuntimeError("worker initialization failed")
        thread = create_thread(*args, **kwargs)
        thread.start = fail_start
        return thread

    with monkeypatch.context() as patch:
        patch.setattr(llm_module, "Thread", broken_thread)
        with pytest.raises(RuntimeError, match="worker initialization failed"):
            handler._start_prefetch_worker(lambda: None, name="failed-prefetch")

    # Another real provider operation must still be admitted after the error.
    worker = handler._start_prefetch_worker(
        lambda: handler.client.responses.create(model="test", input="hello"), name="successful-prefetch"
    )
    assert worker is not None
    worker.join(timeout=2)
    runtime.CLEANUP_WAIT_S = 0.1
    runtime.stop()
    assert http_client.close_count == 1


def test_worker_timeout_defers_cleanup_until_worker_exits(monkeypatch):
    entered, release = Event(), Event()
    http_client = TrackedHttpClient(_reply)
    resource = SharedOpenAIClient(OpenAI(api_key="test", http_client=http_client))
    stop_event = Event()

    def run():
        entered.set()
        release.wait(3)
        # A timed-out foreground worker still has access to the client.
        resource.client.responses.create(model="test", input="hello")

    runtime = PipelineRuntime([SimpleNamespace(stop_event=stop_event, run=run)], resource)
    runtime.CLEANUP_WAIT_S = 0.01
    # Simulate ThreadManager's timeout without waiting its full five seconds.
    monkeypatch.setattr(runtime._manager, "stop", stop_event.set)
    runtime.start()
    assert entered.wait(2)
    try:
        runtime.stop()
        assert stop_event.is_set()
        assert not http_client.is_closed
    finally:
        release.set()
        runtime.wait()
    assert http_client.close_count == 1


@pytest.mark.parametrize("cleanup_can_start", [False, True])
def test_worker_start_failure_closes_client(monkeypatch, cleanup_can_start):
    http_client = TrackedHttpClient(_reply)
    resource = SharedOpenAIClient(OpenAI(api_key="test", http_client=http_client))
    stop_event = Event()
    runtime = PipelineRuntime([SimpleNamespace(stop_event=stop_event, run=lambda: None)], resource)
    start = Thread.start

    def fail_worker_start(thread):
        if not cleanup_can_start or thread.name != "pipeline-resource-cleanup":
            raise RuntimeError("cannot start worker")
        return start(thread)

    monkeypatch.setattr(Thread, "start", fail_worker_start)
    with pytest.raises(RuntimeError, match="cannot start worker"):
        runtime.start()
    assert stop_event.is_set()
    runtime.stop()
    runtime.wait()
    assert http_client.close_count == 1


def test_local_client_construction_failure_closes_server_resources(monkeypatch):
    runtime, http_client, _ = _build(monkeypatch)
    monkeypatch.setattr(s2s_pipeline, "build_pipeline", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(
        "speech_to_speech.api.openai_realtime.audio_client.RealtimeAudioClient",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("audio device failed")),
    )
    with pytest.raises(RuntimeError, match="audio device failed"):
        s2s_pipeline.build_local_pipeline(s2s_pipeline.parse_arguments([], command="local"), Event())
    assert http_client.close_count == 1
