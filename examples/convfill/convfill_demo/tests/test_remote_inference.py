"""Real HTTP framing with fake model output; no model inference in this suite."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest

from convfill_demo.runtime.convfill_frontend import ConvFillFrontend
from convfill_demo.runtime.dialogue_state_manager import DialogueStateManager
from convfill_demo.runtime.remote_inference import InferenceError, RemoteInference, VisibleTextFilter
from convfill_demo.runtime.turn_state_manager import TurnStateManager

from .test_reasoner_cancel import _bare_system


def sse(*events, done=True):
    return (
        "".join("data: " + json.dumps(event) + "\n\n" for event in events) + ("data: [DONE]\n\n" if done else "")
    ).encode()


def chat(text=None, **extra):
    return {"choices": [{"delta": {"content": text, **extra}}]}


def test_reasoner_stream_filters_chunked_thoughts_and_flushes_tail():
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=sse(
                {"choices": []},
                chat(),
                chat("", reasoning_content="private thought"),
                chat("<|chan"),
                chat("nel>thought\nsecret. Never speak this. "),
                chat("<chan"),
                chat("nel|>Useful fact. "),
                chat("Final fact without punctuation"),
            ),
        )

    remote = RemoteInference({"base_url": "http://model/v1", "model": "gemma"}, transport=httpx.MockTransport(respond))
    assert list(remote.knowledge("conversation")) == ["Useful fact.", "Final fact without punctuation"]
    assert requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "<|think|>" not in requests[0]["messages"][0]["content"]


@pytest.mark.parametrize(
    "body,status",
    [
        (b"", 503),
        (sse(chat("unfinished"), done=False), 200),
        (b"data: {broken}\n\n", 200),
        (sse({"error": {"message": "secret"}}), 200),
    ],
)
def test_remote_http_and_truncated_stream_errors(body, status):
    remote = RemoteInference(
        {"base_url": "http://model/v1", "model": "gemma"},
        transport=httpx.MockTransport(lambda request: httpx.Response(status, content=body)),
    )
    with pytest.raises(InferenceError):
        list(remote.knowledge("prompt"))


def test_raw_talker_prompt_ids_preserve_knowledge_silence_history_and_boundaries():
    boundaries = {
        "USER_START": "<|im_start|>user\n",
        "ASSN_START": "<|im_start|>assistant\n",
        "KNOWLEDGE_START": "<|im_start|>knowledge\n",
        "END": "<|im_end|>\n",
    }
    turn = TurnStateManager(boundaries)
    prompt = turn.build_prompt("new", "<sil>", {"user_turns_history": ["old"], "responses_history": ["reply"]})
    assert prompt == (
        "<|im_start|>user\nold<|im_end|>\n<|im_start|>assistant\nreply<|im_end|>\n"
        "<|im_start|>user\nnew<|im_end|>\n<|im_start|>knowledge\n<sil><|im_end|>\n<|im_start|>assistant\n"
    )
    calls = []

    def respond(request):
        calls.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, content=sse({"choices": [{"text": "Spoken sentence."}]}))

    remote = RemoteInference(
        {"base_url": "http://model/v1", "model": "fine-tuned"}, transport=httpx.MockTransport(respond)
    )

    class Tokenizer:
        eos_token_id = 151645

        def encode(self, value, add_special_tokens):
            assert add_special_tokens is False
            return [151645] if value == "<|im_end|>" else [151644, 42, 151669, 151645]

    frontend = ConvFillFrontend.__new__(ConvFillFrontend)
    frontend.remote, frontend.tokenizer, frontend.boundary_tokens = remote, Tokenizer(), boundaries
    assert list(frontend._iter_remote_chunks(prompt)) == [("Spoken sentence.", 1)]
    path, payload = calls[0]
    assert path == "/v1/completions"
    assert payload["prompt"] == [151644, 42, 151669, 151645]
    assert payload["stop_token_ids"] == [151645]
    assert "messages" not in payload
    assert payload["skip_special_tokens"] is False


def test_reasoner_exception_is_raised_by_backend_generator():
    class Backend:
        def backend_infer(self):
            raise RuntimeError("reasoner failed")
            yield

    system = _bare_system(Backend())
    with pytest.raises(RuntimeError, match="reasoner failed"):
        list(system.backend_fn())


def test_background_failure_reaches_run_turn_and_rolls_back_history():
    class Backend:
        def backend_infer(self):
            raise RuntimeError("reasoner failed")
            yield

    system = _bare_system(Backend())
    system._turn_lock = threading.Lock()
    system._commit_lock = threading.RLock()
    system._poisoned = False
    system.convfill_config = SimpleNamespace(turn_timeout_s=1, shutdown_timeout_s=1)
    system.dialogue_state_manager = DialogueStateManager()
    system.convfill_frontend = SimpleNamespace(reset_turn_state=lambda: None)
    system.run_frontend = lambda: system.done_event.wait(1)
    with pytest.raises(RuntimeError, match="reasoner failed"):
        system.run_turn("user")
    assert system.dialogue_state_manager.user_turns == []
    assert system.dialogue_state_manager.responses == []


@pytest.mark.parametrize("cancel", [True, False])
def test_socket_stall_is_bounded_and_cancel_does_not_flush_partial_text(cancel):
    started, release = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(sse(chat("Never commit this partial sentence"), done=False))
            self.wfile.flush()
            started.set()
            release.wait(2)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    remote = RemoteInference(
        {
            "base_url": f"http://127.0.0.1:{server.server_port}/v1",
            "model": "gemma",
            "read_timeout_s": 1,
            "request_timeout_s": 0.25,
        }
    )
    output, errors = [], []

    def run():
        try:
            output.extend(remote.knowledge("prompt"))
        except InferenceError as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    try:
        worker.start()
        assert started.wait(1)
        if cancel:
            remote.cancel()
        worker.join(1)
        assert not worker.is_alive()
        assert output == []
        assert bool(errors) is not cancel
    finally:
        release.set()
        server.shutdown()
        server.server_close()


def test_control_filter_fails_closed_on_incomplete_thought():
    visible = VisibleTextFilter()
    assert visible.feed("<think>private. ") == ""
    assert visible.feed("still private") == ""


def test_cancellation_event_is_not_reused_by_next_turn():
    remote = RemoteInference({"base_url": "http://model/v1", "model": "gemma"})
    first, second = threading.Event(), threading.Event()
    remote.begin_turn(first)
    remote.cancel()
    remote.begin_turn(second)
    assert first.is_set() and not second.is_set()


def test_reasoner_can_deliver_next_knowledge_while_talker_is_generating():
    started, release = threading.Event(), threading.Event()

    class Backend:
        def backend_infer(self):
            yield "First fact."
            assert started.wait(1)
            yield "Second fact."

    system = _bare_system(Backend())
    system._turn_lock, system._commit_lock = threading.Lock(), threading.RLock()
    system._poisoned = False
    system.convfill_config = SimpleNamespace(turn_timeout_s=2, shutdown_timeout_s=1)
    system.dialogue_state_manager = DialogueStateManager()
    system.convfill_frontend = SimpleNamespace(reset_turn_state=lambda: None)
    overlap = []

    def on_thought(value):
        if value == "Second fact.":
            overlap.append(system._frontend_inflight > 0)
            release.set()

    system.on_thought = on_thought

    def frontend():
        assert system.thought_queue.get(timeout=1) == "First fact."
        with system._frontend_inflight_lock:
            system._frontend_inflight = 1
        started.set()
        assert release.wait(1)
        assert system.thought_queue.get(timeout=1) == "Second fact."

    system.run_frontend = frontend
    assert system.run_turn("question") is True
    assert overlap == [True]


def test_stuck_worker_is_bounded_and_pipeline_cannot_reuse_its_state():
    release = threading.Event()

    class Backend:
        def backend_infer(self):
            return
            yield

    system = _bare_system(Backend())
    system._turn_lock, system._commit_lock = threading.Lock(), threading.RLock()
    system._poisoned = False
    system.convfill_config = SimpleNamespace(turn_timeout_s=0.05, shutdown_timeout_s=0.05)
    system.dialogue_state_manager = DialogueStateManager()
    system.convfill_frontend = SimpleNamespace(reset_turn_state=lambda: None)
    system.run_frontend = lambda: release.wait(2)
    try:
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            system.run_turn("stuck")
        assert time.monotonic() - start < 1
        with pytest.raises(RuntimeError, match="restart"):
            system.run_turn("new session")
        assert system.cancel_event.is_set()
        assert system.dialogue_state_manager.user_turns == []
    finally:
        release.set()


def test_empty_reasoner_does_not_commit_a_filler_only_turn():
    remote = RemoteInference(
        {"base_url": "http://model/v1", "model": "gemma"},
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=sse(chat(None)))),
    )
    with pytest.raises(InferenceError, match="no visible knowledge"):
        list(remote.knowledge("prompt"))


def test_talker_stops_on_a_turn_boundary_split_across_chunks():
    frontend = ConvFillFrontend.__new__(ConvFillFrontend)
    frontend.backend = "vllm"
    frontend.boundary_tokens = {"END": "<|im_end|>\n"}
    frontend.cancel_event = threading.Event()
    closed = threading.Event()

    def chunks(prompt):
        try:
            yield "Hello. <|im_", 1
            yield "end|>user\nNever speak this turn.", 2
            raise AssertionError("Talker stream was read beyond its stop boundary")
        finally:
            closed.set()

    frontend._iter_remote_chunks = chunks
    assert list(frontend.run_convfill_inference("serialized prompt")) == ["Hello."]
    assert closed.is_set()


def test_streamed_tools_keep_arguments_out_of_knowledge_and_preserve_context():
    requests, calls = [], []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=sse(
                chat(
                    tool_calls=[
                        {"index": 0, "id": "call_1", "function": {"name": "web_search", "arguments": '{"query":'}}
                    ]
                ),
                chat(tool_calls=[{"index": 0, "function": {"arguments": '"weather"}'}}]),
            ),
        )

    remote = RemoteInference({"base_url": "http://model/v1", "model": "gemma"}, transport=httpx.MockTransport(respond))
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]}]
    remote.configure_request(
        messages=messages,
        tools=[{"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}],
        instructions="Be brief.",
        on_tool_call=lambda *args: calls.append(args),
    )
    assert list(remote.knowledge("prompt")) == []
    assert calls == [("web_search", '{"query":"weather"}', "call_1")]
    assert requests[0]["messages"][-1] == messages[0]
    assert "Be brief." in requests[0]["messages"][0]["content"]
    assert requests[0]["tool_choice"] == "auto"


def test_cancelled_tool_stream_never_dispatches_call():
    remote = RemoteInference({"base_url": "http://model/v1", "model": "gemma"})
    calls = []
    remote.configure_request(
        tools=[{"function": {"name": "web_search"}}], on_tool_call=lambda *args: calls.append(args)
    )

    def events(*args):
        yield chat(tool_calls=[{"index": 0, "function": {"name": "web_search", "arguments": "{}"}}])
        remote.cancel_event.set()

    remote.stream = events
    assert list(remote.knowledge("prompt")) == []
    assert calls == []
