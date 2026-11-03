"""State/cancellation tests with no downloaded models or GPU allocation."""

from queue import Queue
from threading import Event, Thread
from types import ModuleType

import pytest

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import make_user_message
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import EndOfResponse, GenerateResponseRequest, LLMResponseChunk


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]

    def encode(self, text):
        return list(text.encode())


@pytest.fixture
def adapter(monkeypatch):
    from speech_to_speech.LLM import zerogpu_handler as module

    worker = ModuleType("fake_worker")
    worker.get_tokenizer = lambda: Tokenizer()
    worker.get_model_ids = lambda: {"llm": "Qwen/Qwen3.5-4B"}
    worker.cancelled = Event()
    worker.cleaned = Event()
    monkeypatch.setattr(module, "runtime", worker)
    return module, worker


def handler(module, scope):
    return module.ZeroGPULLMHandler(
        Event(),
        Queue(),
        Queue(),
        setup_kwargs={
            "device": "cpu",
            "cancel_scope": scope,
            "gen_kwargs": {"max_new_tokens": 16},
            "stream_batch_sentences": 1,
        },
    )


def request(runtime, key="response-1"):
    return GenerateResponseRequest(
        runtime_config=runtime, language_code="en", response_key=key, response={"output_modalities": ["text"]}
    )


def test_history_stays_host_owned_and_next_turn_sees_it(adapter):
    module, worker = adapter
    seen = []

    def tokens(messages, kwargs, *, cancelled=None):
        seen.append(messages)
        yield "Your name is Ada."

    worker.tokens = tokens
    runtime = RuntimeConfig()
    runtime.chat.add_item(make_user_message("My name is Ada."))
    h = handler(module, CancelScope())
    results = list(h.process(request(runtime)))
    terminal = results[-1]
    assert isinstance(terminal, EndOfResponse) and terminal.error is None
    assert terminal.history.chat is runtime.chat
    assert terminal.history.complete
    assert "".join(x.text for x in results if isinstance(x, LLMResponseChunk)) == "Your name is Ada."
    assert len(runtime.chat.to_transformers_chat()) == 1  # Service, not worker, commits.
    terminal.history.write("response-1")
    terminal.history.clean_up("response-1")
    runtime.chat.add_item(make_user_message("What is my name?"))
    second = list(h.process(request(runtime, "response-2")))
    assert second[-1].error is None
    assert any(m["role"] == "assistant" and "Ada" in str(m["content"]) for m in seen[1])


def test_interrupt_while_waiting_for_first_token_cancels_worker(adapter):
    module, worker = adapter
    started = Event()

    def tokens(messages, kwargs, *, cancelled=None):
        started.set()
        from time import monotonic, sleep

        deadline = monotonic() + 2
        try:
            while not cancelled():
                assert monotonic() < deadline
                sleep(0.005)
            worker.cancelled.set()
            return
            yield
        finally:
            worker.cleaned.set()

    worker.tokens = tokens
    scope = CancelScope()
    runtime = RuntimeConfig()
    runtime.chat.add_item(make_user_message("Start a long reply"))
    results = []
    t = Thread(target=lambda: results.extend(handler(module, scope).process(request(runtime))))
    t.start()
    assert started.wait(1)
    scope.cancel()
    t.join(1)
    assert not t.is_alive()
    assert worker.cancelled.wait(1) and worker.cleaned.wait(1)
    assert not any(isinstance(x, LLMResponseChunk) for x in results)
    assert isinstance(results[-1], EndOfResponse)
    assert not results[-1].history.complete
    assert len(runtime.chat.to_transformers_chat()) == 1


def test_gpu_failure_emits_terminal_error_instead_of_sticking_response(adapter):
    module, worker = adapter

    def tokens(messages, kwargs, *, cancelled=None):
        raise RuntimeError("allocation failed")
        yield

    worker.tokens = tokens
    runtime = RuntimeConfig()
    runtime.chat.add_item(make_user_message("Hello"))
    results = list(handler(module, CancelScope()).process(request(runtime)))
    assert isinstance(results[-1], EndOfResponse)
    assert "allocation failed" in results[-1].error


@pytest.mark.parametrize("expired,abandoned", [(True, False), (False, False), (True, True)])
def test_worker_deadline_is_failure_not_successful_partial_history(adapter, tmp_path, expired, abandoned):
    """Run real worker generation and native history handling with a fake GPU model."""
    import ast
    from contextlib import nullcontext
    from pathlib import Path
    from queue import Empty
    from types import SimpleNamespace

    module, worker = adapter
    marker = tmp_path / "abandoned"
    if abandoned:
        marker.touch()

    class Inputs(dict):
        def to(self, device):
            return self

    class WorkerTokenizer(Tokenizer):
        eos_token_id = 0

        def __call__(self, *args, **kwargs):
            return Inputs()

    class Streamer:
        def __init__(self, *args, **kwargs):
            self.queue = Queue()

        def __next__(self):
            value = self.queue.get(timeout=0.2)
            if value is None:
                raise StopIteration
            return value

        def end(self):
            self.queue.put(None)

    class Model:
        def generate(self, streamer, stopping_criteria, **kwargs):
            streamer.queue.put("The answer is ")
            if not stopping_criteria[0](None, None):
                streamer.queue.put("42.")
            streamer.end()

    tree = ast.parse((Path(__file__).parents[1] / "src/speech_to_speech/zerogpu/_worker.py").read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_tokens"]
    namespace = dict(
        _assert_gpu=lambda model: None,
        llm_model=Model(),
        llm_tokenizer=WorkerTokenizer(),
        Event=Event,
        Thread=Thread,
        Empty=Empty,
        time=SimpleNamespace(monotonic=lambda: 10),
        StoppingCriteria=object,
        StoppingCriteriaList=list,
        TextIteratorStreamer=Streamer,
        torch=SimpleNamespace(inference_mode=nullcontext),
    )
    exec(compile(tree, "worker.py", "exec"), namespace)
    worker.tokens = lambda messages, options, **kwargs: namespace["_tokens"](
        messages, options, deadline=9 if expired else 11, abandoned_path=marker
    )
    if abandoned:
        # The host has already cancelled this request; do not turn it into a
        # worker timeout if abandonment and the deadline coincide.
        assert list(worker.tokens([], {})) == ["The answer is "]
        return
    config = RuntimeConfig()
    config.chat.add_item(make_user_message("What is the answer?"))
    results = list(handler(module, CancelScope()).process(request(config)))
    terminal = results[-1]
    assert isinstance(terminal, EndOfResponse)
    if expired:
        assert "GPU lease deadline" in terminal.error
        assert not terminal.history.complete
    else:
        assert terminal.error is None
        assert terminal.history.complete
        assert "".join(chunk.text for chunk in results if isinstance(chunk, LLMResponseChunk)) == "The answer is 42."
