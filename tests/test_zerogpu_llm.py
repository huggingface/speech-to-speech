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
