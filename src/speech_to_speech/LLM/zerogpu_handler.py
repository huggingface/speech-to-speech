"""Native conversation handling with streamed ZeroGPU token generation."""

from queue import Queue
from threading import Event, Thread
from uuid import uuid4

from speech_to_speech.LLM.language_model import BaseLanguageModelHandler
from speech_to_speech.zerogpu import runtime


class _RemoteTokens:
    """Pollable token stream so cancellation works even while waiting for a GPU."""

    def __init__(self, messages, kwargs):
        self.ident = uuid4().hex
        self.queue = Queue()
        self.done = Event()
        self.closed = Event()
        self.thread = Thread(target=self._produce, args=(messages, kwargs), daemon=True)
        self.thread.start()

    def _produce(self, messages, kwargs):
        try:
            for text in runtime.tokens(messages, kwargs, self.ident):
                if self.closed.is_set():
                    runtime.cancel_generation(self.ident)
                self.queue.put(("text", text))
        except BaseException as exc:
            self.queue.put(("error", exc))
        finally:
            self.done.set()
            runtime.cleanup_cancel(self.ident)
            self.queue.put(("end", None))

    def __iter__(self):
        return self

    def __next__(self):
        kind, value = self.queue.get(timeout=0.1)
        if kind == "end":
            raise StopIteration
        if kind == "error":
            raise value
        return value

    def close(self):
        self.closed.set()
        if not self.done.is_set():
            runtime.cancel_generation(self.ident)


class ZeroGPULLMHandler(BaseLanguageModelHandler):
    def setup(self, compact_history=False, **kwargs):
        if compact_history:
            raise ValueError("ZeroGPU background history compaction is not supported yet.")
        # The parent only formats prompts; inference always runs on worker CUDA.
        kwargs["device"] = "cpu"
        super().setup(compact_history=False, **kwargs)

    def _load_model(self, model_name, device, torch_dtype, gen_kwargs):
        # Token counting and prompt formatting are CPU operations.
        if model_name != runtime.get_model_ids()["llm"]:
            raise ValueError("ZeroGPU LLM model must match runtime.initialize(llm_model=...).")
        self.tokenizer = runtime.get_tokenizer()
        self.gen_kwargs = dict(gen_kwargs)

    def _generate(self, chat, language_code, gen, ctx, runtime_config=None, response=None):
        messages = chat.to_transformers_chat()
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, enable_thinking=False)
        ctx.input_tokens += len(ids if isinstance(ids, list) else ids["input_ids"])
        source = _RemoteTokens(messages, self.gen_kwargs)
        if ctx.prefetch_transaction is not None:
            ctx.prefetch_transaction.register_abort(source.close)
        try:
            yield from self._stream_tokens(source, gen, language_code, ctx, runtime_config, response)
        finally:
            source.close()
