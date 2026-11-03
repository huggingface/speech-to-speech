"""Native conversation handling with streamed ZeroGPU token generation."""

from threading import Event

from speech_to_speech.LLM.language_model import BaseLanguageModelHandler
from speech_to_speech.zerogpu import runtime


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
        aborted = Event()

        def cancelled():
            if aborted.is_set():
                ctx.cancelled = True
                return True
            return self._check_stop(gen, ctx)

        source = runtime.tokens(messages, self.gen_kwargs, cancelled=cancelled)
        if ctx.prefetch_transaction is not None:
            ctx.prefetch_transaction.register_abort(aborted.set)
        try:
            yield from self._stream_tokens(source, gen, language_code, ctx, runtime_config, response)
        finally:
            source.close()
