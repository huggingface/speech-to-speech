from contextlib import contextmanager
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import speech_to_speech.LLM.language_model as lm
from speech_to_speech.context_extension import PreparedContext
from speech_to_speech.LLM.chat import Chat, make_user_message
from tests.test_local_llm_token_usage import ChatMLTokenizer


@pytest.mark.parametrize("backend", ["transformers", "mlx"])
@pytest.mark.parametrize("state", ["valid", "edited", "revoked", "revoked_during_projection"])
def test_admission_after_model_lock(monkeypatch, backend, state):
    """The fake inference boundary exercises both real backend branches without weights."""
    chat = Chat(8)
    chat.add_item(make_user_message("question"))
    retired, edited = Event(), Event()
    lock_acquired = Event()

    def current():
        if state == "revoked_during_projection" and lock_acquired.is_set():
            retired.set()
        return not (retired.is_set() or edited.is_set())

    prepared = PreparedContext(chat.context_anchor(), "private fact", current)
    ctx = lm.StreamContext(prepared_context=prepared, context_current=lambda: not retired.is_set())
    handler = object.__new__(lm.LanguageModelHandler)
    handler.backend = backend
    handler.device = "cpu"
    handler.gen_kwargs = {"max_new_tokens": 16}
    handler.model = object()
    handler.tokenizer = ChatMLTokenizer()
    handler._cancel_criteria = Mock()
    handler.streamer = Mock()
    handler._stream_tokens = lambda *args: iter(())
    handler._finish_transformers_generation = lambda thread, _: thread.join(timeout=2)
    submitted = []

    @contextmanager
    def admission(**kwargs):
        lock_acquired.set()
        if state == "revoked":
            retired.set()
        if state == "edited":
            edited.set()
        yield

    handler._transformers_lock = admission()
    handler.pipe = lambda prompt, **kwargs: submitted.append(prompt)
    monkeypatch.setattr(lm, "MLXLockContext", admission, raising=False)

    def mlx_generate(model, tokenizer, prompt, **kwargs):
        submitted.append(prompt)
        return (item for item in ())

    monkeypatch.setattr(lm, "mlx_stream_generate", mlx_generate, raising=False)
    monkeypatch.setattr(lm, "mx", SimpleNamespace(clear_cache=lambda: None), raising=False)
    monkeypatch.setattr(lm.torch.mps, "empty_cache", lambda: None)
    list(handler._generate(chat, None, None, ctx))
    if state.startswith("revoked"):
        assert not submitted and ctx.cancelled
    else:
        assert len(submitted) == 1
        assert ("private fact" in submitted[0]) == (state == "valid")
        assert ctx.input_tokens == len(handler.tokenizer.encode(submitted[0]))
    assert "private fact" not in str(chat.to_transformers_chat())
