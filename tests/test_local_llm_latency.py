from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace

import pytest
from openai.types.realtime import RealtimeSessionCreateRequest

import speech_to_speech.LLM.language_model as language_model
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.language_model import LanguageModelHandler
from speech_to_speech.pipeline.messages import GenerateResponseRequest, LLMResponseChunk
from speech_to_speech.pipeline.turn_latency import TurnLatencyStore


@pytest.mark.parametrize("native_template", [False, True])
@pytest.mark.parametrize("tail", [". How are you?", ""])
def test_local_ttft_precedes_sentence_batching_and_trailing_reply(monkeypatch, tail, native_template):
    clock = [10.0]

    def tokens(*args, **kwargs):
        yield SimpleNamespace(text=" ")
        clock[0] = 10.05
        yield SimpleNamespace(text="Hello")
        clock[0] = 10.25
        if tail:
            yield SimpleNamespace(text=tail)

    monkeypatch.setattr(language_model, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(language_model, "mlx_stream_generate", tokens, raising=False)
    monkeypatch.setattr(language_model, "MLXLockContext", lambda **kwargs: nullcontext(True), raising=False)
    monkeypatch.setattr(language_model, "mx", SimpleNamespace(clear_cache=lambda: None), raising=False)

    def unexpected_torch_cleanup():
        pytest.fail("MLX generation must not call PyTorch MPS cleanup")

    monkeypatch.setattr(language_model.torch.mps, "empty_cache", unexpected_torch_cleanup)
    handler = object.__new__(LanguageModelHandler)
    handler.backend = "mlx"
    handler.model = object()
    templates = []

    def apply_template(messages, tokenize, **kwargs):
        templates.append((messages, kwargs))
        return [1] if tokenize else "prompt"

    handler.tokenizer = SimpleNamespace(
        chat_template="<tool_call><function=example>" if native_template else "plain template",
        apply_chat_template=apply_template,
        encode=lambda text: [],
    )
    handler.gen_kwargs = {"max_new_tokens": 16}
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.stop_event = Event()
    handler.enable_lang_prompt = False
    handler.compactor = None
    handler.stream_batch_sentences = 1
    store = handler.turn_latency_store = TurnLatencyStore()
    chat = Chat(10)
    chat.add_item(make_user_message("Hi"))
    request = GenerateResponseRequest(
        runtime_config=RuntimeConfig(
            chat=chat,
            session=RealtimeSessionCreateRequest(
                type="realtime",
                tools=[{"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {}}}],
            ),
        ),
        turn_id="turn_1",
        turn_revision=0,
    )
    tracker = store.get_or_create_response(request.response_key, turn_id="turn_1", turn_revision=0)

    outputs = list(handler.process(request))

    assert any(isinstance(output, LLMResponseChunk) and "Hello" in output.text for output in outputs)
    assert tracker.llm_ttft_s == pytest.approx(0.05)
    assert tracker.llm_s == pytest.approx(0.25)
    assert tracker.llm_total_s == pytest.approx(0.25)
    assert tracker.llm_rounds == 1
    for messages, kwargs in templates:
        if native_template:
            assert kwargs["tools"][0]["function"]["name"] == "lookup"
            assert "<code>" not in messages[0]["content"]
        else:
            assert "tools" not in kwargs
            assert "<code>" in messages[0]["content"]


def test_transformers_stops_at_chat_end_before_generating_another_role():
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    from speech_to_speech.LLM.language_model import _set_chat_eos

    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=8, n_embd=8, n_layer=1, n_head=1, bos_token_id=0, eos_token_id=2, pad_token_id=2)
    ).eval()
    prompt = torch.tensor([[0]])
    # A completed assistant turn (3, 1), then an invented user turn (4, 2).
    continuation = [3, 1, 4, 2]
    kwargs = dict(
        max_new_tokens=4,
        do_sample=False,
        attention_mask=torch.ones_like(prompt),
        prefix_allowed_tokens_fn=lambda _, ids: [continuation[len(ids) - 1]],
    )
    assert model.generate(prompt, **kwargs).tolist() == [[0, 3, 1, 4, 2]]
    _set_chat_eos(model, SimpleNamespace(eos_token_id=1))
    assert model.generate(prompt, **kwargs).tolist() == [[0, 3, 1]]
    # Explicit generation overrides still take precedence.
    assert model.generate(prompt, eos_token_id=2, **kwargs).tolist() == [[0, 3, 1, 4, 2]]
