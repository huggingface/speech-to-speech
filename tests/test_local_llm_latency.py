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


@pytest.mark.parametrize("tail", [". How are you?", ""])
def test_local_ttft_precedes_sentence_batching_and_trailing_reply(monkeypatch, tail):
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
    monkeypatch.setattr(language_model.torch.mps, "empty_cache", lambda: None)
    handler = object.__new__(LanguageModelHandler)
    handler.backend = "mlx"
    handler.model = object()
    handler.tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, tokenize, **kwargs: [1] if tokenize else "prompt",
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
        runtime_config=RuntimeConfig(chat=chat, session=RealtimeSessionCreateRequest(type="realtime")),
        turn_id="turn_1",
        turn_revision=0,
    )
    tracker = store.get_or_create_response(request.response_key, turn_id="turn_1", turn_revision=0)

    outputs = list(handler.process(request))

    assert any(isinstance(output, LLMResponseChunk) and "Hello" in output.text for output in outputs)
    assert tracker.llm_ttft_s == pytest.approx(0.05)
    assert tracker.llm_s == pytest.approx(0.25)
