from contextlib import nullcontext
from threading import Event, Lock
from types import SimpleNamespace

import pytest
from openai.types.realtime import RealtimeSessionCreateRequest

import speech_to_speech.LLM.language_model as language_model
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.language_model import LanguageModelHandler, VisionLanguageModelHandler
from speech_to_speech.pipeline.messages import EndOfResponse, GenerateResponseRequest, LLMResponseChunk
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


@pytest.mark.parametrize("handler_type", [LanguageModelHandler, VisionLanguageModelHandler])
@pytest.mark.parametrize("interrupt", [None, "cancel", "shutdown"])
def test_transformers_cleanup_consumes_end_marker_once_and_reuses_streamer(monkeypatch, handler_type, interrupt):
    handler = object.__new__(handler_type)
    handler.backend = "transformers"
    handler.device = "cpu"
    handler.tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, tokenize, **kwargs: [1] if tokenize else "prompt",
        encode=lambda text: [1],
    )
    handler.streamer = language_model.TextIteratorStreamer(handler.tokenizer, timeout=0.01)
    handler._cancel_criteria = language_model._CancelCriteria()
    handler._transformers_lock = Lock()
    handler.cancel_scope = language_model.CancelScope()
    handler.speculative_turns = None
    handler.stop_event = Event()
    handler.enable_lang_prompt = False
    handler.compactor = None
    handler.stream_batch_sentences = 1
    handler.gen_kwargs = {}
    store = handler.turn_latency_store = TurnLatencyStore()
    reads_after_end = []
    consumed_end = [False]
    queue_get = handler.streamer.text_queue.get

    def get(**kwargs):
        if consumed_end[0]:
            reads_after_end.append(True)
        value = queue_get(**kwargs)
        if value == handler.streamer.stop_signal:
            consumed_end[0] = True
        return value

    monkeypatch.setattr(handler.streamer.text_queue, "get", get)
    turns = [interrupt, None]

    def generate(*args, **kwargs):
        mode = turns.pop(0)
        if mode == "cancel":
            handler.cancel_scope.cancel()
        elif mode == "shutdown":
            handler.stop_event.set()
        handler.streamer.on_finalized_text("Discarded. " if mode else "Hello. ")
        handler.streamer.on_finalized_text("Pending. " if mode else "", stream_end=True)

    handler.pipe = generate
    handler.model = SimpleNamespace(generate=generate)
    if handler_type is VisionLanguageModelHandler:
        handler._prepare_vlm_inputs = lambda prepared: ({}, 1)

    for turn in range(2):
        handler.stop_event.clear()
        consumed_end[0] = False
        chat = Chat(10)
        chat.add_item(make_user_message("Hi"))
        request = GenerateResponseRequest(
            runtime_config=RuntimeConfig(chat=chat, session=RealtimeSessionCreateRequest(type="realtime")),
        )
        tracker = store.get_or_create_response(request.response_key)
        outputs = list(handler.process(request))
        if turn == 0 and interrupt == "shutdown":
            assert not outputs
        else:
            assert isinstance(outputs[-1], EndOfResponse)
            assert outputs[-1].error is None
        assert handler._cancel_criteria(None, None)
        text = "".join(output.text for output in outputs if isinstance(output, LLMResponseChunk))
        assert "Discarded" not in text and "Pending" not in text
        if turn or interrupt is None:
            assert text == "Hello."
            assert tracker.llm_s is not None
        assert handler.streamer.text_queue.empty()
    assert not reads_after_end
