import threading
from threading import Event, Lock
from types import SimpleNamespace

import pytest
import torch
from openai.types.realtime import RealtimeSessionCreateRequest

import speech_to_speech.LLM.language_model as language_model
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.language_model import LanguageModelHandler, VisionLanguageModelHandler
from speech_to_speech.pipeline.messages import EndOfResponse, GenerateResponseRequest, LLMResponseChunk

HANDLERS = [LanguageModelHandler, VisionLanguageModelHandler]


def build_handler(handler_type, generate):
    handler = object.__new__(handler_type)
    handler.backend = "transformers"
    handler.device = "cpu"
    handler.tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, tokenize, **kwargs: [1] if tokenize else "prompt",
        encode=lambda text: [1],
        decode=lambda tokens, **kwargs: "",
    )
    handler.streamer = language_model.TextIteratorStreamer(handler.tokenizer, skip_prompt=True, timeout=0.01)
    handler._cancel_criteria = language_model._CancelCriteria()
    handler._transformers_lock = Lock()
    handler.cancel_scope = language_model.CancelScope()
    handler.speculative_turns = None
    handler.stop_event = Event()
    handler.enable_lang_prompt = False
    handler.compactor = None
    handler.stream_batch_sentences = 1
    handler.gen_kwargs = {"max_new_tokens": 16}
    handler.pipe = generate
    handler.model = SimpleNamespace(generate=generate)
    if handler_type is VisionLanguageModelHandler:
        handler._prepare_vlm_inputs = lambda prepared: ({}, 1)
    return handler


def run_turn(handler, timeout=5.0):
    """Run one response on a thread so a hang fails the test instead of blocking it."""
    chat = Chat(10)
    chat.add_item(make_user_message("Hi"))
    request = GenerateResponseRequest(
        runtime_config=RuntimeConfig(chat=chat, session=RealtimeSessionCreateRequest(type="realtime")),
    )
    outputs = []
    thread = threading.Thread(target=lambda: outputs.extend(handler.process(request)), daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "the response never finished"
    return outputs


def text_of(outputs):
    return "".join(output.text for output in outputs if isinstance(output, LLMResponseChunk))


@pytest.mark.parametrize("handler_type", HANDLERS)
def test_generate_failure_ends_the_response_with_an_error_and_the_next_turn_works(handler_type):
    calls = []

    def generate(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("temperature (=0.0) has to be a strictly positive float")
        handler.streamer.on_finalized_text("Hello. ", stream_end=True)

    handler = build_handler(handler_type, generate)

    failed = run_turn(handler)
    assert isinstance(failed[-1], EndOfResponse)
    assert "temperature (=0.0)" in failed[-1].error
    assert text_of(failed) == ""

    recovered = run_turn(handler)
    assert isinstance(recovered[-1], EndOfResponse)
    assert recovered[-1].error is None
    assert text_of(recovered) == "Hello."
    assert handler.streamer.text_queue.empty()
    assert not handler._transformers_lock.locked()


@pytest.mark.parametrize("handler_type", HANDLERS)
def test_failure_after_some_tokens_keeps_the_text_and_reports_the_error(handler_type):
    def generate(*args, **kwargs):
        handler.streamer.on_finalized_text("Hello. Wor")
        raise RuntimeError("CUDA out of memory")

    handler = build_handler(handler_type, generate)

    outputs = run_turn(handler)
    assert text_of(outputs) == "Hello."
    assert isinstance(outputs[-1], EndOfResponse)
    assert "CUDA out of memory" in outputs[-1].error
    assert handler.streamer.text_queue.empty()


@pytest.mark.parametrize("handler_type", HANDLERS)
def test_failure_after_the_response_was_cancelled_is_not_reported(handler_type):
    def generate(*args, **kwargs):
        handler.cancel_scope.cancel()
        handler.streamer.on_finalized_text("Discarded. ")
        raise RuntimeError("stopped mid step")

    handler = build_handler(handler_type, generate)

    outputs = run_turn(handler)
    assert isinstance(outputs[-1], EndOfResponse)
    assert outputs[-1].error is None
    assert "Discarded" not in text_of(outputs)
    assert handler.streamer.text_queue.empty()


@pytest.mark.parametrize("handler_type", HANDLERS)
def test_failed_generation_resets_the_streamer_to_expect_a_new_prompt(handler_type):
    """The next turn must start with the streamer skipping its prompt again."""

    def generate(*args, **kwargs):
        handler.streamer.put(torch.tensor([[1, 2, 3]]))
        assert handler.streamer.next_tokens_are_prompt is False
        raise RuntimeError("boom")

    handler = build_handler(handler_type, generate)

    assert "boom" in run_turn(handler)[-1].error
    assert handler.streamer.next_tokens_are_prompt is True


@pytest.mark.parametrize("handler_type", HANDLERS)
def test_failure_inside_the_streamer_itself_still_ends_the_response(handler_type):
    """If decoding fails, end() fails too. The reader must still be woken."""

    def generate(*args, **kwargs):
        handler.streamer.put(torch.tensor([[1, 2, 3]]))
        handler.streamer.put(torch.tensor([4]))

    handler = build_handler(handler_type, generate)

    def broken_decode(tokens, **kwargs):
        raise ValueError("cannot decode")

    handler.tokenizer.decode = broken_decode

    outputs = run_turn(handler)
    assert isinstance(outputs[-1], EndOfResponse)
    assert "cannot decode" in outputs[-1].error
