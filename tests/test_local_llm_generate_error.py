from threading import Event, Lock
from types import SimpleNamespace

import pytest
from openai.types.realtime import RealtimeSessionCreateRequest

import speech_to_speech.LLM.language_model as language_model
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.language_model import LanguageModelHandler, VisionLanguageModelHandler
from speech_to_speech.pipeline.messages import EndOfResponse, GenerateResponseRequest
from speech_to_speech.pipeline.turn_latency import TurnLatencyStore


@pytest.mark.parametrize("handler_type", [LanguageModelHandler, VisionLanguageModelHandler])
def test_transformers_generate_error_ends_response(monkeypatch, handler_type):
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
    handler.turn_latency_store = TurnLatencyStore()
    handler.gen_kwargs = {}

    def generate(*args, **kwargs):
        raise ValueError("simulated generation failure")

    handler.pipe = generate
    handler.model = SimpleNamespace(generate=generate)
    if handler_type is VisionLanguageModelHandler:
        handler._prepare_vlm_inputs = lambda prepared: ({}, 1)

    chat = Chat(10)
    chat.add_item(make_user_message("Hi"))
    request = GenerateResponseRequest(
        runtime_config=RuntimeConfig(chat=chat, session=RealtimeSessionCreateRequest(type="realtime")),
    )

    outputs = list(handler.process(request))
    assert isinstance(outputs[-1], EndOfResponse)
    assert "simulated generation failure" in outputs[-1].error
