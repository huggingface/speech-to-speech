from threading import Event, Lock
from types import SimpleNamespace

from openai.types.realtime import RealtimeSessionCreateRequest

import speech_to_speech.LLM.language_model as language_model
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.language_model import LanguageModelHandler
from speech_to_speech.pipeline.messages import GenerateResponseRequest, TokenUsage
from speech_to_speech.pipeline.turn_latency import TurnLatencyStore


class ChatMLTokenizer:
    """Whitespace tokenizer with a ChatML-style template, like Qwen3."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=True, **_):
        prompt = "".join(f"<|im_start|>{m['role']} {m['content']} <|im_end|> " for m in messages)
        if add_generation_prompt:
            prompt += "<|im_start|>assistant "
            if not enable_thinking:
                prompt += "<think> </think> "
        return prompt.split() if tokenize else prompt

    def encode(self, text):
        return text.split()


def test_transformers_input_tokens_count_the_prompt_sent_to_the_model():
    handler = object.__new__(LanguageModelHandler)
    handler.backend = "transformers"
    handler.device = "cpu"
    handler.tokenizer = ChatMLTokenizer()
    handler.streamer = language_model.TextIteratorStreamer(handler.tokenizer, timeout=0.01)
    handler._cancel_criteria = language_model._CancelCriteria()
    handler._transformers_lock = Lock()
    handler.cancel_scope = language_model.CancelScope()
    handler.speculative_turns = None
    handler.stop_event = Event()
    handler.enable_lang_prompt = False
    handler.compactor = None
    handler.stream_batch_sentences = 1
    handler.gen_kwargs = {"max_new_tokens": 16}
    handler.turn_latency_store = TurnLatencyStore()
    prompts = []

    def pipe(prompt, **kwargs):
        prompts.append(prompt)
        handler.streamer.on_finalized_text("Hello there.", stream_end=True)

    handler.pipe = pipe
    handler.model = SimpleNamespace()
    chat = Chat(10)
    chat.add_item(make_user_message("Hi"))
    request = GenerateResponseRequest(
        runtime_config=RuntimeConfig(chat=chat, session=RealtimeSessionCreateRequest(type="realtime")),
    )

    usage = [output for output in handler.process(request) if isinstance(output, TokenUsage)]

    assert len(prompts) == 1
    assert len(usage) == 1
    assert usage[0].input_tokens == len(handler.tokenizer.encode(prompts[0]))
