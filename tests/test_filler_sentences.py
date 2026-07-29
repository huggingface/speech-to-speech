import time
from unittest.mock import MagicMock

import pytest

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import make_user_message
from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler
from speech_to_speech.LLM.utils import run_generator_with_filler_sentences
from speech_to_speech.pipeline.messages import EndOfResponse, GenerateResponseRequest, LLMResponseChunk



def test_filler_sentences_disabled_by_default():
    def slow_gen():
        time.sleep(0.15)
        yield LLMResponseChunk(text="Hello world!")

    chunks = list(
        run_generator_with_filler_sentences(
            gen_fn=slow_gen,
            enable_filler_sentences=False,
            filler_sentence_delay_s=0.05,
            filler_sentences=["Thinking..."],
            language_code="en",
            runtime_config=None,
            response=None,
            turn_id="t1",
            turn_revision=1,
            speech_stopped_at_s=None,
            gen=0,
        )
    )

    assert len(chunks) == 1
    assert chunks[0].text == "Hello world!"


def test_filler_sentence_emitted_when_slow():
    def slow_gen():
        time.sleep(0.15)
        yield LLMResponseChunk(text="Here is your response.")

    chunks = list(
        run_generator_with_filler_sentences(
            gen_fn=slow_gen,
            enable_filler_sentences=True,
            filler_sentence_delay_s=0.05,
            filler_sentences=["Let me check that for you."],
            language_code="en",
            runtime_config=None,
            response=None,
            turn_id="t1",
            turn_revision=1,
            speech_stopped_at_s=None,
            gen=0,
        )
    )

    assert len(chunks) == 2
    assert chunks[0].text == "Let me check that for you."
    assert chunks[1].text == "Here is your response."


def test_no_filler_sentence_when_fast():
    def fast_gen():
        yield LLMResponseChunk(text="Immediate answer.")

    chunks = list(
        run_generator_with_filler_sentences(
            gen_fn=fast_gen,
            enable_filler_sentences=True,
            filler_sentence_delay_s=0.5,
            filler_sentences=["Thinking..."],
            language_code="en",
            runtime_config=None,
            response=None,
            turn_id="t1",
            turn_revision=1,
            speech_stopped_at_s=None,
            gen=0,
        )
    )

    assert len(chunks) == 1
    assert chunks[0].text == "Immediate answer."


def test_filler_sentence_skipped_if_stale():
    def slow_gen():
        time.sleep(0.15)
        yield LLMResponseChunk(text="Late answer.")

    chunks = list(
        run_generator_with_filler_sentences(
            gen_fn=slow_gen,
            enable_filler_sentences=True,
            filler_sentence_delay_s=0.05,
            filler_sentences=["Thinking..."],
            language_code="en",
            runtime_config=None,
            response=None,
            turn_id="t1",
            turn_revision=1,
            speech_stopped_at_s=None,
            gen=0,
            is_stale_fn=lambda: True,
        )
    )

    assert len(chunks) == 1
    assert chunks[0].text == "Late answer."


def test_handler_integration_filler_sentences(monkeypatch):
    monkeypatch.setattr(ChatCompletionsApiModelHandler, "warmup", lambda self: None)
    mock_stop_event = MagicMock()
    mock_queue_in = MagicMock()
    mock_queue_out = MagicMock()

    handler = ChatCompletionsApiModelHandler(
        stop_event=mock_stop_event,
        queue_in=mock_queue_in,
        queue_out=mock_queue_out,
        setup_kwargs={
            "api_key": "test-key",
            "enable_filler_sentences": True,
            "filler_sentence_delay_s": 0.05,
            "filler_sentences": ["Hmm, let me see."],
        },
    )



    def slow_request(*args, **kwargs):
        time.sleep(0.15)
        chunk = MagicMock()
        chunk.choices = [
            MagicMock(
                delta=MagicMock(content=" Paris is the capital.", refusal=None, tool_calls=None),
                finish_reason=None,
            )
        ]
        chunk.usage = None
        return [chunk]

    handler.client = MagicMock()
    handler.client.chat.completions.create = slow_request

    rc = RuntimeConfig()
    rc.chat.add_item(make_user_message("What is the capital of France?"))
    req = GenerateResponseRequest(runtime_config=rc, turn_id="t1", turn_revision=1)


    outputs = list(handler.process(req))

    texts = [out.text for out in outputs if isinstance(out, LLMResponseChunk)]
    assert "Hmm, let me see." in texts
    assert any("Paris is the capital." in t for t in texts)

    # Verify filler sentence was not saved to chat history
    history = rc.chat.to_transformers_chat()
    messages = [m.get("content") for m in history]
    assert "Hmm, let me see." not in messages

