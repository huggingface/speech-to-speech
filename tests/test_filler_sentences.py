import sys
import time
from threading import Event, Thread
from unittest.mock import MagicMock

from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import make_user_message
from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler
from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.LLM.utils import run_generator_with_filler_sentences
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import AssistantOutputEvent
from speech_to_speech.pipeline.log_context import pipeline_log_ctx
from speech_to_speech.pipeline.messages import (
    AssistantTextPart,
    GenerateResponseRequest,
    LLMResponseChunk,
    TTSInput,
)


def _get_language_model_handler_class():
    for mod in ("torch", "transformers"):
        if mod not in sys.modules:
            try:
                __import__(mod)
            except ModuleNotFoundError:
                mock = MagicMock()
                if mod == "torch":
                    class _Tensor:
                        pass
                    class _Module:
                        pass
                    mock.Tensor = _Tensor
                    mock.nn.Module = _Module
                sys.modules[mod] = mock
    from speech_to_speech.LLM.language_model import LanguageModelHandler

    return LanguageModelHandler



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
    stop_event = Event()
    mock_queue_in = MagicMock()
    mock_queue_out = MagicMock()

    handler = ChatCompletionsApiModelHandler(
        stop_event=stop_event,
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


def test_text_only_response_skips_filler_with_lm_output_processor(monkeypatch):
    """Regression test: text-only responses must bypass filler sentences and preserve clean output/history."""
    monkeypatch.setattr(ChatCompletionsApiModelHandler, "warmup", lambda self: None)
    stop_event = Event()
    mock_queue_in = MagicMock()
    mock_queue_out = MagicMock()

    handler = ChatCompletionsApiModelHandler(
        stop_event=stop_event,
        queue_in=mock_queue_in,
        queue_out=mock_queue_out,
        setup_kwargs={
            "api_key": "test-key",
            "enable_filler_sentences": True,
            "filler_sentence_delay_s": 0.05,
            "filler_sentences": ["Thinking..."],
        },
    )

    def slow_request(*args, **kwargs):
        time.sleep(0.15)
        chunk = MagicMock()
        chunk.choices = [
            MagicMock(
                delta=MagicMock(content='{"answer":42}', refusal=None, tool_calls=None),
                finish_reason="stop",
            )
        ]
        chunk.usage = None
        return [chunk]

    handler.client = MagicMock()
    handler.client.chat.completions.create = slow_request

    rc = RuntimeConfig()
    rc.chat.add_item(make_user_message("Provide JSON answer"))
    # Response explicitly requesting text-only modalities
    response_params = RealtimeResponseCreateParams(output_modalities=["text"])
    req = GenerateResponseRequest(
        runtime_config=rc,
        response=response_params,
        turn_id="turn_json",
        turn_revision=1,
    )

    outputs = list(handler.process(req))
    chunks = [out for out in outputs if isinstance(out, LLMResponseChunk)]
    texts = [c.text for c in chunks]

    # Handler output must NOT contain filler
    assert "Thinking..." not in texts
    assert any('{"answer":42}' in t for t in texts)

    # Run through LMOutputProcessor to verify client events and history
    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup(speculative_turns=None)

    processor_outputs = []
    for chunk in chunks:
        processor_outputs.extend(list(processor.process(chunk)))

    # Verify no TTS input is forwarded for text-only output
    tts_inputs = [ev for ev in processor_outputs if isinstance(ev, TTSInput)]
    assert len(tts_inputs) == 0

    # Collect assistant output text delivered to client
    client_parts = [
        part.text
        for ev in processor_outputs
        if isinstance(ev, AssistantOutputEvent)
        for part in ev.parts
        if isinstance(part, AssistantTextPart)
    ]
    full_client_text = "".join(client_parts)
    assert "Thinking..." not in full_client_text
    assert '{"answer":42}' in full_client_text

    # Conversation history stores only {"answer":42}
    history = rc.chat.to_transformers_chat()
    messages = [m.get("content") for m in history]
    assert '{"answer":42}' in messages
    assert "Thinking..." not in messages


def test_run_generator_with_filler_sentences_text_only_bypass():
    """run_generator_with_filler_sentences directly bypasses filler when response is text-only."""
    def slow_gen():
        time.sleep(0.15)
        yield LLMResponseChunk(text='{"data":1}')

    response_text_only = RealtimeResponseCreateParams(output_modalities=["text"])

    chunks = list(
        run_generator_with_filler_sentences(
            gen_fn=slow_gen,
            enable_filler_sentences=True,
            filler_sentence_delay_s=0.05,
            filler_sentences=["Thinking..."],
            language_code="en",
            runtime_config=None,
            response=response_text_only,
            turn_id="t1",
            turn_revision=1,
            speech_stopped_at_s=None,
            gen=0,
        )
    )

    assert len(chunks) == 1
    assert chunks[0].text == '{"data":1}'


def test_cancellation_blocks_filler_when_blocked_before_first_output(monkeypatch):
    """Regression test: cancel_scope.cancel() before first output must suppress stale filler emission."""
    cancel_scope = CancelScope()
    stop_event = Event()

    LanguageModelHandler = _get_language_model_handler_class()
    handler = object.__new__(LanguageModelHandler)
    handler.stop_event = stop_event
    handler.cancel_scope = cancel_scope
    handler.speculative_turns = None
    handler.enable_lang_prompt = False
    handler.enable_filler_sentences = True
    handler.filler_sentence_delay_s = 0.05
    handler.filler_sentences = ["Thinking..."]
    handler.compactor = None

    entered_generate = Event()
    unblock_generate = Event()

    def blocked_generate(chat, language_code, gen, ctx, runtime_config, response):
        entered_generate.set()
        unblock_generate.wait(2.0)
        if not handler._check_stop(gen, ctx):
            yield LLMResponseChunk(
                text="Delayed output",
                turn_id=ctx.turn_id,
                turn_revision=ctx.turn_revision,
                cancel_generation=gen,
            )

    monkeypatch.setattr(handler, "_generate", blocked_generate)
    monkeypatch.setattr(handler, "_apply_instructions", lambda *args, **kwargs: None)

    rc = RuntimeConfig()
    rc.chat.add_item(make_user_message("Hello"))
    req = GenerateResponseRequest(
        runtime_config=rc,
        turn_id="turn_cancel_test",
        turn_revision=1,
    )

    outputs = []

    def run_process():
        outputs.extend(list(handler.process(req)))

    worker = Thread(target=run_process)
    worker.start()

    try:
        # Wait until generation is blocked before first output
        assert entered_generate.wait(timeout=2.0)
        # Cancel: advances cancel_scope generation from 0 to 1
        cancel_scope.cancel()
        assert cancel_scope.generation == 1

        # Wait past the filler sentence delay threshold
        time.sleep(0.12)
    finally:
        unblock_generate.set()
        worker.join(timeout=2.0)

    # Handler must suppress filler chunk tagged with the stale generation 0
    filler_chunks = [
        out for out in outputs
        if isinstance(out, LLMResponseChunk) and "Thinking..." in out.text
    ]
    assert len(filler_chunks) == 0


def test_shutdown_suppresses_filler(monkeypatch):
    """Regression test: stop_event set before first output must suppress filler during shutdown."""
    cancel_scope = CancelScope()
    stop_event = Event()

    LanguageModelHandler = _get_language_model_handler_class()
    handler = object.__new__(LanguageModelHandler)
    handler.stop_event = stop_event
    handler.cancel_scope = cancel_scope
    handler.speculative_turns = None
    handler.enable_lang_prompt = False
    handler.enable_filler_sentences = True
    handler.filler_sentence_delay_s = 0.05
    handler.filler_sentences = ["Thinking..."]
    handler.compactor = None

    entered_generate = Event()
    unblock_generate = Event()

    def blocked_generate(chat, language_code, gen, ctx, runtime_config, response):
        entered_generate.set()
        unblock_generate.wait(2.0)
        if not handler._check_stop(gen, ctx):
            yield LLMResponseChunk(
                text="Delayed output",
                turn_id=ctx.turn_id,
                turn_revision=ctx.turn_revision,
                cancel_generation=gen,
            )

    monkeypatch.setattr(handler, "_generate", blocked_generate)
    monkeypatch.setattr(handler, "_apply_instructions", lambda *args, **kwargs: None)

    rc = RuntimeConfig()
    rc.chat.add_item(make_user_message("Hello"))
    req = GenerateResponseRequest(
        runtime_config=rc,
        turn_id="turn_shutdown_test",
        turn_revision=1,
    )

    outputs = []

    def run_process():
        outputs.extend(list(handler.process(req)))

    worker = Thread(target=run_process)
    worker.start()

    try:
        assert entered_generate.wait(timeout=2.0)
        # Signal handler shutdown
        stop_event.set()
        time.sleep(0.12)
    finally:
        unblock_generate.set()
        worker.join(timeout=2.0)

    filler_chunks = [
        out for out in outputs
        if isinstance(out, LLMResponseChunk) and "Thinking..." in out.text
    ]
    assert len(filler_chunks) == 0


def test_filler_worker_thread_preserves_logging_context():
    captured_pipeline_id = []

    def gen_fn():
        captured_pipeline_id.append(pipeline_log_ctx.get())
        yield LLMResponseChunk(text="Finished")

    token = pipeline_log_ctx.set(7)
    try:
        chunks = list(
            run_generator_with_filler_sentences(
                gen_fn=gen_fn,
                enable_filler_sentences=True,
                filler_sentence_delay_s=0.05,
                filler_sentences=["Thinking..."],
                language_code="en",
                runtime_config=None,
                response=None,
                turn_id="turn_contextvar_test",
                turn_revision=1,
            )
        )
    finally:
        pipeline_log_ctx.reset(token)

    assert captured_pipeline_id == [7]
    assert any(isinstance(c, LLMResponseChunk) and c.text == "Finished" for c in chunks)


