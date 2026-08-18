import json
import logging
from queue import Queue
from threading import BoundedSemaphore, Event, Lock, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
from openai import Stream
from openai.types.realtime.conversation_item import (
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
)
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_output_text import ResponseOutputText

import speech_to_speech.LLM.base_openai_compatible_language_model as base_openai_compatible_language_model
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.base_openai_compatible_language_model import WARMUP_MAX_RETRIES, ToolCall, Usage
from speech_to_speech.LLM.chat import (
    AUDIO_INPUT_HISTORY_PLACEHOLDER,
    Chat,
    make_user_message,
)
from speech_to_speech.LLM.responses_api_language_model import ResponsesApiModelHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import (
    EndOfResponse,
    GenerateResponseRequest,
    LLMResponseChunk,
    ResponsePrefetchTransaction,
    TokenUsage,
)


def _make_text_delta_event(text):
    evt = MagicMock(spec=ResponseTextDeltaEvent)
    evt.type = "response.output_text.delta"
    evt.delta = text
    return evt


def _make_output_item_done_event(role="assistant", content="Hello.", item_type="message"):
    evt = MagicMock(spec=ResponseOutputItemDoneEvent)
    evt.type = "response.output_item.done"
    if item_type == "function_call":
        evt.item = SimpleNamespace(
            type="function_call",
            model_dump=lambda: {"type": "function_call", "name": "test_fn"},
        )
    else:
        evt.item = SimpleNamespace(
            type="message",
            role=role,
            content=content,
        )
    return evt


def _make_stream(events):
    stream = MagicMock(spec=Stream)
    stream.__iter__.return_value = iter(events)
    return stream


def _make_function_call_done_event(name="camera", arguments="{}"):
    return ResponseOutputItemDoneEvent(
        type="response.output_item.done",
        output_index=1,
        sequence_number=2,
        item=ResponseFunctionToolCall(
            type="function_call",
            call_id="call_original",
            name=name,
            arguments=arguments,
        ),
    )


def _make_response(output, usage=None):
    resp = MagicMock(spec=Response)
    resp.usage = usage
    resp.output = output
    return resp


def _make_runtime_config(chat_size=2, instructions="You are a helpful AI assistant."):
    from openai.types.realtime import RealtimeSessionCreateRequest

    return RuntimeConfig(
        chat=Chat(chat_size),
        session=RealtimeSessionCreateRequest(type="realtime", instructions=instructions),
    )


def _make_request(text="Hi", chat_size=2):
    cfg = _make_runtime_config(chat_size=chat_size)
    cfg.chat.add_item(make_user_message(text))
    return GenerateResponseRequest(runtime_config=cfg)


def _make_audio_request(cfg=None):
    return GenerateResponseRequest(
        runtime_config=cfg or _make_runtime_config(chat_size=5),
        audio=np.zeros(1600, dtype=np.float32),
        audio_sample_rate=16000,
    )


def _chat_chunk(*, content=None, tool_calls=None, usage=None, refusal=None):
    choices = []
    if content is not None or tool_calls is not None or refusal is not None:
        choices = [
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=tool_calls, refusal=refusal),
                finish_reason=None,
            )
        ]
    return SimpleNamespace(choices=choices, usage=usage)


def _chat_tool_delta(index, *, tool_id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index,
        id=tool_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _make_handler(*, disable_thinking=False, stream=True, cancel_scope=None):
    handler = object.__new__(ResponsesApiModelHandler)
    handler.model_name = "test-model"
    handler.stream = stream
    handler.stream_batch_sentences = 1
    handler.gen_kwargs = {}
    handler.request_timeout_s = 20.0
    handler.request_timeout = 20.0
    handler.disable_thinking = disable_thinking
    handler._extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if disable_thinking else None
    handler.user_role = "user"
    handler.cancel_scope = cancel_scope
    handler.speculative_turns = None
    handler.tools = None
    handler.tools_choice = None
    handler.enable_lang_prompt = False
    handler.compactor = None
    handler.audio_max_tokens = 80
    handler.audio_temperature = 0.0
    handler.audio_content_type = "input_audio"
    handler.audio_history_turns = 1
    handler._prefetch_worker_slots = BoundedSemaphore(
        base_openai_compatible_language_model.PREFETCH_PROVIDER_WORKER_LIMIT
    )
    handler._prefetch_workers_lock = Lock()
    handler._prefetch_workers = set()
    return handler


def test_discarded_prefetch_aborts_blocked_provider_stream():
    class BlockingStream:
        def __init__(self):
            self.started = Event()
            self.released = Event()
            self.closed = False

        def __iter__(self):
            self.started.set()
            self.released.wait(timeout=2.0)
            return iter(())

        def close(self):
            self.closed = True
            self.released.set()

    cancel_scope = CancelScope()
    handler = _make_handler(stream=True, cancel_scope=cancel_scope)
    stream = BlockingStream()
    handler.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: stream))
    request = _make_request()
    transaction = ResponsePrefetchTransaction()
    request.prefetch_transaction = transaction
    outputs: list[object] = []
    worker = Thread(target=lambda: outputs.extend(handler.process(request)))

    worker.start()
    assert stream.started.wait(timeout=1.0)
    transaction.discard()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert stream.closed
    assert cancel_scope.generation == 0


def test_failed_prefetch_is_discarded_before_terminal_is_yielded():
    handler = _make_handler(stream=True, cancel_scope=CancelScope())

    def fail_create(**kwargs):
        raise RuntimeError("provider failed")

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=fail_create))
    request = _make_request()
    transaction = ResponsePrefetchTransaction()
    request.prefetch_transaction = transaction
    generation = handler.process(request)

    terminal = next(generation)

    assert isinstance(terminal, EndOfResponse)
    assert terminal.error == "Language model generation failed: provider failed"
    assert transaction.discarded
    generation.close()


def test_claimed_failed_prefetch_emits_transactional_fallback():
    handler = _make_handler(stream=True, cancel_scope=CancelScope())

    def fail_create(**kwargs):
        raise RuntimeError("provider failed")

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=fail_create))
    request = _make_request()
    transaction = ResponsePrefetchTransaction()
    assert transaction.claim()
    request.prefetch_transaction = transaction

    outputs = list(handler.process(request))

    assert isinstance(outputs[0], LLMResponseChunk)
    assert outputs[0].text == base_openai_compatible_language_model.PROVIDER_FAILURE_FALLBACK
    assert outputs[0].prefetch_transaction is transaction
    assert isinstance(outputs[1], EndOfResponse)
    assert outputs[1].error == "Language model generation failed: provider failed"


def test_discarded_prefetch_does_not_block_on_provider_connect():
    request_started = Event()
    release_request = Event()
    response_closed = Event()

    class LateStream:
        def __iter__(self):
            return iter(())

        def close(self):
            response_closed.set()

    def create(**kwargs):
        request_started.set()
        release_request.wait(timeout=2.0)
        return LateStream()

    cancel_scope = CancelScope()
    handler = _make_handler(stream=True, cancel_scope=cancel_scope)
    handler.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    request = _make_request()
    transaction = ResponsePrefetchTransaction()
    request.prefetch_transaction = transaction
    outputs: list[object] = []
    worker = Thread(target=lambda: outputs.extend(handler.process(request)))

    worker.start()
    assert request_started.wait(timeout=1.0)
    transaction.discard()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert cancel_scope.generation == 0

    release_request.set()
    assert response_closed.wait(timeout=1.0)


def test_claimed_prefetch_cancel_does_not_block_on_provider_connect():
    request_started = Event()
    release_request = Event()
    response_closed = Event()

    class LateStream:
        def __iter__(self):
            return iter(())

        def close(self):
            response_closed.set()

    def create(**kwargs):
        request_started.set()
        release_request.wait(timeout=2.0)
        return LateStream()

    cancel_scope = CancelScope()
    handler = _make_handler(stream=True, cancel_scope=cancel_scope)
    handler.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    request = _make_request()
    transaction = ResponsePrefetchTransaction()
    request.prefetch_transaction = transaction
    outputs: list[object] = []
    worker = Thread(target=lambda: outputs.extend(handler.process(request)))

    worker.start()
    assert request_started.wait(timeout=1.0)
    transaction.claim()
    cancel_scope.cancel()
    worker.join(timeout=1.0)

    assert not worker.is_alive()

    release_request.set()
    assert response_closed.wait(timeout=1.0)


def test_claimed_prefetch_cancel_aborts_blocked_provider_read():
    class BlockingStream:
        def __init__(self):
            self.started = Event()
            self.released = Event()
            self.closed = False

        def __iter__(self):
            self.started.set()
            self.released.wait(timeout=2.0)
            return iter(())

        def close(self):
            self.closed = True
            self.released.set()

    cancel_scope = CancelScope()
    handler = _make_handler(stream=True, cancel_scope=cancel_scope)
    stream = BlockingStream()
    handler.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: stream))
    request = _make_request()
    transaction = ResponsePrefetchTransaction()
    request.prefetch_transaction = transaction
    outputs: list[object] = []
    worker = Thread(target=lambda: outputs.extend(handler.process(request)))

    worker.start()
    assert stream.started.wait(timeout=1.0)
    transaction.claim()
    cancel_scope.cancel()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert stream.closed


def test_repeated_prefetch_invalidation_bounds_provider_connect_workers():
    request_started = Event()
    release_request = Event()
    response_closed = Event()
    create_calls = 0

    class LateStream:
        def __iter__(self):
            return iter(())

        def close(self):
            response_closed.set()

    def create(**kwargs):
        nonlocal create_calls
        create_calls += 1
        request_started.set()
        release_request.wait(timeout=2.0)
        return LateStream()

    handler = _make_handler(stream=True, cancel_scope=CancelScope())
    handler.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    first_request = _make_request()
    first_transaction = ResponsePrefetchTransaction()
    first_request.prefetch_transaction = first_transaction
    first_outputs: list[object] = []
    request_worker = Thread(target=lambda: first_outputs.extend(handler.process(first_request)))

    request_worker.start()
    assert request_started.wait(timeout=1.0)
    first_transaction.discard()
    request_worker.join(timeout=1.0)
    assert not request_worker.is_alive()

    with handler._prefetch_workers_lock:
        provider_worker = next(iter(handler._prefetch_workers))

    for _ in range(5):
        request = _make_request()
        transaction = ResponsePrefetchTransaction()
        request.prefetch_transaction = transaction

        list(handler.process(request))

        assert transaction.discarded
    assert create_calls == 1
    with handler._prefetch_workers_lock:
        assert handler._prefetch_workers == {provider_worker}

    # Capacity only disables hidden speculation. Once response.create claims a
    # later prefetch, it must still execute with ordinary response semantics.
    claimed_request = _make_request()
    claimed_transaction = ResponsePrefetchTransaction()
    assert claimed_transaction.claim()
    claimed_request.prefetch_transaction = claimed_transaction
    handler.client.responses.create = lambda **kwargs: _make_stream(
        [
            _make_text_delta_event("public"),
            _make_output_item_done_event(content="public"),
        ]
    )

    claimed_outputs = list(handler.process(claimed_request))

    assert any(isinstance(output, LLMResponseChunk) and output.text == "public" for output in claimed_outputs)

    release_request.set()
    assert response_closed.wait(timeout=1.0)
    provider_worker.join(timeout=1.0)
    with handler._prefetch_workers_lock:
        assert not handler._prefetch_workers


def test_prefetch_uses_one_worker_and_one_bounded_queue(monkeypatch):
    queue_limits: list[int] = []
    worker_names: list[str] = []

    class TrackingQueue(Queue):
        def __init__(self, maxsize=0):
            queue_limits.append(maxsize)
            super().__init__(maxsize=maxsize)

    monkeypatch.setattr(base_openai_compatible_language_model, "Queue", TrackingQueue)
    handler = _make_handler(stream=True, cancel_scope=CancelScope())
    start_worker = handler._start_prefetch_worker

    def track_worker(target, *, name):
        worker_names.append(name)
        return start_worker(target, name=name)

    monkeypatch.setattr(handler, "_start_prefetch_worker", track_worker)
    handler.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: _make_stream([])))
    request = _make_request()
    request.prefetch_transaction = ResponsePrefetchTransaction()

    list(handler.process(request))

    assert worker_names == ["realtime-tool-prefetch"]
    assert queue_limits == [base_openai_compatible_language_model.PREFETCH_STREAM_QUEUE_MAXSIZE]
    assert all(limit > 0 for limit in queue_limits)
    with handler._prefetch_workers_lock:
        assert not handler._prefetch_workers
    assert handler._prefetch_worker_slots.acquire(blocking=False)
    handler._prefetch_worker_slots.release()


def test_warmup_uses_request_scoped_sdk_retries():
    handler = _make_handler()
    handler.client = MagicMock()
    handler.client.with_options.return_value = handler.client

    handler.warmup()

    handler.client.with_options.assert_called_once_with(max_retries=WARMUP_MAX_RETRIES)
    handler.client.responses.create.assert_called_once()


def test_warmup_failure_propagates_and_prevents_readiness():
    handler = _make_handler()
    handler.client = MagicMock()
    handler.client.with_options.return_value = handler.client
    handler.client.responses.create.side_effect = RuntimeError("provider unavailable")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        handler.warmup()


def test_process_streams_text_from_response_events():
    handler = _make_handler()

    streamed_events = [
        _make_text_delta_event("Hello. "),
        _make_text_delta_event("How are you?"),
        _make_output_item_done_event(content="Hello. How are you?"),
    ]

    handler.client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: _make_stream(streamed_events),
        )
    )

    outputs = list(handler.process(_make_request("Hi")))

    assert len(outputs) == 3
    assert isinstance(outputs[0], LLMResponseChunk) and outputs[0].text == "Hello."
    assert isinstance(outputs[1], LLMResponseChunk) and outputs[1].text == "How are you?"
    assert isinstance(outputs[2], EndOfResponse)


def test_text_only_streams_raw_deltas_without_sentence_trimming():
    """Text-only streams (so a new speech turn can interrupt it) and forwards each
    delta verbatim — no sent_tokenize (newlines / markdown survive) and no
    remove_unspeechable (emoji / symbols survive)."""
    handler = _make_handler(stream=True)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _make_stream(
            [
                _make_text_delta_event("# Title 🎉\n"),
                _make_text_delta_event("- one\n- two 😀\n"),
                _make_output_item_done_event(content="# Title 🎉\n- one\n- two 😀\n"),
            ]
        )

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

    cfg = _make_runtime_config()
    cfg.chat.add_item(make_user_message("Hi"))
    req = GenerateResponseRequest(
        runtime_config=cfg,
        response=RealtimeResponseCreateParams(output_modalities=["text"]),
    )

    outputs = list(handler.process(req))

    # Still streamed, not a single buffered chunk.
    assert captured["stream"] is True
    texts = [o.text for o in outputs if isinstance(o, LLMResponseChunk)]
    # Raw deltas: markdown layout AND emoji preserved (no trimming, no unspeechable filter).
    assert texts == ["# Title 🎉\n", "- one\n- two 😀\n"]
    assert "".join(texts) == "# Title 🎉\n- one\n- two 😀\n"


def test_audio_response_sentence_batches_streaming_call():
    handler = _make_handler(stream=True)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _make_stream([_make_text_delta_event("Hi."), _make_output_item_done_event(content="Hi.")])

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

    # response=None defaults to audio, so streaming is preserved.
    list(handler.process(_make_request("Hi")))

    assert captured["stream"] is True


def test_process_flushes_tool_lead_in_before_function_call_with_sentence_batching():
    handler = _make_handler()
    handler.stream_batch_sentences = 3

    streamed_events = [
        _make_text_delta_event("Let me check with my camera."),
        _make_function_call_done_event(name="camera"),
    ]

    handler.client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: _make_stream(streamed_events),
        )
    )

    request = _make_request("What do you see?")
    outputs = list(handler.process(request))

    assert len(outputs) == 3
    assert isinstance(outputs[0], LLMResponseChunk)
    assert outputs[0].text == "Let me check with my camera."
    assert outputs[0].tools == []
    assert isinstance(outputs[1], LLMResponseChunk)
    assert outputs[1].text == ""
    assert [tool.name for tool in outputs[1].tools] == ["camera"]
    assert isinstance(outputs[2], EndOfResponse)


def test_process_preserves_streamed_text_after_function_call_order():
    handler = _make_handler()
    handler.stream_batch_sentences = 3

    streamed_events = [
        _make_text_delta_event("Let me check."),
        _make_function_call_done_event(name="camera"),
        _make_text_delta_event("This may take a second."),
    ]

    handler.client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: _make_stream(streamed_events),
        )
    )

    outputs = list(handler.process(_make_request("What do you see?")))

    assert len(outputs) == 4
    assert isinstance(outputs[0], LLMResponseChunk)
    assert outputs[0].text == "Let me check."
    assert outputs[0].tools == []
    assert isinstance(outputs[1], LLMResponseChunk)
    assert outputs[1].text == ""
    assert [tool.name for tool in outputs[1].tools] == ["camera"]
    assert isinstance(outputs[2], LLMResponseChunk)
    assert outputs[2].text == "This may take a second."
    assert outputs[2].tools == []
    assert isinstance(outputs[3], EndOfResponse)


def test_audio_streaming_preserves_provider_whitespace_across_chunks():
    handler = _make_handler(stream=True)
    handler.stream_batch_sentences = 3

    handler.client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: _make_stream(
                [
                    _make_text_delta_event("Mock brain is working. I heard you say: What is "),
                    _make_text_delta_event("the weather today?"),
                    _make_output_item_done_event(
                        content="Mock brain is working. I heard you say: What is the weather today?"
                    ),
                ]
            )
        )
    )

    outputs = [o.text for o in handler.process(_make_request("Ask about weather")) if isinstance(o, LLMResponseChunk)]
    text = "".join(outputs)

    assert outputs == ["Mock brain is working. I heard you say: What is the weather today?"]
    assert "working.I" not in text
    assert "isthe" not in text


def test_process_preserves_nonstreaming_text_tool_text_order():
    handler = _make_handler(stream=False)

    api_response = _make_response(
        output=[
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="completed",
                content=[ResponseOutputText(type="output_text", text="Let me check.", annotations=[])],
            ),
            ResponseFunctionToolCall(
                type="function_call",
                call_id="call_original",
                name="camera",
                arguments="{}",
            ),
            ResponseOutputMessage(
                id="msg_2",
                type="message",
                role="assistant",
                status="completed",
                content=[ResponseOutputText(type="output_text", text="This may take a second.", annotations=[])],
            ),
        ],
    )

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: api_response))

    request = _make_request("What do you see?")
    outputs = list(handler.process(request))

    assert len(outputs) == 4
    assert isinstance(outputs[0], LLMResponseChunk)
    assert outputs[0].text == "Let me check."
    assert outputs[0].tools == []
    assert isinstance(outputs[1], LLMResponseChunk)
    assert outputs[1].text == ""
    assert [tool.name for tool in outputs[1].tools] == ["camera"]
    assert isinstance(outputs[2], LLMResponseChunk)
    assert outputs[2].text == "This may take a second."
    assert outputs[2].tools == []
    assert isinstance(outputs[3], EndOfResponse)
    call_id = outputs[1].tools[0].call_id

    chat = request.runtime_config.chat
    assert [item.type for item in chat.buffer] == ["message", "message", "function_call", "message"]
    chat.add_item(
        RealtimeConversationItemFunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output="camera result",
        )
    )
    assert [item.type for item in chat.buffer] == [
        "message",
        "message",
        "function_call",
        "message",
        "function_call_output",
    ]
    assert [item["type"] for item in chat.to_responses_api_chat()] == [
        "message",
        "message",
        "function_call",
        "function_call_output",
        "message",
    ]


def test_process_handles_cancellation():
    scope = CancelScope()
    handler = _make_handler(cancel_scope=scope)

    def fake_create(**kwargs):
        scope.cancel()
        return _make_stream([_make_text_delta_event("Hello")])

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

    outputs = list(handler.process(_make_request("Hi")))

    assert len(outputs) == 1
    assert isinstance(outputs[0], EndOfResponse)


def test_cancelled_stream_emits_provider_usage_and_rolls_back_history():
    scope = CancelScope()
    handler = _make_handler(cancel_scope=scope)
    handler._iter_stream_events = lambda _: iter(
        [
            ToolCall(
                item=ResponseFunctionToolCall(
                    type="function_call",
                    call_id="call_provider",
                    id="fc_provider",
                    name="camera",
                    arguments="{}",
                )
            ),
            Usage(input_tokens=11, output_tokens=7),
        ]
    )
    handler.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: _make_stream([])))
    request = _make_request("Use a tool")
    generation = handler.process(request)

    tool_chunk = next(generation)
    assert isinstance(tool_chunk, LLMResponseChunk) and tool_chunk.tools

    scope.cancel()
    remaining = list(generation)

    usage = next(output for output in remaining if isinstance(output, TokenUsage))
    assert (usage.input_tokens, usage.output_tokens, usage.response_key) == (11, 7, request.response_key)
    assert any(isinstance(output, EndOfResponse) for output in remaining)
    assert [item.type for item in request.runtime_config.chat.buffer] == ["message"]


def test_cancelled_nonstreaming_response_emits_provider_usage():
    scope = CancelScope()
    handler = _make_handler(stream=False, cancel_scope=scope)

    def fake_create(**kwargs):
        scope.cancel()
        return _make_response(
            [
                ResponseOutputMessage(
                    id="msg_1",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[ResponseOutputText(type="output_text", text="unseen", annotations=[])],
                )
            ],
            usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        )

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))
    request = _make_request("Hi")

    outputs = list(handler.process(request))

    assert not any(isinstance(output, LLMResponseChunk) for output in outputs)
    usage = next(output for output in outputs if isinstance(output, TokenUsage))
    assert (usage.input_tokens, usage.output_tokens) == (11, 7)
    assert isinstance(outputs[-1], EndOfResponse)
    assert [item.type for item in request.runtime_config.chat.buffer] == ["message"]


def test_cancelled_text_tool_turn_rolls_back_ordered_call():
    scope = CancelScope()
    handler = _make_handler(cancel_scope=scope)
    handler.client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **kwargs: _make_stream([_make_function_call_done_event()]))
    )
    request = _make_request("Use a tool")
    generation = handler.process(request)

    tool_chunk = next(generation)
    assert isinstance(tool_chunk, LLMResponseChunk)
    assert tool_chunk.tools
    assert request.runtime_config.chat.has_pending_tool_calls()

    scope.cancel()
    remaining = list(generation)

    assert any(isinstance(output, EndOfResponse) for output in remaining)
    assert [item.type for item in request.runtime_config.chat.buffer] == ["message"]
    assert not request.runtime_config.chat.has_pending_tool_calls()


def test_provider_error_after_text_tool_rolls_back_ordered_call():
    handler = _make_handler()

    class FailingStream:
        def __iter__(self):
            yield _make_function_call_done_event()
            raise RuntimeError("provider stream failed")

        def close(self):
            pass

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: FailingStream()))
    request = _make_request("Use a tool")
    outputs = list(handler.process(request))

    end = next(output for output in outputs if isinstance(output, EndOfResponse))
    assert end.error is not None and "provider stream failed" in end.error
    assert [item.type for item in request.runtime_config.chat.buffer] == ["message"]
    assert not request.runtime_config.chat.has_pending_tool_calls()


def test_generation_is_rejected_until_ordered_tool_output_arrives():
    handler = _make_handler()
    called = False

    def create(**kwargs):
        nonlocal called
        called = True
        return _make_stream([])

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    chat = Chat(5)
    chat.add_item(make_user_message("first"))
    chat.add_ordered_function_call(
        RealtimeConversationItemFunctionCall(
            type="function_call",
            call_id="call_pending",
            name="pending",
            arguments="{}",
        )
    )
    chat.add_item(make_user_message("second"))
    request = GenerateResponseRequest(runtime_config=RuntimeConfig(chat=chat))

    outputs = list(handler.process(request))

    assert called is False
    assert len(outputs) == 1
    assert isinstance(outputs[0], EndOfResponse)
    assert outputs[0].error is not None and "function call outputs are pending" in outputs[0].error
    assert [item.type for item in chat.buffer] == ["message", "function_call", "message"]


def test_responses_api_timing_logs_only_text_chunks():
    handler = object.__new__(ResponsesApiModelHandler)
    handler._times = [0.01]

    assert handler.timing_log_level == logging.INFO
    assert handler.should_log_timing(LLMResponseChunk(text="Hello."))
    assert not handler.should_log_timing(TokenUsage(input_tokens=1, output_tokens=1))
    assert not handler.should_log_timing(EndOfResponse())


def test_setup_uses_dummy_api_key_for_custom_base_url(monkeypatch):
    captured = {}
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class FakeOpenAI:
        def __init__(self, *, api_key, base_url):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr(base_openai_compatible_language_model, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(ResponsesApiModelHandler, "warmup", lambda self: None)

    handler = object.__new__(ResponsesApiModelHandler)
    handler.setup(base_url="http://127.0.0.1:8080/v1", api_key=None, compact_history=False)

    assert captured == {
        "api_key": "none",
        "base_url": "http://127.0.0.1:8080/v1",
    }


def test_setup_preserves_environment_api_key_for_custom_base_url(monkeypatch):
    captured = {}
    monkeypatch.setenv("OPENAI_API_KEY", "secret-from-environment")

    class FakeOpenAI:
        def __init__(self, *, api_key, base_url):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr(base_openai_compatible_language_model, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(ResponsesApiModelHandler, "warmup", lambda self: None)

    handler = object.__new__(ResponsesApiModelHandler)
    handler.setup(base_url="http://127.0.0.1:8080/v1", api_key=None, compact_history=False)

    assert captured == {
        "api_key": None,
        "base_url": "http://127.0.0.1:8080/v1",
    }


def test_setup_does_not_inject_dummy_key_for_remote_custom_url(monkeypatch):
    captured = {}
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    class FakeOpenAI:
        def __init__(self, *, api_key, base_url):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr(base_openai_compatible_language_model, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(ResponsesApiModelHandler, "warmup", lambda self: None)

    handler = object.__new__(ResponsesApiModelHandler)
    handler.setup(base_url="https://provider.example/v1", api_key=None, compact_history=False)

    assert captured == {
        "api_key": None,
        "base_url": "https://provider.example/v1",
    }


def test_setup_resolves_gemini_api_key_and_endpoint(monkeypatch):
    captured = {}
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret-123")

    class FakeOpenAI:
        def __init__(self, *, api_key, base_url):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr(base_openai_compatible_language_model, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(ResponsesApiModelHandler, "warmup", lambda self: None)

    handler = object.__new__(ResponsesApiModelHandler)
    handler.setup(model_name="gemini-2.5-flash", base_url=None, api_key=None, compact_history=False)

    assert captured == {
        "api_key": "gemini-secret-123",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    }


def test_process_read_timeout_speaks_fallback_and_preserves_failure():
    handler = _make_handler()

    def make_timeout_stream():
        stream = MagicMock(spec=Stream)
        stream.__iter__.side_effect = httpx.ReadTimeout("timed out")
        return stream

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: make_timeout_stream()))

    request = _make_request("Hi")
    outputs = list(handler.process(request))

    assert len(outputs) == 2
    assert isinstance(outputs[0], LLMResponseChunk)
    assert outputs[0].text == base_openai_compatible_language_model.PROVIDER_FAILURE_FALLBACK
    assert isinstance(outputs[1], EndOfResponse)
    assert outputs[1].error is not None and "timed out" in outputs[1].error
    assert [getattr(item, "role", None) for item in request.runtime_config.chat.buffer] == ["user"]


def test_read_timeout_after_partial_text_fails_without_apology_or_history_commit():
    handler = _make_handler()

    class PartialThenTimeoutStream:
        def __iter__(self):
            yield _make_text_delta_event("Partial answer. ")
            raise httpx.ReadTimeout("timed out")

        def close(self):
            pass

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: PartialThenTimeoutStream()))
    request = _make_request("Hi")
    request.response = RealtimeResponseCreateParams(output_modalities=["text"])

    outputs = list(handler.process(request))

    chunks = [output.text for output in outputs if isinstance(output, LLMResponseChunk)]
    assert chunks == ["Partial answer. "]
    end = next(output for output in outputs if isinstance(output, EndOfResponse))
    assert end.error is not None and "timed out" in end.error
    assert [item.type for item in request.runtime_config.chat.buffer] == ["message"]


def test_read_timeout_after_tool_call_fails_and_rolls_back_call():
    handler = _make_handler()

    class ToolThenTimeoutStream:
        def __iter__(self):
            yield _make_function_call_done_event()
            raise httpx.ReadTimeout("timed out")

        def close(self):
            pass

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: ToolThenTimeoutStream()))
    request = _make_request("Use a tool")

    outputs = list(handler.process(request))

    assert isinstance(outputs[0], LLMResponseChunk)
    assert outputs[0].tools
    assert not any(isinstance(output, LLMResponseChunk) and "slow today" in output.text for output in outputs)
    end = next(output for output in outputs if isinstance(output, EndOfResponse))
    assert end.error is not None and "timed out" in end.error
    assert [item.type for item in request.runtime_config.chat.buffer] == ["message"]
    assert not request.runtime_config.chat.has_pending_tool_calls()


def test_generation_error_emits_failed_end_of_response():
    """A non-timeout failure (e.g. provider rejecting empty input) must still emit a
    terminating EndOfResponse carrying the error, so the response is closed instead
    of escaping process() and locking st.in_response forever."""
    handler = _make_handler()

    def boom(**kwargs):
        raise RuntimeError("input must not be empty")

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=boom))

    request = _make_request("Hi")
    outputs = list(handler.process(request))

    eors = [o for o in outputs if isinstance(o, EndOfResponse)]
    assert len(eors) == 1
    assert eors[0].error is not None
    assert "input must not be empty" in eors[0].error
    chunks = [o for o in outputs if isinstance(o, LLMResponseChunk)]
    assert [chunk.text for chunk in chunks] == [base_openai_compatible_language_model.PROVIDER_FAILURE_FALLBACK]
    assert [getattr(item, "role", None) for item in request.runtime_config.chat.buffer] == ["user"]


def test_empty_context_fails_with_clear_message_without_calling_provider():
    """Out-of-band, text-only, empty `instructions`, input=[] -> empty context. We
    fail fast with a clear, instructions-aware message and never call the provider
    (which would reject the empty input), so the response terminates instead of
    hanging."""
    handler = _make_handler()
    called = False

    def fake_create(**kwargs):
        nonlocal called
        called = True
        return _make_response(output=[])

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

    cfg = _make_runtime_config(instructions="")  # empty instructions -> no system message
    req = GenerateResponseRequest(
        runtime_config=cfg,
        response=RealtimeResponseCreateParams(
            conversation="none",
            output_modalities=["text"],
            input=[],
        ),
    )

    outputs = list(handler.process(req))

    assert not called  # short-circuited before reaching the provider
    eors = [o for o in outputs if isinstance(o, EndOfResponse)]
    assert len(eors) == 1
    assert eors[0].error is not None
    assert "instructions" in eors[0].error
    assert "input" in eors[0].error


def test_disable_thinking_passes_extra_body():
    handler = _make_handler(disable_thinking=True)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _make_stream(
            [
                _make_text_delta_event("Ok"),
                _make_output_item_done_event(content="Ok"),
            ]
        )

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

    list(handler.process(_make_request("Hi")))

    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_no_disable_thinking_omits_extra_body():
    handler = _make_handler(disable_thinking=False)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _make_stream(
            [
                _make_text_delta_event("Ok"),
                _make_output_item_done_event(content="Ok"),
            ]
        )

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

    list(handler.process(_make_request("Hi")))

    assert captured.get("extra_body") is None


def test_second_turn_flattens_assistant_history_for_responses():
    handler = _make_handler(stream=False)
    captured = {}
    cfg = _make_runtime_config(chat_size=2)

    first_response = _make_response(
        output=[
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="completed",
                content=[ResponseOutputText(type="output_text", text="Hello.", annotations=[])],
            )
        ],
    )
    second_response = _make_response(output=[])
    call_count = 0

    def fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return first_response
        captured.update(kwargs)
        return second_response

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

    cfg.chat.add_item(make_user_message("Hi"))
    list(handler.process(GenerateResponseRequest(runtime_config=cfg)))
    cfg.chat.add_item(make_user_message("Again"))
    list(handler.process(GenerateResponseRequest(runtime_config=cfg)))

    assistant_items = [item for item in captured["input"] if item.get("role") == "assistant"]
    assert len(assistant_items) == 1
    ai = assistant_items[0]
    assert ai["role"] == "assistant"
    assert ai["type"] == "message"
    assert ai["status"] == "completed"
    assert len(ai["content"]) == 1
    assert ai["content"][0]["type"] == "output_text"
    assert ai["content"][0]["text"] == "Hello."


def test_audio_request_uses_chat_completions_input_audio_payload():
    handler = _make_handler(stream=False)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Yes, I heard you.", refusal=None, tool_calls=[]),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5),
        )

    handler.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)),
    )

    outputs = list(handler.process(_make_audio_request()))

    assert isinstance(outputs[0], LLMResponseChunk)
    assert outputs[0].text == "Yes, I heard you."
    assert isinstance(outputs[1], TokenUsage)
    assert outputs[1].input_tokens == 12
    assert outputs[1].output_tokens == 5
    assert isinstance(outputs[2], EndOfResponse)

    assert captured["model"] == "test-model"
    assert captured["stream"] is False
    assert captured["max_tokens"] == 80
    user_messages = [m for m in captured["messages"] if m["role"] == "user"]
    assert user_messages[-1]["content"][0]["type"] == "input_audio"
    audio_part = user_messages[-1]["content"][0]["input_audio"]
    assert audio_part["format"] == "wav"
    assert audio_part["data"]


def test_audio_second_turn_retains_recent_audio_then_compacts_older_turn():
    handler = _make_handler(stream=False)
    captured_calls = []
    responses = iter(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="First answer.", refusal=None, tool_calls=[]),
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Second answer.", refusal=None, tool_calls=[]),
                    )
                ],
                usage=None,
            ),
        ]
    )

    def fake_create(**kwargs):
        captured_calls.append(kwargs)
        return next(responses)

    handler.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    cfg = _make_runtime_config(chat_size=5)

    list(handler.process(_make_audio_request(cfg)))
    list(handler.process(_make_audio_request(cfg)))

    second_messages = captured_calls[1]["messages"]
    assert [message["role"] for message in second_messages] == ["system", "user", "assistant", "user"]
    assert second_messages[1]["content"][0]["type"] == "input_audio"
    assert second_messages[2] == {"role": "assistant", "content": "First answer."}
    assert second_messages[3]["content"][0]["type"] == "input_audio"
    assert cfg.chat._user_turn_count == 2
    stored_users = [item for item in cfg.chat.buffer if getattr(item, "role", None) == "user"]
    assert stored_users[0].content[0].text == AUDIO_INPUT_HISTORY_PLACEHOLDER
    assert stored_users[1].content[0].type == "input_audio"


def test_audio_nonstreaming_tool_call_uses_chat_protocol_and_survives_next_turn():
    handler = _make_handler(stream=False)
    captured_calls = []
    first_tool = SimpleNamespace(
        id="server_call",
        function=SimpleNamespace(name="lookup", arguments='{"q": "weather"}'),
    )
    responses = iter(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=None, refusal=None, tool_calls=[first_tool]),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="It is sunny.", refusal=None, tool_calls=[]),
                    )
                ],
                usage=None,
            ),
        ]
    )

    def fake_create(**kwargs):
        captured_calls.append(kwargs)
        return next(responses)

    handler.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    cfg = _make_runtime_config(chat_size=5)
    cfg.session.tools = [
        {
            "type": "function",
            "name": "lookup",
            "description": "Look something up",
            "parameters": {"type": "object"},
        }
    ]
    cfg.session.tool_choice = {"type": "function", "name": "lookup"}

    first_outputs = list(handler.process(_make_audio_request(cfg)))
    emitted_tools = [tool for output in first_outputs if isinstance(output, LLMResponseChunk) for tool in output.tools]
    assert len(emitted_tools) == 1
    emitted_tool = emitted_tools[0]
    assert emitted_tool.name == "lookup"
    assert json.loads(emitted_tool.arguments) == {"q": "weather"}
    assert captured_calls[0]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look something up",
                "parameters": {"type": "object"},
            },
        }
    ]
    assert captured_calls[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "lookup"},
    }

    cfg.chat.add_item(
        RealtimeConversationItemFunctionCallOutput(
            type="function_call_output",
            call_id=emitted_tool.call_id,
            output='{"temperature": 22}',
        )
    )
    list(handler.process(_make_audio_request(cfg)))

    second_messages = captured_calls[1]["messages"]
    assert [message["role"] for message in second_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
    ]
    tool_call_message = second_messages[2]
    assert tool_call_message["tool_calls"][0]["function"]["name"] == "lookup"
    assert tool_call_message["tool_calls"][0]["function"]["arguments"] == '{"q": "weather"}'
    assert second_messages[3] == {
        "role": "tool",
        "tool_call_id": emitted_tool.call_id,
        "content": '{"temperature": 22}',
    }
    assert second_messages[4]["content"][0]["type"] == "input_audio"


def test_audio_streaming_reuses_tool_parser_and_emits_trailing_usage():
    handler = _make_handler(stream=True)
    captured = {}
    stream = _make_stream(
        [
            _chat_chunk(content="Let me check."),
            _chat_chunk(
                tool_calls=[
                    _chat_tool_delta(
                        0,
                        tool_id="server_call",
                        name="lookup",
                        arguments='{"q"',
                    )
                ]
            ),
            _chat_chunk(tool_calls=[_chat_tool_delta(0, arguments=': "weather"}')]),
            _chat_chunk(usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5)),
        ]
    )

    def fake_create(**kwargs):
        captured.update(kwargs)
        return stream

    handler.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    cfg = _make_runtime_config(chat_size=5)
    cfg.session.tools = [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}]

    outputs = list(handler.process(_make_audio_request(cfg)))

    chunks = [output for output in outputs if isinstance(output, LLMResponseChunk)]
    usage = [output for output in outputs if isinstance(output, TokenUsage)]
    assert any(chunk.text == "Let me check." for chunk in chunks)
    assert [tool.name for chunk in chunks for tool in chunk.tools] == ["lookup"]
    assert len(usage) == 1
    assert (usage[0].input_tokens, usage[0].output_tokens) == (12, 5)
    assert captured["stream_options"] == {"include_usage": True}


def test_audio_nonstreaming_refusal_is_emitted_and_stored():
    handler = _make_handler(stream=False)
    handler.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=None,
                                refusal="I cannot help with that.",
                                tool_calls=[],
                            )
                        )
                    ],
                    usage=None,
                )
            )
        )
    )
    cfg = _make_runtime_config(chat_size=5)

    outputs = list(handler.process(_make_audio_request(cfg)))

    assert any(isinstance(output, LLMResponseChunk) and output.text == "I cannot help with that." for output in outputs)
    assert any(
        getattr(item, "role", None) == "assistant" and item.content[0].text == "I cannot help with that."
        for item in cfg.chat.buffer
    )


def test_failed_audio_request_rolls_back_provisional_history():
    handler = _make_handler(stream=False)
    handler.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("provider rejected audio"))
            )
        )
    )
    cfg = _make_runtime_config(chat_size=5)

    outputs = list(handler.process(_make_audio_request(cfg)))

    assert [type(output) for output in outputs] == [LLMResponseChunk, EndOfResponse]
    assert outputs[0].text == base_openai_compatible_language_model.PROVIDER_FAILURE_FALLBACK
    assert outputs[1].error is not None and "provider rejected audio" in outputs[1].error
    assert cfg.chat.buffer == []
    assert cfg.chat._user_turn_count == 0


def test_interrupted_audio_tool_turn_rolls_back_user_call_and_fast_output():
    handler = _make_handler(stream=True)
    stream = _make_stream(
        [
            _chat_chunk(
                tool_calls=[
                    _chat_tool_delta(
                        0,
                        tool_id="server_call",
                        name="lookup",
                        arguments="{}",
                    )
                ]
            )
        ]
    )
    handler.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: stream)))
    cfg = _make_runtime_config(chat_size=5)
    cfg.session.tools = [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}]

    generation = handler.process(_make_audio_request(cfg))
    tool_chunk = next(output for output in generation if isinstance(output, LLMResponseChunk) and output.tools)
    call_id = tool_chunk.tools[0].call_id
    cfg.chat.add_item(
        RealtimeConversationItemFunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output='{"result": "ok"}',
        )
    )

    generation.close()

    assert cfg.chat.buffer == []
    assert cfg.chat._pending_tool_calls == {}
    assert cfg.chat._user_turn_count == 0


# ── Out-of-band (conversation="none") responses ──────────────────────────


def _make_oob_request(input_items, *, conversation="none", chat_size=2, seed_default="Hi"):
    cfg = _make_runtime_config(chat_size=chat_size)
    if seed_default is not None:
        cfg.chat.add_item(make_user_message(seed_default))
    resp = RealtimeResponseCreateParams(conversation=conversation, input=input_items)
    return GenerateResponseRequest(runtime_config=cfg, response=resp), cfg


def _capture_create(handler, events):
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _make_stream(events)

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))
    return captured


def test_out_of_band_emits_output_but_does_not_commit_to_default_conversation():
    handler = _make_handler()
    req, cfg = _make_oob_request([make_user_message("OOB question")])
    events = [_make_text_delta_event("OOB answer."), _make_output_item_done_event(content="OOB answer.")]
    _capture_create(handler, events)

    outputs = list(handler.process(req))

    # The response is still produced and streamed back to the client...
    assert any(isinstance(o, LLMResponseChunk) and o.text == "OOB answer." for o in outputs)
    # ...but the default conversation keeps only the seeded user turn — no assistant commit.
    assert len(cfg.chat.buffer) == 1
    assert not any(isinstance(i, RealtimeConversationItemAssistantMessage) for i in cfg.chat.buffer)


def test_empty_response_overrides_disable_session_instructions_and_tools():
    handler = _make_handler()
    cfg = _make_runtime_config(instructions="SESSION INSTRUCTIONS")
    cfg.session.tools = [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}]
    cfg.chat.add_item(make_user_message("Answer without tools."))
    request = GenerateResponseRequest(
        runtime_config=cfg,
        response=RealtimeResponseCreateParams(
            instructions="",
            tools=[],
            output_modalities=["text"],
        ),
    )
    captured = _capture_create(handler, [_make_output_item_done_event(content="ok")])

    list(handler.process(request))

    assert captured["tools"] == []
    assert "SESSION INSTRUCTIONS" not in str(captured["input"])


def test_out_of_band_timeout_after_tool_call_fails_without_apology():
    handler = _make_handler()

    class ToolThenTimeoutStream:
        def __iter__(self):
            yield _make_function_call_done_event()
            raise httpx.ReadTimeout("timed out")

        def close(self):
            pass

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: ToolThenTimeoutStream()))
    request, cfg = _make_oob_request([make_user_message("OOB tool request")])

    outputs = list(handler.process(request))

    assert isinstance(outputs[0], LLMResponseChunk)
    assert outputs[0].tools
    assert not any(isinstance(output, LLMResponseChunk) and "slow today" in output.text for output in outputs)
    end = next(output for output in outputs if isinstance(output, EndOfResponse))
    assert end.error is not None and "timed out" in end.error
    assert [item.type for item in cfg.chat.buffer] == ["message"]


def test_out_of_band_input_builds_fresh_context():
    handler = _make_handler()
    req, _cfg = _make_oob_request([make_user_message("OOB question")])
    captured = _capture_create(handler, [_make_output_item_done_event(content="ok")])

    list(handler.process(req))

    serialized = str(captured["input"])
    assert "OOB question" in serialized
    assert "Hi" not in serialized  # default conversation history is excluded


def test_out_of_band_empty_input_clears_context():
    handler = _make_handler()
    req, _cfg = _make_oob_request([])
    captured = _capture_create(handler, [_make_output_item_done_event(content="ok")])

    list(handler.process(req))

    serialized = str(captured["input"])
    assert "Hi" not in serialized  # default conversation not used
    assert "helpful AI assistant" in serialized  # only the system prompt remains


def test_out_of_band_absent_input_reads_default_conversation():
    handler = _make_handler()
    req, cfg = _make_oob_request(None)
    captured = _capture_create(handler, [_make_output_item_done_event(content="ok")])

    list(handler.process(req))

    serialized = str(captured["input"])
    assert "Hi" in serialized  # default conversation used as read-only context
    # Still read-only: no assistant message committed back.
    assert len(cfg.chat.buffer) == 1


def test_out_of_band_invalid_input_emits_failed_end_of_response():
    scope = CancelScope()
    scope.cancel()
    handler = _make_handler(cancel_scope=scope)
    called = False

    def fake_create(**kwargs):
        nonlocal called
        called = True
        return _make_stream([])

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))
    # function_call_output referencing an unknown call_id fails validation.
    orphan = RealtimeConversationItemFunctionCallOutput(
        type="function_call_output", call_id="call_missing", output="{}"
    )
    req, _cfg = _make_oob_request([orphan])

    outputs = list(handler.process(req))

    assert not called  # generation never started
    assert len(outputs) == 1
    assert isinstance(outputs[0], EndOfResponse)
    assert outputs[0].error is not None
    assert outputs[0].cancel_generation == scope.generation
