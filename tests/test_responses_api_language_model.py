import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from openai import Stream
from openai.types.realtime.conversation_item import (
    RealtimeConversationItemAssistantMessage,
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

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.base_openai_compatible_language_model import WARMUP_MAX_RETRIES
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.responses_api_language_model import ResponsesApiModelHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import EndOfResponse, GenerateResponseRequest, LLMResponseChunk, TokenUsage


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
    return handler


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

    outputs = list(handler.process(_make_request("What do you see?")))

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


def test_responses_api_timing_logs_only_text_chunks():
    handler = object.__new__(ResponsesApiModelHandler)
    handler._times = [0.01]

    assert handler.timing_log_level == logging.INFO
    assert handler.should_log_timing(LLMResponseChunk(text="Hello."))
    assert not handler.should_log_timing(TokenUsage(input_tokens=1, output_tokens=1))
    assert not handler.should_log_timing(EndOfResponse())


def test_process_read_timeout_ends_response_cleanly():
    handler = _make_handler()

    def make_timeout_stream():
        stream = MagicMock(spec=Stream)
        stream.__iter__.side_effect = httpx.ReadTimeout("timed out")
        return stream

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: make_timeout_stream()))

    outputs = list(handler.process(_make_request("Hi")))

    assert len(outputs) == 2
    assert (
        isinstance(outputs[0], LLMResponseChunk)
        and outputs[0].text == "Wow I'm a bit slow today, could you repeat that?"
    )
    assert isinstance(outputs[1], EndOfResponse)


def test_generation_error_emits_failed_end_of_response():
    """A non-timeout failure (e.g. provider rejecting empty input) must still emit a
    terminating EndOfResponse carrying the error, so the response is closed instead
    of escaping process() and locking st.in_response forever."""
    handler = _make_handler()

    def boom(**kwargs):
        raise RuntimeError("input must not be empty")

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=boom))

    outputs = list(handler.process(_make_request("Hi")))

    eors = [o for o in outputs if isinstance(o, EndOfResponse)]
    assert len(eors) == 1
    assert eors[0].error is not None
    assert "input must not be empty" in eors[0].error
    # No partial output committed; the only thing emitted is the failed EndOfResponse.
    assert all(isinstance(o, EndOfResponse) for o in outputs)


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
    handler = _make_handler()
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
