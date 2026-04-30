from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
from openai import Stream
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_output_text import ResponseOutputText

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.openai_api_language_model import OpenApiModelHandler
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


def _make_response(output, usage=None, service_tier=None):
    resp = MagicMock(spec=Response)
    resp.usage = usage
    resp.output = output
    resp.service_tier = service_tier
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


def _make_handler(
    *,
    disable_thinking=False,
    stream=True,
    cancel_scope=None,
    service_tier=None,
    reasoning_effort=None,
    max_output_tokens=None,
):
    handler = object.__new__(OpenApiModelHandler)
    handler.model_name = "test-model"
    handler.stream = stream
    handler.stream_batch_sentences = 1
    handler.gen_kwargs = {}
    handler.service_tier = service_tier
    handler.reasoning_effort = reasoning_effort
    handler.max_output_tokens = max_output_tokens
    handler._request_kwargs = OpenApiModelHandler._build_response_request_kwargs(
        service_tier=service_tier,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
    )
    handler.request_timeout_s = 20.0
    handler.request_timeout = 20.0
    handler.disable_thinking = disable_thinking
    handler._extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if disable_thinking else None
    handler.user_role = "user"
    handler.cancel_scope = cancel_scope
    handler.tools = None
    handler.tools_choice = None
    return handler


def _capture_create_kwargs(handler):
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _make_response(output=[])

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))
    list(handler.process(_make_request("Hi")))
    return captured


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


def test_service_tier_is_passed_when_set():
    handler = _make_handler(stream=False, service_tier="priority")

    captured = _capture_create_kwargs(handler)

    assert captured["service_tier"] == "priority"


def test_reasoning_effort_is_passed_when_set_to_none_string():
    handler = _make_handler(stream=False, reasoning_effort="none")

    captured = _capture_create_kwargs(handler)

    assert captured["reasoning"] == {"effort": "none"}


def test_max_output_tokens_is_passed_when_set():
    handler = _make_handler(stream=False, max_output_tokens=64)

    captured = _capture_create_kwargs(handler)

    assert captured["max_output_tokens"] == 64


def test_optional_openai_response_fields_are_omitted_by_default():
    handler = _make_handler(stream=False)

    captured = _capture_create_kwargs(handler)

    assert "service_tier" not in captured
    assert "reasoning" not in captured
    assert "max_output_tokens" not in captured


def test_nonstream_usage_tracks_cached_reasoning_total_and_service_tier():
    handler = _make_handler(stream=False)
    usage = SimpleNamespace(
        input_tokens=12,
        input_tokens_details=SimpleNamespace(cached_tokens=4),
        output_tokens=6,
        output_tokens_details=SimpleNamespace(reasoning_tokens=2),
        total_tokens=18,
    )
    response = _make_response(output=[], usage=usage, service_tier="priority")
    handler.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response))

    outputs = list(handler.process(_make_request("Hi")))

    token_usage = next(output for output in outputs if isinstance(output, TokenUsage))
    assert token_usage.input_tokens == 12
    assert token_usage.cached_tokens == 4
    assert token_usage.output_tokens == 6
    assert token_usage.reasoning_tokens == 2
    assert token_usage.total_tokens == 18
    assert token_usage.service_tier == "priority"


def test_stream_usage_tracks_cached_reasoning_total_and_service_tier():
    handler = _make_handler(stream=True)
    usage = SimpleNamespace(
        input_tokens=12,
        input_tokens_details=SimpleNamespace(cached_tokens=4),
        output_tokens=6,
        output_tokens_details=SimpleNamespace(reasoning_tokens=2),
        total_tokens=18,
    )
    completed = MagicMock(spec=ResponseCompletedEvent)
    completed.response = SimpleNamespace(usage=usage, service_tier="priority")
    handler.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: _make_stream([completed])))

    outputs = list(handler.process(_make_request("Hi")))

    token_usage = next(output for output in outputs if isinstance(output, TokenUsage))
    assert token_usage.input_tokens == 12
    assert token_usage.cached_tokens == 4
    assert token_usage.output_tokens == 6
    assert token_usage.reasoning_tokens == 2
    assert token_usage.total_tokens == 18
    assert token_usage.service_tier == "priority"


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
