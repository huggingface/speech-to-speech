"""Responses item fidelity through the existing history/response lifecycle."""

import json
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI
from openai.types.realtime import RealtimeConversationItemFunctionCall, RealtimeConversationItemFunctionCallOutput
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseReasoningItem,
)

from speech_to_speech.LLM.chat import Chat, CompactionResult, ResponsesFunctionCall, make_user_message
from speech_to_speech.LLM.responses_api_language_model import ResponsesApiModelHandler
from speech_to_speech.pipeline.messages import EndOfResponse, LLMResponseChunk, ResponsePrefetchTransaction
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from tests.test_responses_api_language_model import _make_handler, _make_request, _make_response, _make_stream


def _items(suffix="1"):
    return [
        ResponseReasoningItem(
            id=f"rs_{suffix}",
            type="reasoning",
            summary=[],
            encrypted_content=f"opaque-{suffix}",
            status="completed",
        ),
        ResponseFunctionToolCall(
            id=f"fc_{suffix}",
            call_id=f"call_{suffix}",
            type="function_call",
            name="lookup",
            arguments="{}",
            status="completed",
        ),
    ]


def _stream(items):
    return _make_stream(
        [
            ResponseOutputItemDoneEvent(type="response.output_item.done", output_index=i, sequence_number=i, item=item)
            for i, item in enumerate(items)
        ]
    )


def _message(content="mixed"):
    parts = [
        {
            "type": "output_text",
            "text": "Checking the source.",
            "annotations": [
                {
                    "type": "url_citation",
                    "start_index": 0,
                    "end_index": 8,
                    "title": "Source",
                    "url": "https://example.com/source",
                }
            ],
            "logprobs": None,
        },
        {"type": "refusal", "refusal": "Cannot provide that detail."},
    ]
    return ResponseOutputMessage.model_validate(
        {
            "id": "msg_provider",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "phase": "commentary",
            "content": parts if content == "mixed" else parts[1:] if content == "refusal" else [],
        }
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("content", ["mixed", "refusal", "empty"])
def test_assistant_message_payload_survives_complete_reasoning_tool_continuation(stream, content):
    reasoning, call = _items()
    message = _message(content)
    items = [reasoning, message, call]
    originals = [item.model_dump(exclude_unset=True) for item in items]
    handler = _make_handler(stream=stream)
    request = _make_request(chat_size=5)
    captured = []

    def create(**kwargs):
        captured.append(kwargs["input"])
        if len(captured) == 1:
            return _stream(items) if stream else _make_response(items)
        assert kwargs["input"][2:5] == originals
        assert kwargs["input"][5]["type"] == "function_call_output"
        assert kwargs["input"][5]["call_id"] == call.call_id
        return _stream([]) if stream else _make_response([])

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    first = list(handler.process(request))
    tool = next(item.tools[0] for item in first if isinstance(item, LLMResponseChunk) and item.tools)
    local_message = next(
        item for item in request.runtime_config.chat.buffer if getattr(item, "role", None) == "assistant"
    )
    assert local_message.id.startswith("msg_") and local_message.id != message.id
    assert [part.text for part in local_message.content] == [
        part.text if part.type == "output_text" else part.refusal for part in message.content
    ]
    _finish(request.runtime_config.chat, tool.call_id)
    # A copy uses the same replay path while retaining its own provider payload.
    assert request.runtime_config.chat.copy(deep=True).to_responses_api_chat()[1:4] == originals
    second = list(handler.process(request))
    assert next(item for item in second if isinstance(item, EndOfResponse)).error is None
    assert len(captured) == 2
    assert [item.model_dump(exclude_unset=True) for item in items] == originals


def _stage(chat, items):
    reasoning, call = items
    mapped = ResponsesApiModelHandler._tool_call(call)
    chat.add_item(reasoning)
    chat.add_ordered_function_call(
        ResponsesFunctionCall(
            **mapped.item.model_dump(exclude_unset=True),
            response_item=call.model_copy(deep=True),
        )
    )


def _finish(chat, call_id="call_1"):
    call_id = next(
        (
            item.call_id
            for item in chat.buffer
            if isinstance(item, ResponsesFunctionCall) and item.response_item.call_id == call_id
        ),
        call_id,
    )
    chat.add_item(
        RealtimeConversationItemFunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output="done",
        )
    )


@pytest.mark.parametrize("stream", [False, True])
def test_consecutive_tools_replay_all_reasoning_since_user(stream):
    handler = _make_handler(stream=stream)
    request = _make_request(chat_size=5)
    first, second = _items("1"), _items("2")
    replies = iter([first, second, []])
    captured = []

    def create(**kwargs):
        captured.append(kwargs["input"])
        items = next(replies)
        return _stream(items) if stream else _make_response(items)

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    for expected in ["call_1", "call_2"]:
        outputs = list(handler.process(request))
        tool = next(o.tools[0] for o in outputs if isinstance(o, LLMResponseChunk) and o.tools)
        assert tool.call_id.startswith("call_") and tool.call_id != expected
        _finish(request.runtime_config.chat, expected)
    list(handler.process(request))
    replay = captured[2][2:]
    assert [item["type"] for item in replay] == [
        "reasoning",
        "function_call",
        "function_call_output",
        "reasoning",
        "function_call",
        "function_call_output",
    ]
    assert [replay[i] for i in [0, 1, 3, 4]] == [item.model_dump(exclude_unset=True) for item in [*first, *second]]


@pytest.mark.parametrize("stream", [False, True])
def test_reasoning_before_assistant_message_is_retained_without_client_output(stream):
    reasoning = _items()[0]
    message = ResponseOutputMessage(
        id="msg_provider",
        type="message",
        role="assistant",
        status="completed",
        content=[{"type": "output_text", "text": "Answer.", "annotations": []}],
    )
    handler = _make_handler(stream=stream)
    request = _make_request(chat_size=5)
    handler.client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kw: _stream([reasoning, message]) if stream else _make_response([reasoning, message]),
        )
    )
    outputs = list(handler.process(request))
    replay = request.runtime_config.chat.to_responses_api_chat()
    assert [item["type"] for item in replay] == ["message", "reasoning", "message"]
    assert replay[1] == reasoning.model_dump(exclude_unset=True)
    assert not any(isinstance(item, LLMResponseChunk) and "opaque" in item.text for item in outputs)
    assert next(item for item in outputs if isinstance(item, EndOfResponse)).error is None


@pytest.mark.parametrize("discard", ["cancel", "prefetch"])
@pytest.mark.parametrize("with_message", [None, "mixed", "empty"])
def test_reasoning_and_fast_tool_output_are_rolled_back(discard, with_message):
    handler = _make_handler(stream=True)
    request = _make_request(chat_size=5)
    if discard == "prefetch":
        request.prefetch_transaction = ResponsePrefetchTransaction()
    items = _items()
    if with_message:
        items.insert(1, _message(with_message))
    handler.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: _stream(items)))
    generation = handler.process(request)
    chunk = next(generation)
    assert isinstance(chunk, LLMResponseChunk) and chunk.tools
    assert request.runtime_config.chat.buffer[1].type == "reasoning"
    _finish(request.runtime_config.chat)
    if discard == "prefetch":
        request.prefetch_transaction.discard()
    generation.close()
    assert [item.type for item in request.runtime_config.chat.buffer] == ["message"]
    assert not request.runtime_config.chat.has_pending_tool_calls()


def test_stale_speculative_turn_rolls_back_reasoning_and_call():
    handler = _make_handler(stream=True)
    tracker = SpeculativeTurnTracker()
    handler.speculative_turns = tracker
    request = _make_request(chat_size=5)
    request.turn_id, request.turn_revision = tracker.start_turn()
    handler.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: _stream(_items())))
    generation = handler.process(request)
    chunk = next(generation)
    assert isinstance(chunk, LLMResponseChunk) and chunk.tools
    tracker.start_turn()
    list(generation)
    assert [item.type for item in request.runtime_config.chat.buffer] == ["message"]
    assert not request.runtime_config.chat.has_pending_tool_calls()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("with_message", [False, True])
def test_out_of_band_reasoning_does_not_enter_default_history(stream, with_message):
    handler = _make_handler(stream=stream)
    request = _make_request()
    request.response = RealtimeResponseCreateParams(conversation="none")
    items = _items()
    if with_message:
        items.insert(1, _message())
    handler.client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kw: _stream(items) if stream else _make_response(items),
        )
    )
    outputs = list(handler.process(request))
    assert any(isinstance(o, LLMResponseChunk) and o.tools for o in outputs)
    assert next(o for o in outputs if isinstance(o, EndOfResponse)).error is None
    assert [item.type for item in request.runtime_config.chat.buffer] == ["message"]


@pytest.mark.parametrize("stream", [False, True])
def test_out_of_band_opaque_provider_call_can_continue_with_explicit_client_input(stream):
    handler = _make_handler(stream=stream)
    request = _make_request()
    request.response = RealtimeResponseCreateParams(conversation="none")
    call = _items()[1].model_copy(update={"id": "opaque-item-id", "call_id": "opaque-call-id"})
    captured = []

    def create(**kwargs):
        captured.append(kwargs["input"])
        items = [call] if len(captured) == 1 else []
        return _stream(items) if stream else _make_response(items)

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    first = list(handler.process(request))
    tool = next(item.tools[0] for item in first if isinstance(item, LLMResponseChunk) and item.tools)
    request.response = RealtimeResponseCreateParams(
        conversation="none",
        input=[
            RealtimeConversationItemFunctionCall(
                type="function_call", id=tool.id, call_id=tool.call_id, name=tool.name, arguments=tool.arguments
            ),
            RealtimeConversationItemFunctionCallOutput(
                type="function_call_output", call_id=tool.call_id, output="done"
            ),
        ],
    )
    second = list(handler.process(request))
    assert next(item for item in second if isinstance(item, EndOfResponse)).error is None
    assert len(captured) == 2
    assert [item["type"] for item in captured[1][1:]] == ["function_call", "function_call_output"]
    assert captured[1][1]["call_id"] == captured[1][2]["call_id"] == tool.call_id
    assert [item.type for item in request.runtime_config.chat.buffer] == ["message"]


def test_fast_output_does_not_allow_eviction_before_late_stream_reasoning_commits():
    reasoning, call = _items()
    handler = _make_handler(stream=True)
    request = _make_request(chat_size=1)
    response = _make_response([reasoning, call])
    response.status = "completed"
    events = [
        ResponseOutputItemDoneEvent(type="response.output_item.done", sequence_number=0, output_index=1, item=call),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done", sequence_number=1, output_index=0, item=reasoning
        ),
        ResponseCompletedEvent.model_construct(type="response.completed", sequence_number=2, response=response),
    ]
    handler.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: _make_stream(events)))
    generation = handler.process(request)
    chunk = next(generation)
    assert isinstance(chunk, LLMResponseChunk) and chunk.tools
    _finish(request.runtime_config.chat, chunk.tools[0].call_id)
    for text in ["later 1", "later 2", "later 3", "latest"]:
        request.runtime_config.chat.add_item(make_user_message(text))
    # The response still owns this turn even though its tool is already resolved.
    assert any(isinstance(item, ResponsesFunctionCall) for item in request.runtime_config.chat.buffer)
    outputs = list(generation)
    assert next(item for item in outputs if isinstance(item, EndOfResponse)).error is None
    # Completion-time trimming evicts the whole old chain, including late reasoning.
    replay = request.runtime_config.chat.to_responses_api_chat()
    assert len(replay) == 1
    assert replay[0]["content"][0]["text"] == "latest"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("completed_prefix", [False, True])
def test_out_of_band_snapshot_omits_reasoning_for_unresolved_call(stream, completed_prefix):
    handler = _make_handler(stream=stream)
    request = _make_request(chat_size=5)
    responses = [_items("1"), _items("2")] if completed_prefix else [_items("2")]
    responses.append([])
    replies = iter(responses)
    captured = []

    def create(**kwargs):
        captured.append(kwargs["input"])
        items = next(replies)
        return _stream(items) if stream else _make_response(items)

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    if completed_prefix:
        list(handler.process(request))
        _finish(request.runtime_config.chat, "call_1")
    list(handler.process(request))
    canonical = request.runtime_config.chat.copy(deep=True)
    request.response = RealtimeResponseCreateParams(conversation="none")
    outputs = list(handler.process(request))
    assert next(item for item in outputs if isinstance(item, EndOfResponse)).error is None
    assert not any(item.get("id") in {"rs_2", "fc_2"} for item in captured[-1])
    if completed_prefix:
        assert captured[-1][2:4] == [item.model_dump(exclude_unset=True) for item in _items("1")]
    assert request.runtime_config.chat.buffer == canonical.buffer


def test_stream_completion_order_does_not_reorder_replay_or_delay_tool_dispatch():
    reasoning, call = _items()
    message = ResponseOutputMessage(
        id="msg_provider",
        type="message",
        role="assistant",
        status="completed",
        content=[{"type": "output_text", "text": "Trailing text.", "annotations": []}],
    )
    provider_items = [reasoning, call, message]
    events = [
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=sequence,
            output_index=index,
            item=provider_items[index],
        )
        for sequence, index in enumerate([2, 1, 0])
    ]
    handler = _make_handler(stream=True)
    request = _make_request(chat_size=5)
    handler.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: _make_stream(events)))
    generation = handler.process(request)
    # The call remains exposed as soon as its own done event arrives, before
    # the reasoning's later done event or stream exhaustion.
    chunk = next(generation)
    assert isinstance(chunk, LLMResponseChunk) and chunk.tools
    assert not any(item.type == "reasoning" for item in request.runtime_config.chat.buffer)
    request.runtime_config.chat.add_item(make_user_message("later user"))
    _finish(request.runtime_config.chat)
    list(generation)
    replay = request.runtime_config.chat.to_responses_api_chat()
    assert [item["type"] for item in replay] == [
        "message",
        "reasoning",
        "function_call",
        "function_call_output",
        "message",
        "message",
    ]
    assert replay[1:3] == [item.model_dump(exclude_unset=True) for item in [reasoning, call]]
    assert replay[4]["content"][0]["text"] == "Trailing text."
    assert replay[5]["content"][0]["text"] == "later user"


def test_pending_tool_chain_survives_bounded_history_then_evicts_as_a_turn():
    chat = Chat(size=1)
    chat.add_item(make_user_message("tool question"))
    originals = _items()
    _stage(chat, originals)
    for text in ["later 1", "later 2", "later 3", "later 4"]:
        chat.add_item(make_user_message(text))
    assert chat._user_turn_count <= 3  # hard cap + the pending chain
    chat.trim_if_needed()
    assert chat._user_turn_count == 2  # one complete turn + pending chain
    _finish(chat)
    replay = chat.to_responses_api_chat()
    assert replay[0]["content"][0]["text"] == "tool question"
    assert replay[1:3] == [item.model_dump(exclude_unset=True) for item in originals]
    assert replay[3]["type"] == "function_call_output"
    chat.trim_if_needed()
    assert [item.type for item in chat.buffer] == ["message"]
    assert chat.buffer[0].content[0].text == "later 4"


@pytest.mark.parametrize("result_timing", ["pending", "late", "during_compaction"])
def test_compaction_preserves_complete_context_for_retained_tool_chain(result_timing):
    chat = Chat(size=10)
    chat.add_item(make_user_message("tool question"))
    originals = _items()
    _stage(chat, originals)
    if result_timing == "during_compaction":
        _finish(chat)
    for text in ["later 1", "later 2", "latest"]:
        chat.add_item(make_user_message(text))
    if result_timing == "late":
        _finish(chat)
    with chat._lock:
        snapshot, markers, _ = chat._snapshot_for_compaction()
    if result_timing == "during_compaction":
        # Simulate the existing late-output race: the compaction snapshot was
        # taken before a result appended outside its turn-aligned prefix.
        output = next(item for item in chat.buffer if item.type == "function_call_output")
        markers.remove(output.id)
    else:
        assert not any(item.get("type") in {"reasoning", "function_call"} for item in snapshot)
    chat._apply_compaction(
        CompactionResult(user_summary="older questions", assistant_summary="older answers"), markers, chat._gen_counter
    )
    if result_timing == "pending":
        _finish(chat)
    replay = chat.to_responses_api_chat()
    index = next(i for i, item in enumerate(replay) if item.get("type") == "reasoning")
    assert replay[index - 1]["content"][0]["text"] == "tool question"
    assert replay[index : index + 2] == [item.model_dump(exclude_unset=True) for item in originals]
    assert replay[index + 2]["type"] == "function_call_output"


def test_provider_payload_fields_and_opaque_ids_are_not_normalized():
    chat = Chat(size=5)
    chat.add_item(make_user_message("question"))
    reasoning = ResponseReasoningItem.model_validate(
        {
            "id": "opaque-reasoning-id",
            "type": "reasoning",
            "summary": [],
            "encrypted_content": None,
            "provider_extension": {"value": 3},
        }
    )
    call = ResponseFunctionToolCall(
        id="opaque-item-id",
        call_id="opaque-call-id",
        type="function_call",
        name="lookup",
        arguments="{}",
    )
    _stage(chat, [reasoning, call])
    _finish(chat, "opaque-call-id")
    assert chat.to_responses_api_chat()[1:3] == [item.model_dump(exclude_unset=True) for item in [reasoning, call]]
    assert [item["role"] for item in chat.to_transformers_chat()] == ["user", "assistant", "tool"]
    clone = chat.copy(deep=True)
    chat.reset()
    assert chat.buffer == []
    assert clone.to_responses_api_chat()[1:3] == [item.model_dump(exclude_unset=True) for item in [reasoning, call]]


@pytest.mark.parametrize("stream, reverse_done", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("with_message", [False, True])
def test_pinned_sdk_parses_and_serializes_reasoning_tool_continuation(stream, reverse_done, with_message):
    """Exercise SDK HTTP/SSE handling; no hosted API is contacted."""
    items = _items()
    if with_message:
        items.insert(1, _message())
    originals = [item.model_dump(exclude_unset=True) for item in items]
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        output = originals if len(requests) == 1 else []
        response = {
            "id": f"resp_{len(requests)}",
            "object": "response",
            "created_at": 0,
            "model": "test-model",
            "status": "completed",
            "output": output,
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": {},
            "parallel_tool_calls": True,
            "temperature": 1,
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1,
            "usage": None,
        }
        if len(requests) == 2:
            end = 2 + len(originals)
            assert body["input"][2:end] == originals
            assert body["input"][end]["call_id"] == "call_1"
        if not stream:
            return httpx.Response(200, json=response)
        indexes = list(range(len(output)))
        if reverse_done:
            indexes.reverse()
        events = [
            {
                "type": "response.output_item.done",
                "sequence_number": sequence,
                "output_index": index,
                "item": output[index],
            }
            for sequence, index in enumerate(indexes)
        ]
        events.append({"type": "response.completed", "sequence_number": len(events), "response": response})
        data = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=data)

    handler = _make_handler(stream=stream)
    request = _make_request(chat_size=5)
    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        handler.client = client
        first = list(handler.process(request))
        assert any(isinstance(item, LLMResponseChunk) and item.tools for item in first)
        _finish(request.runtime_config.chat)
        second = list(handler.process(request))
        assert next(item for item in second if isinstance(item, EndOfResponse)).error is None
    assert len(requests) == 2
