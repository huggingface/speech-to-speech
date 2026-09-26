"""Numpy-free tests for Responses output-item assembly and ID adoption."""

from __future__ import annotations

import logging
from copy import deepcopy
from types import SimpleNamespace

from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseReasoningItem
from openai.types.responses.response_output_text import ResponseOutputText

from speech_to_speech.LLM.responses_items import (
    AssembledAssistant,
    AssembledToolCall,
    ResponsesSegmentAssembler,
    adopt_provider_ids,
    reasoning_from_wire,
)


def _reasoning(
    rid: str = "rs_1",
    *,
    summary_text: str = "plan",
    encrypted_content: str = "gAAAA",
) -> ResponseReasoningItem:
    return ResponseReasoningItem(
        id=rid,
        type="reasoning",
        summary=[{"type": "summary_text", "text": summary_text}],
        encrypted_content=encrypted_content,
    )


def _function_call(
    *,
    item_id: str = "fc_orig",
    call_id: str = "call_original",
    name: str = "camera",
    arguments: str = "{}",
) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        type="function_call",
        id=item_id,
        call_id=call_id,
        name=name,
        arguments=arguments,
    )


def _message(text: str = "hello", item_id: str = "msg_1") -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=item_id,
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def _assemble(items, *, capture_reasoning: bool = True):
    assembler = ResponsesSegmentAssembler(capture_reasoning=capture_reasoning)
    events = []
    for item in items:
        events.extend(assembler.feed_item(item))
    events.extend(assembler.finish())
    return events


def _payload(rid: str = "rs_1") -> dict:
    return {
        "id": rid,
        "type": "reasoning",
        "summary": [{"text": "plan", "type": "summary_text"}],
        "encrypted_content": "gAAAA",
    }


def test_reasoning_from_wire_copies_encrypted_payload():
    item = _reasoning()
    record = reasoning_from_wire(item)
    assert record.id == "rs_1"
    assert record.payload["type"] == "reasoning"
    assert record.payload["id"] == "rs_1"
    assert record.payload["encrypted_content"] == "gAAAA"
    assert record.payload["summary"] == [{"text": "plan", "type": "summary_text"}]
    record.payload["encrypted_content"] = "mutated"
    assert item.encrypted_content == "gAAAA"


def test_assemble_list_and_sequential_feed_keep_provider_ids_and_reasoning_order():
    items = [_reasoning("rs_1"), _reasoning("rs_2"), _function_call()]
    from_list = _assemble(deepcopy(items))
    assembler = ResponsesSegmentAssembler()
    sequential = []
    for item in deepcopy(items):
        sequential.extend(assembler.feed_item(item))
    sequential.extend(assembler.finish())

    assert len(from_list) == len(sequential) == 1
    for assembled in (from_list[0], sequential[0]):
        assert isinstance(assembled, AssembledToolCall)
        assert assembled.replay_exact is True
        assert assembled.item.id == "fc_orig"
        assert assembled.item.call_id == "call_original"
        assert [record.id for record in assembled.leading_reasoning] == ["rs_1", "rs_2"]
        assert assembled.leading_reasoning[0].payload == _payload("rs_1")
        assert assembled.leading_reasoning[1].payload == _payload("rs_2")


def test_trailing_reasoning_is_dropped(caplog):
    assembler = ResponsesSegmentAssembler()
    assert list(assembler.feed_item(_reasoning())) == []
    with caplog.at_level(logging.DEBUG):
        assert list(assembler.finish()) == []
    assert any("trailing" in message.lower() or "reasoning" in message.lower() for message in caplog.messages)


def test_unknown_type_does_not_consume_reasoning_buffer(caplog):
    items = [_reasoning(), SimpleNamespace(type="web_search_call"), _function_call()]
    with caplog.at_level(logging.WARNING):
        events = _assemble(items)
    assert any("Not supported message type" in message for message in caplog.messages)
    assert len(events) == 1
    assembled = events[0]
    assert isinstance(assembled, AssembledToolCall)
    assert assembled.replay_exact is True
    assert assembled.item.call_id == "call_original"
    assert assembled.item.id == "fc_orig"
    assert [record.id for record in assembled.leading_reasoning] == ["rs_1"]
    assert assembled.leading_reasoning[0].payload["encrypted_content"] == "gAAAA"


def test_prefixless_ids_are_minted_and_reasoning_is_dropped():
    call = ResponseFunctionToolCall(
        type="function_call",
        id="plain_id",
        call_id="plain_call",
        name="camera",
        arguments="{}",
    )
    events = _assemble([_reasoning(), call])
    assert len(events) == 1
    assembled = events[0]
    assert isinstance(assembled, AssembledToolCall)
    assert assembled.replay_exact is False
    assert assembled.leading_reasoning == ()
    assert assembled.item.id is not None and assembled.item.id.startswith("fc_")
    assert assembled.item.call_id is not None and assembled.item.call_id.startswith("call_")
    assert assembled.item.id != "plain_id"
    assert assembled.item.call_id != "plain_call"


def test_adopt_provider_ids_keeps_exact_prefixes():
    item = _function_call()
    assert adopt_provider_ids(item) is True
    assert item.id == "fc_orig"
    assert item.call_id == "call_original"


def test_missing_function_call_item_id_is_minted_without_dropping_reasoning():
    call = ResponseFunctionToolCall(
        type="function_call",
        call_id="call_original",
        name="camera",
        arguments="{}",
    )
    events = _assemble([_reasoning(), call])
    assert len(events) == 1
    assembled = events[0]
    assert isinstance(assembled, AssembledToolCall)
    assert assembled.replay_exact is True
    assert assembled.item.call_id == "call_original"
    assert assembled.item.id is not None and assembled.item.id.startswith("fc_")
    assert [record.id for record in assembled.leading_reasoning] == ["rs_1"]


def test_assistant_message_flushes_reasoning_as_exact_anchor():
    events = _assemble([_reasoning(), _message("done")])
    assert len(events) == 1
    assembled = events[0]
    assert isinstance(assembled, AssembledAssistant)
    assert [record.id for record in assembled.leading_reasoning] == ["rs_1"]
    assert assembled.content == [AssistantContent(type="output_text", text="done")]
