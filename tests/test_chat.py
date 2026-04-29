"""Extensive tests for speech_to_speech.LLM.chat.

Covers Chat class (init, add_item validation/eviction,
serialization to both Response API and transformers formats,
copy/reset, strip_images, internal helpers) and the three factory
functions (make_user_message, make_assistant_message, make_system_message).
"""

from __future__ import annotations

import pytest
from openai.types.realtime.conversation_item import (
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
    RealtimeConversationItemSystemMessage,
    RealtimeConversationItemUserMessage,
)
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)
from openai.types.realtime.realtime_conversation_item_system_message import (
    Content as SystemContent,
)
from openai.types.realtime.realtime_conversation_item_user_message import (
    Content as UserContent,
)

from speech_to_speech.LLM.chat import (
    Chat,
    ChatItemError,
    make_assistant_message,
    make_system_message,
    make_user_message,
)

# ===================================================================
# Helpers
# ===================================================================


def _user(text: str) -> RealtimeConversationItemUserMessage:
    return make_user_message(text)


def _assistant(text: str) -> RealtimeConversationItemAssistantMessage:
    return make_assistant_message(text)


def _system(text: str) -> RealtimeConversationItemSystemMessage:
    return make_system_message(text)


def _fc(call_id: str = "call_1", name: str = "my_func", arguments: str = "{}") -> RealtimeConversationItemFunctionCall:
    if not call_id.startswith("call_"):
        call_id = f"call_{call_id}"
    return RealtimeConversationItemFunctionCall(
        type="function_call",
        id=f"fc_{call_id}",
        call_id=call_id,
        name=name,
        arguments=arguments,
    )


def _fco(
    call_id: str = "call_1", output: str = '{"ok": true}', status=None
) -> RealtimeConversationItemFunctionCallOutput:
    if not call_id.startswith("call_"):
        call_id = f"call_{call_id}"
    return RealtimeConversationItemFunctionCallOutput(
        type="function_call_output",
        call_id=call_id,
        output=output,
        status=status,
    )


def _user_msg_with_parts(*parts) -> RealtimeConversationItemUserMessage:
    """Build a user message with arbitrary content parts.

    Each *part* is a tuple like ``("text", "hello")`` or ``("image", "url")``.
    """
    content = []
    for kind, value in parts:
        if kind == "text":
            content.append(UserContent(type="input_text", text=value))
        elif kind == "image":
            content.append(UserContent(type="input_image", image_url=value))
        elif kind == "audio":
            content.append(UserContent(type="input_audio", transcript=value))
    return RealtimeConversationItemUserMessage(type="message", role="user", content=content)


def _assistant_msg_with_parts(*parts) -> RealtimeConversationItemAssistantMessage:
    content = []
    for kind, value in parts:
        if kind == "text":
            content.append(AssistantContent(type="output_text", text=value))
        elif kind == "audio":
            content.append(AssistantContent(type="output_audio", transcript=value))
    return RealtimeConversationItemAssistantMessage(type="message", role="assistant", content=content)


# ===================================================================
# 1. TestChatInit
# ===================================================================


class TestChatInit:
    def test_default_state(self):
        chat = Chat(size=5)
        assert chat.size == 5
        assert chat.buffer == []
        assert chat.init_chat_message is None
        assert chat._pending_tool_calls == {}
        assert chat._user_turn_count == 0

    def test_size_stored(self):
        for s in (0, 1, 100):
            assert Chat(size=s).size == s


# ===================================================================
# 2. TestFactoryHelpers
# ===================================================================


class TestFactoryHelpers:
    def test_make_user_message(self):
        msg = make_user_message("hello")
        assert isinstance(msg, RealtimeConversationItemUserMessage)
        assert msg.role == "user"
        assert msg.type == "message"
        assert len(msg.content) == 1
        assert msg.content[0].type == "input_text"
        assert msg.content[0].text == "hello"

    def test_make_assistant_message(self):
        msg = make_assistant_message("world")
        assert isinstance(msg, RealtimeConversationItemAssistantMessage)
        assert msg.role == "assistant"
        assert msg.type == "message"
        assert len(msg.content) == 1
        assert msg.content[0].type == "output_text"
        assert msg.content[0].text == "world"

    def test_make_system_message(self):
        msg = make_system_message("You are helpful.")
        assert isinstance(msg, RealtimeConversationItemSystemMessage)
        assert msg.role == "system"
        assert msg.type == "message"
        assert len(msg.content) == 1
        assert msg.content[0].type == "input_text"
        assert msg.content[0].text == "You are helpful."


# ===================================================================
# 3. TestInitChat
# ===================================================================


class TestInitChat:
    def test_sets_init_chat_message(self):
        chat = Chat(size=5)
        sys_msg = _system("Be concise.")
        chat.init_chat(sys_msg)
        assert chat.init_chat_message is sys_msg

    def test_overwrite_replaces_previous(self):
        chat = Chat(size=5)
        chat.init_chat(_system("first"))
        chat.init_chat(_system("second"))
        assert chat.init_chat_message.content[0].text == "second"

    def test_system_message_not_in_buffer(self):
        chat = Chat(size=5)
        chat.init_chat(_system("system"))
        assert chat.buffer == []


# ===================================================================
# 4. TestAddItemEviction
# ===================================================================


class TestAddItemEviction:
    def test_add_user_increments_turn_count(self):
        chat = Chat(size=5)
        assert chat._user_turn_count == 0
        chat.add_item(_user("hi"))
        assert chat._user_turn_count == 1
        chat.add_item(_user("there"))
        assert chat._user_turn_count == 2

    def test_add_function_call_registers_pending(self):
        chat = Chat(size=5)
        fc = _fc("cid_1")
        chat.add_item(fc)
        assert "call_cid_1" in chat._pending_tool_calls
        assert chat._pending_tool_calls["call_cid_1"] is fc

    def test_add_function_call_none_call_id_auto_generates(self):
        chat = Chat(size=5)
        fc = RealtimeConversationItemFunctionCall(
            type="function_call",
            call_id=None,
            name="f",
            arguments="{}",
        )
        chat.add_item(fc)
        assert fc.call_id is not None
        assert fc.call_id.startswith("call_")

    def test_eviction_when_exceeding_size(self):
        chat = Chat(size=1)
        chat.add_item(_user("t1"))
        chat.add_item(_assistant("r1"))
        assert chat._user_turn_count == 1

        chat.add_item(_user("t2"))
        assert chat._user_turn_count == 1
        assert chat.buffer[0].content[0].text == "t2"

    def test_eviction_removes_up_to_next_user_boundary(self):
        chat = Chat(size=1)
        chat.add_item(_user("t1"))
        chat.add_item(_assistant("a1"))
        chat.add_item(_fc("c1"))
        chat.add_item(_fco("c1"))
        chat.add_item(_assistant("a2"))
        assert len(chat.buffer) == 5

        chat.add_item(_user("t2"))
        assert chat._user_turn_count == 1
        remaining_types = [e.type for e in chat.buffer]
        assert "message" in remaining_types
        assert chat.buffer[0].content[0].text == "t2"

    def test_size_zero_evicts_every_user_message(self):
        chat = Chat(size=0)
        chat.add_item(_user("a"))
        assert chat._user_turn_count == 0
        assert len(chat.buffer) == 0

    def test_non_user_items_do_not_trigger_eviction(self):
        chat = Chat(size=1)
        chat.add_item(_assistant("a"))
        chat.add_item(_fc("c1"))
        chat.add_item(_assistant("b"))
        assert len(chat.buffer) == 3

    def test_multiple_evictions(self):
        chat = Chat(size=2)
        for i in range(5):
            chat.add_item(_user(f"u{i}"))
            chat.add_item(_assistant(f"a{i}"))
        assert chat._user_turn_count == 2
        user_texts = [e.content[0].text for e in chat.buffer if isinstance(e, RealtimeConversationItemUserMessage)]
        assert user_texts == ["u3", "u4"]


# ===================================================================
# 5. TestAppendToolOutput
# ===================================================================


class TestAppendToolOutput:
    def test_happy_path(self):
        chat = Chat(size=5)
        chat.add_item(_fc("c1"))
        fco = _fco("c1")
        chat.append_tool_output("call_c1", fco)

        assert "call_c1" not in chat._pending_tool_calls
        assert chat.buffer[-1] is fco

    def test_marks_function_call_completed_on_none_status(self):
        chat = Chat(size=5)
        fc = _fc("c1")
        chat.add_item(fc)
        fco = _fco("c1", status=None)
        chat.append_tool_output("call_c1", fco)

        assert fc.status == "completed"

    def test_status_propagation_from_output(self):
        chat = Chat(size=5)
        fc = _fc("c1")
        chat.add_item(fc)
        fco = _fco("c1", status="incomplete")
        chat.append_tool_output("call_c1", fco)

        assert fc.status == "incomplete"

    def test_reinjection_path(self):
        chat = Chat(size=1)
        chat.add_item(_user("u1"))
        chat.add_item(_fc("cx"))
        chat.add_item(_user("u2"))
        assert not chat._has_call_id_in_buffer("call_cx")
        assert "call_cx" in chat._pending_tool_calls

        chat.append_tool_output("call_cx", _fco("cx"))
        assert chat._has_call_id_in_buffer("call_cx")
        assert any(isinstance(e, RealtimeConversationItemFunctionCall) and e.call_id == "call_cx" for e in chat.buffer)
        assert any(
            isinstance(e, RealtimeConversationItemFunctionCallOutput) and e.call_id == "call_cx" for e in chat.buffer
        )

    def test_reinjection_sets_status(self):
        chat = Chat(size=1)
        chat.add_item(_user("u1"))
        fc = _fc("cx")
        chat.add_item(fc)
        chat.add_item(_user("u2"))

        fco = _fco("cx", status="incomplete")
        chat.append_tool_output("call_cx", fco)
        reinjected = next(
            e for e in chat.buffer if isinstance(e, RealtimeConversationItemFunctionCall) and e.call_id == "call_cx"
        )
        assert reinjected.status == "incomplete"

    def test_unknown_call_id_raises(self):
        chat = Chat(size=5)
        with pytest.raises(ChatItemError, match="unknown_id"):
            chat.append_tool_output("unknown_id", _fco("unknown_id"))


# ===================================================================
# 6. TestAddItem
# ===================================================================


class TestAddItem:
    # -- System message --

    def test_system_message_routed_to_init_chat(self):
        chat = Chat(size=5)
        sys_msg = _system("You are an expert.")
        chat.add_item(sys_msg)
        assert chat.init_chat_message is sys_msg
        assert chat.buffer == []

    # -- User message --

    def test_user_message_text_appended(self):
        chat = Chat(size=5)
        chat.add_item(_user("hi"))
        assert len(chat.buffer) == 1
        assert chat.buffer[0].content[0].text == "hi"

    def test_user_message_filters_unsupported_content(self):
        chat = Chat(size=5)
        msg = _user_msg_with_parts(("text", "hello"), ("audio", "transcript"))
        chat.add_item(msg)
        assert len(chat.buffer[0].content) == 1
        assert chat.buffer[0].content[0].type == "input_text"

    def test_user_message_keeps_image_content(self):
        chat = Chat(size=5)
        msg = _user_msg_with_parts(("text", "look"), ("image", "http://img.png"))
        chat.add_item(msg)
        assert len(chat.buffer[0].content) == 2

    def test_user_message_empty_after_filter_raises(self):
        chat = Chat(size=5)
        msg = _user_msg_with_parts(("audio", "transcript only"))
        with pytest.raises(ChatItemError, match="no supported content"):
            chat.add_item(msg)

    def test_user_message_empty_text_raises(self):
        chat = Chat(size=5)
        msg = RealtimeConversationItemUserMessage(
            type="message",
            role="user",
            content=[UserContent(type="input_text", text="")],
        )
        with pytest.raises(ChatItemError, match="no supported content"):
            chat.add_item(msg)

    # -- Assistant message --

    def test_assistant_message_appended(self):
        chat = Chat(size=5)
        chat.add_item(_user("hi"))
        chat.add_item(_assistant("hello"))
        assert len(chat.buffer) == 2
        assert chat.buffer[1].content[0].text == "hello"

    def test_assistant_message_filters_non_text(self):
        chat = Chat(size=5)
        msg = _assistant_msg_with_parts(("text", "ok"), ("audio", "audio_data"))
        chat.add_item(msg)
        assert len(chat.buffer[0].content) == 1
        assert chat.buffer[0].content[0].type == "output_text"

    def test_assistant_message_empty_after_filter_raises(self):
        chat = Chat(size=5)
        msg = _assistant_msg_with_parts(("audio", "only audio"))
        with pytest.raises(ChatItemError, match="no text content"):
            chat.add_item(msg)

    def test_assistant_message_empty_text_raises(self):
        chat = Chat(size=5)
        msg = RealtimeConversationItemAssistantMessage(
            type="message",
            role="assistant",
            content=[AssistantContent(type="output_text", text="")],
        )
        with pytest.raises(ChatItemError, match="no text content"):
            chat.add_item(msg)

    # -- Function call --

    def test_function_call_appended(self):
        chat = Chat(size=5)
        fc = _fc("c1", "do_stuff")
        chat.add_item(fc)
        assert len(chat.buffer) == 1
        assert chat.buffer[0].call_id == "call_c1"

    def test_function_call_missing_call_id_auto_generates(self):
        chat = Chat(size=5)
        fc = RealtimeConversationItemFunctionCall(
            type="function_call",
            call_id=None,
            name="f",
            arguments="{}",
        )
        chat.add_item(fc)
        assert fc.call_id is not None
        assert fc.call_id.startswith("call_")

    def test_function_call_empty_call_id_auto_generates(self):
        chat = Chat(size=5)
        fc = RealtimeConversationItemFunctionCall(
            type="function_call",
            call_id="",
            name="f",
            arguments="{}",
        )
        chat.add_item(fc)
        assert fc.call_id is not None
        assert fc.call_id.startswith("call_")

    # -- Function call output --

    def test_function_call_output_delegates_to_append_tool_output(self):
        chat = Chat(size=5)
        chat.add_item(_fc("c1"))
        fco = _fco("c1")
        chat.add_item(fco)
        assert chat.buffer[-1] is fco

    def test_function_call_output_unknown_raises(self):
        chat = Chat(size=5)
        with pytest.raises(ChatItemError, match="no_such_call"):
            chat.add_item(_fco("no_such_call"))


# ===================================================================
# 7. TestToResponseApiChat
# ===================================================================


class TestToResponseApiChat:
    def test_empty_chat(self):
        chat = Chat(size=5)
        assert chat.to_response_api_chat() == []

    def test_system_message_serialized(self):
        chat = Chat(size=5)
        chat.init_chat(_system("Be brief."))
        result = chat.to_response_api_chat()
        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["type"] == "message"
        assert result[0]["content"][0]["text"] == "Be brief."
        assert result[0]["content"][0]["type"] == "input_text"

    def test_system_message_empty_text_fallback(self):
        chat = Chat(size=5)
        sys_msg = RealtimeConversationItemSystemMessage(
            type="message",
            role="system",
            content=[SystemContent(type="input_text", text="")],
        )
        chat.init_chat(sys_msg)
        result = chat.to_response_api_chat()
        assert result[0]["content"][0]["text"] == "A helpful AI assistant."

    def test_user_text_message(self):
        chat = Chat(size=5)
        chat.add_item(_user("What is 2+2?"))
        result = chat.to_response_api_chat()
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["text"] == "What is 2+2?"

    def test_user_image_message(self):
        chat = Chat(size=5)
        msg = _user_msg_with_parts(("text", "Describe"), ("image", "http://img.png"))
        chat.add_item(msg)
        result = chat.to_response_api_chat()
        content = result[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "input_text"
        assert content[1]["type"] == "input_image"
        assert content[1]["image_url"] == "http://img.png"

    def test_assistant_message(self):
        chat = Chat(size=5)
        msg = make_assistant_message("Hello there.")
        msg.status = "completed"
        chat.add_item(msg)
        result = chat.to_response_api_chat()
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["id"] == msg.id
        assert result[0]["status"] == "completed"
        assert result[0]["content"][0]["text"] == "Hello there."

    def test_assistant_message_default_status(self):
        chat = Chat(size=5)
        msg = make_assistant_message("hi")
        chat.add_item(msg)
        result = chat.to_response_api_chat()
        assert result[0]["status"] == "completed"

    def test_function_call_with_id_and_status(self):
        chat = Chat(size=5)
        fc = _fc("c1", "search", '{"q": "test"}')
        fc.status = "completed"
        chat.add_item(fc)
        result = chat.to_response_api_chat()
        assert len(result) == 1
        entry = result[0]
        assert entry["type"] == "function_call"
        assert entry["call_id"] == "call_c1"
        assert entry["name"] == "search"
        assert entry["arguments"] == '{"q": "test"}'
        assert entry["id"] == fc.id
        assert entry["status"] == "completed"

    def test_function_call_without_optional_fields(self):
        chat = Chat(size=5)
        fc = _fc("c2", "noop")
        chat.add_item(fc)
        result = chat.to_response_api_chat()
        entry = result[0]
        assert entry["call_id"] == "call_c2"
        assert entry["id"] == fc.id
        assert "status" not in entry

    def test_function_call_output_with_id_and_status(self):
        chat = Chat(size=5)
        chat.add_item(_fc("c1"))
        fco = _fco("c1", '{"result": 42}')
        fco.status = "completed"
        chat.add_item(fco)
        result = chat.to_response_api_chat()
        entry = result[-1]
        assert entry["type"] == "function_call_output"
        assert entry["call_id"] == "call_c1"
        assert entry["output"] == '{"result": 42}'
        assert entry["id"] == fco.id
        assert entry["status"] == "completed"

    def test_function_call_output_without_optional_fields(self):
        chat = Chat(size=5)
        chat.add_item(_fc("c1"))
        fco = _fco("c1")
        chat.add_item(fco)
        result = chat.to_response_api_chat()
        entry = result[-1]
        assert entry["id"] == fco.id
        assert "status" not in entry

    def test_full_mixed_conversation(self):
        chat = Chat(size=10)
        chat.init_chat(_system("You are helpful."))
        chat.add_item(_user("Call my tool"))
        chat.add_item(_fc("c1", "tool_a", '{"x": 1}'))
        fco = _fco("c1", '{"y": 2}')
        chat.add_item(fco)
        chat.add_item(_assistant("Done."))

        result = chat.to_response_api_chat()
        assert len(result) == 5
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["type"] == "function_call"
        assert result[3]["type"] == "function_call_output"
        assert result[4]["role"] == "assistant"


# ===================================================================
# 8. TestToTransformersChat
# ===================================================================


class TestToTransformersChat:
    def test_empty_chat(self):
        chat = Chat(size=5)
        assert chat.to_transformers_chat() == []

    def test_system_message(self):
        chat = Chat(size=5)
        chat.init_chat(_system("Be concise."))
        result = chat.to_transformers_chat()
        assert result == [{"role": "system", "content": "Be concise."}]

    def test_user_text_only_produces_string_content(self):
        chat = Chat(size=5)
        chat.add_item(_user("hi there"))
        result = chat.to_transformers_chat()
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], str)
        assert result[0]["content"] == "hi there"

    def test_user_multi_text_parts_joined(self):
        chat = Chat(size=5)
        msg = _user_msg_with_parts(("text", "hello"), ("text", "world"))
        chat.add_item(msg)
        result = chat.to_transformers_chat()
        assert result[0]["content"] == "hello world"

    def test_user_with_images_produces_list_content(self):
        chat = Chat(size=5)
        msg = _user_msg_with_parts(("text", "look"), ("image", "http://img.png"))
        chat.add_item(msg)
        result = chat.to_transformers_chat()
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 2

    def test_assistant_message_text_joined(self):
        chat = Chat(size=5)
        msg = RealtimeConversationItemAssistantMessage(
            type="message",
            role="assistant",
            content=[
                AssistantContent(type="output_text", text="part1"),
                AssistantContent(type="output_text", text="part2"),
            ],
        )
        chat.add_item(msg)
        result = chat.to_transformers_chat()
        assert result[0] == {"role": "assistant", "content": "part1 part2"}

    def test_function_call_valid_json_args(self):
        chat = Chat(size=5)
        chat.add_item(_fc("c1", "search", '{"query": "test"}'))
        result = chat.to_transformers_chat()
        entry = result[0]
        assert entry["role"] == "assistant"
        assert len(entry["tool_calls"]) == 1
        tc = entry["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["id"] == "call_c1"
        assert tc["function"]["name"] == "search"
        assert tc["function"]["arguments"] == {"query": "test"}

    def test_function_call_invalid_json_falls_back(self):
        chat = Chat(size=5)
        chat.add_item(_fc("c1", "broken", "not valid json"))
        result = chat.to_transformers_chat()
        assert result[0]["tool_calls"][0]["function"]["arguments"] == {}

    def test_function_call_empty_string_args(self):
        chat = Chat(size=5)
        fc = _fc("c1", "f", "")
        chat.add_item(fc)
        result = chat.to_transformers_chat()
        assert result[0]["tool_calls"][0]["function"]["arguments"] == {}

    def test_function_call_output_resolves_name(self):
        chat = Chat(size=5)
        chat.add_item(_fc("c1", "lookup"))
        chat.add_item(_fco("c1", "result_data"))
        result = chat.to_transformers_chat()
        tool_entry = result[1]
        assert tool_entry["role"] == "tool"
        assert tool_entry["tool_call_id"] == "call_c1"
        assert tool_entry["name"] == "lookup"
        assert tool_entry["content"] == "result_data"

    def test_function_call_output_no_matching_call_empty_name(self):
        chat = Chat(size=5)
        fco = _fco("orphan_id", "data")
        fco.id = "fco_orphan"
        chat.buffer.append(fco)
        result = chat.to_transformers_chat()
        assert result[0]["name"] == ""

    def test_full_mixed_conversation(self):
        chat = Chat(size=10)
        chat.init_chat(_system("System prompt"))
        chat.add_item(_user("Do it"))
        chat.add_item(_fc("c1", "action", '{"a": 1}'))
        chat.add_item(_fco("c1", "done"))
        chat.add_item(_assistant("All set."))

        result = chat.to_transformers_chat()
        assert len(result) == 5
        assert result[0] == {"role": "system", "content": "System prompt"}
        assert result[1] == {"role": "user", "content": "Do it"}
        assert result[2]["role"] == "assistant"
        assert "tool_calls" in result[2]
        assert result[3]["role"] == "tool"
        assert result[3]["name"] == "action"
        assert result[4] == {"role": "assistant", "content": "All set."}


# ===================================================================
# 9. TestCopyAndReset
# ===================================================================


class TestCopyAndReset:
    def test_copy_buffer_independent(self):
        chat = Chat(size=5)
        chat.add_item(_user("original"))
        clone = chat.copy()
        clone.add_item(_user("extra"))
        assert len(chat.buffer) == 1
        assert len(clone.buffer) == 2

    def test_copy_preserves_init_chat_message(self):
        chat = Chat(size=5)
        sys_msg = _system("Keep it short.")
        chat.init_chat(sys_msg)
        clone = chat.copy()
        assert clone.init_chat_message is sys_msg

    def test_copy_preserves_pending_tool_calls_independently(self):
        chat = Chat(size=5)
        chat.add_item(_fc("c1"))
        clone = chat.copy()
        assert "call_c1" in clone._pending_tool_calls
        clone._pending_tool_calls.pop("call_c1")
        assert "call_c1" in chat._pending_tool_calls

    def test_copy_preserves_size(self):
        chat = Chat(size=7)
        clone = chat.copy()
        assert clone.size == 7

    def test_copy_preserves_user_turn_count(self):
        chat = Chat(size=5)
        chat.add_item(_user("u1"))
        chat.add_item(_user("u2"))
        clone = chat.copy()
        assert clone._user_turn_count == 2

    def test_reset_clears_everything(self):
        chat = Chat(size=5)
        chat.init_chat(_system("sys"))
        chat.add_item(_user("u"))
        chat.add_item(_fc("c1"))
        assert len(chat.buffer) > 0
        assert chat.init_chat_message is not None
        assert len(chat._pending_tool_calls) > 0
        assert chat._user_turn_count > 0

        chat.reset()
        assert chat.buffer == []
        assert chat.init_chat_message is None
        assert chat._pending_tool_calls == {}
        assert chat._user_turn_count == 0

    def test_reset_preserves_size(self):
        chat = Chat(size=3)
        chat.reset()
        assert chat.size == 3


# ===================================================================
# 10. TestStripImages
# ===================================================================


class TestStripImages:
    def test_multiple_user_messages_images_removed(self):
        chat = Chat(size=10)
        chat.add_item(_user_msg_with_parts(("text", "a"), ("image", "url1")))
        chat.add_item(_assistant("ok"))
        chat.add_item(_user_msg_with_parts(("text", "b"), ("image", "url2")))
        chat.strip_images()

        for item in chat.buffer:
            if isinstance(item, RealtimeConversationItemUserMessage):
                assert all(p.type != "input_image" for p in item.content)
                assert any(p.type == "input_text" for p in item.content)

    def test_no_user_messages_noop(self):
        chat = Chat(size=10)
        chat.add_item(_assistant("solo"))
        chat.add_item(_fc("c1"))
        chat.strip_images()
        assert len(chat.buffer) == 2

    def test_text_only_messages_unchanged(self):
        chat = Chat(size=10)
        chat.add_item(_user("just text"))
        chat.strip_images()
        assert chat.buffer[0].content[0].text == "just text"
        assert len(chat.buffer[0].content) == 1


# ===================================================================
# 11. TestMarkCallCompleted
# ===================================================================


class TestMarkCallCompleted:
    def test_none_status_sets_completed(self):
        chat = Chat(size=5)
        fc = _fc("c1")
        chat.add_item(fc)
        chat._mark_call_completed("call_c1", status=None)
        assert fc.status == "completed"

    def test_explicit_status_used(self):
        chat = Chat(size=5)
        fc = _fc("c1")
        chat.add_item(fc)
        chat._mark_call_completed("call_c1", status="incomplete")
        assert fc.status == "incomplete"

    def test_in_progress_status(self):
        chat = Chat(size=5)
        fc = _fc("c1")
        chat.add_item(fc)
        chat._mark_call_completed("call_c1", status="in_progress")
        assert fc.status == "in_progress"

    def test_no_match_is_noop(self):
        chat = Chat(size=5)
        chat.add_item(_fc("c1"))
        chat._mark_call_completed("nonexistent", status=None)
        assert chat.buffer[0].status is None

    def test_only_function_calls_checked(self):
        chat = Chat(size=5)
        chat.add_item(_user("hi"))
        chat.add_item(_fc("c1"))
        chat._mark_call_completed("call_c1")
        fc = next(e for e in chat.buffer if isinstance(e, RealtimeConversationItemFunctionCall))
        assert fc.status == "completed"
