"""Unit tests for language_model helpers that shape messages before
apply_chat_template, independent of any loaded model/tokenizer.
"""

from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.language_model import _ensure_user_turn


class TestEnsureUserTurn:
    def test_leaves_messages_with_a_user_turn_unchanged(self):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi there."},
        ]
        assert _ensure_user_turn(messages) == messages

    def test_appends_placeholder_user_turn_when_none_present(self):
        messages = [{"role": "system", "content": "Be helpful."}]
        result = _ensure_user_turn(messages)
        assert result[:-1] == messages
        assert result[-1]["role"] == "user"
        assert result[-1]["content"]

    def test_appends_when_messages_only_contain_assistant_turns(self):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = _ensure_user_turn(messages)
        assert result[-1]["role"] == "user"

    def test_appends_when_only_tool_turns_follow_the_system_message(self):
        """A tool result alone is not a user turn, so the template still needs one."""
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "tool", "tool_call_id": "call_1", "name": "f", "content": "42"},
        ]
        result = _ensure_user_turn(messages)
        assert result[-1]["role"] == "user"

    def test_does_not_mutate_the_input_list(self):
        messages = [{"role": "system", "content": "Be helpful."}]
        _ensure_user_turn(messages)
        assert messages == [{"role": "system", "content": "Be helpful."}]

    def test_real_serialized_user_turn_is_recognised(self):
        """Guards the coupling to Chat.to_transformers_chat(): the role checked
        here has to be the one the serializer actually emits, or a real user
        turn goes unnoticed and a placeholder is appended on every call.
        """
        chat = Chat(size=10)
        chat.add_item(make_user_message("Hi there."))
        serialized = chat.to_transformers_chat()

        assert _ensure_user_turn(serialized) == serialized

    def test_empty_serialized_chat_gets_a_placeholder(self):
        chat = Chat(size=10)
        serialized = chat.to_transformers_chat()

        result = _ensure_user_turn(serialized)
        assert [m["role"] for m in result] == ["user"]
