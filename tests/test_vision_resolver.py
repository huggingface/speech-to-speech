from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from openai.types.realtime.conversation_item import (
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemUserMessage,
)
from openai.types.realtime.realtime_conversation_item_user_message import Content as UserContent

from speech_to_speech.LLM.chat import Chat
from speech_to_speech.LLM.vision_resolver import VisionResolver


@pytest.fixture
def mock_openai():
    with patch("speech_to_speech.LLM.vision_resolver.OpenAI") as mock_cls:
        client_mock = MagicMock()
        mock_cls.return_value = client_mock
        yield client_mock


class TestVisionResolver:
    def test_resolve_success(self, mock_openai):
        completion_mock = MagicMock()
        completion_mock.choices = [MagicMock(message=MagicMock(content="A detailed description of the cat."))]
        completion_mock.usage = MagicMock(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        mock_openai.chat.completions.create.return_value = completion_mock

        resolver = VisionResolver(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            max_tokens=200,
            timeout_s=5.0,
        )

        res = resolver.resolve("data:image/jpeg;base64,12345", "What is in this image?")
        assert res == "A detailed description of the cat."

        mock_openai.chat.completions.create.assert_called_once()
        kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["max_tokens"] == 200
        assert kwargs["timeout"] == 5.0

        messages = kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"][0] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,12345"}}
        assert messages[1]["content"][1] == {"type": "text", "text": "What is in this image?"}

    def test_resolve_failure_fallback(self, mock_openai):
        mock_openai.chat.completions.create.side_effect = RuntimeError("API connection error")

        resolver = VisionResolver(model_name="gpt-4o-mini")
        res = resolver.resolve("http://example.com/photo.jpg", "What is this?")
        assert res == "image could not be analyzed in time"

    def test_resolve_stale_cancel_scope(self, mock_openai):
        cancel_scope = MagicMock()
        cancel_scope.is_stale.return_value = True

        resolver = VisionResolver(model_name="gpt-4o-mini")
        res = resolver.resolve("http://example.com/photo.jpg", "What is this?", cancel_scope=cancel_scope)
        assert res == "image resolution cancelled"
        mock_openai.chat.completions.create.assert_not_called()


class TestChatResolveImages:
    def test_resolve_images_with_question_in_same_message(self):
        chat = Chat(30)
        user_msg = RealtimeConversationItemUserMessage(
            type="message",
            role="user",
            content=[
                UserContent(type="input_text", text="What plant is this?"),
                UserContent(type="input_image", image_url="http://example.com/plant.jpg"),
            ],
        )
        chat.add_item(user_msg)

        resolver = MagicMock()
        resolver.resolve.return_value = "A healthy Monstera plant."

        chat.resolve_images(resolver)

        resolver.resolve.assert_called_once_with(
            ["http://example.com/plant.jpg"], "What plant is this?", cancel_scope=None
        )

        item = chat.buffer[0]
        assert len(item.content) == 2
        assert item.content[0].type == "input_text"
        assert item.content[0].text == "What plant is this?"
        assert item.content[1].type == "input_text"
        assert item.content[1].text == "[Camera observation] A healthy Monstera plant."

    def test_resolve_images_fallback_to_preceding_camera_tool_call(self):
        chat = Chat(30)
        fc = RealtimeConversationItemFunctionCall(
            type="function_call",
            id="fc_1",
            call_id="call_1",
            name="camera",
            arguments='{"question": "What object am I holding?"}',
        )
        chat.add_item(fc)

        user_msg = RealtimeConversationItemUserMessage(
            type="message",
            role="user",
            content=[
                UserContent(type="input_image", image_url="http://example.com/holding.jpg"),
            ],
        )
        chat.add_item(user_msg)

        resolver = MagicMock()
        resolver.resolve.return_value = "A blue mug."

        chat.resolve_images(resolver)

        resolver.resolve.assert_called_once_with(
            ["http://example.com/holding.jpg"], "What object am I holding?", cancel_scope=None
        )

        user_item = chat.buffer[0]
        assert len(user_item.content) == 1
        assert user_item.content[0].type == "input_text"
        assert user_item.content[0].text == "[Camera observation] A blue mug."

    def test_resolve_images_fallback_to_default_question(self):
        chat = Chat(30)
        user_msg = RealtimeConversationItemUserMessage(
            type="message",
            role="user",
            content=[
                UserContent(type="input_image", image_url="http://example.com/view.jpg"),
            ],
        )
        chat.add_item(user_msg)

        resolver = MagicMock()
        resolver.resolve.return_value = "A desk with a laptop."

        chat.resolve_images(resolver)

        resolver.resolve.assert_called_once_with(
            ["http://example.com/view.jpg"], "Describe what is relevant in this image.", cancel_scope=None
        )

    def test_resolved_images_persist_after_strip_images(self):
        chat = Chat(30)
        user_msg = RealtimeConversationItemUserMessage(
            type="message",
            role="user",
            content=[
                UserContent(type="input_image", image_url="http://example.com/photo.jpg"),
            ],
        )
        chat.add_item(user_msg)

        resolver = MagicMock()
        resolver.resolve.return_value = "A red apple."

        chat.resolve_images(resolver)
        chat.strip_images()

        item = chat.buffer[0]
        assert len(item.content) == 1
        assert item.content[0].type == "input_text"
        assert item.content[0].text == "[Camera observation] A red apple."
