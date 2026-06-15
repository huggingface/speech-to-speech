import json

from openai.types.realtime.conversation_item import (
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
    RealtimeConversationItemUserMessage,
)
from openai.types.realtime.realtime_conversation_item_user_message import Content as UserContent

from speech_to_speech.LLM.chat import Chat
from speech_to_speech.LLM.vision_router import VisionRouter


def _user_with_image(text: str, image_url: str) -> RealtimeConversationItemUserMessage:
    return RealtimeConversationItemUserMessage(
        type="message",
        role="user",
        content=[
            UserContent(type="input_text", text=text),
            UserContent(type="input_image", image_url=image_url),
        ],
    )


def _camera_call(call_id: str = "call_camera") -> RealtimeConversationItemFunctionCall:
    return RealtimeConversationItemFunctionCall(
        type="function_call",
        id="fc_camera",
        call_id=call_id,
        name="camera",
        arguments='{"question": "What color is the object?"}',
    )


def _camera_output(call_id: str = "call_camera") -> RealtimeConversationItemFunctionCallOutput:
    return RealtimeConversationItemFunctionCallOutput(
        type="function_call_output",
        id="fco_camera",
        call_id=call_id,
        output='{"b64_im": "abc123", "ok": true}',
    )


def test_routes_user_image_message_to_text_observation() -> None:
    seen: list[tuple[list[str], str]] = []

    def describe(image_urls: list[str], question: str) -> str:
        seen.append((image_urls, question))
        return "A red cup is on the table."

    chat = Chat(size=5)
    chat.add_item(_user_with_image("What is in front of you?", "data:image/jpeg;base64,zzz"))

    routed = VisionRouter(describe).process_chat(chat)

    assert routed == 1
    assert seen == [(["data:image/jpeg;base64,zzz"], "What is in front of you?")]
    user_item = chat.buffer[0]
    assert isinstance(user_item, RealtimeConversationItemUserMessage)
    assert len(user_item.content) == 1
    assert user_item.content[0].type == "input_text"
    assert user_item.content[0].text is not None
    assert "What is in front of you?" in user_item.content[0].text
    assert "Image analysis: A red cup is on the table." in user_item.content[0].text


def test_routes_camera_b64_tool_output_with_tool_question() -> None:
    seen: list[tuple[list[str], str]] = []

    def describe(image_urls: list[str], question: str) -> str:
        seen.append((image_urls, question))
        return "The object is blue."

    chat = Chat(size=5)
    chat.add_item(_camera_call())
    chat.add_item(_camera_output())

    routed = VisionRouter(describe).process_chat(chat)

    assert routed == 1
    assert seen == [(["data:image/jpeg;base64,abc123"], "What color is the object?")]
    output_item = chat.buffer[1]
    assert isinstance(output_item, RealtimeConversationItemFunctionCallOutput)
    payload = json.loads(output_item.output)
    assert payload == {
        "ok": True,
        "image_description": "The object is blue.",
        "question": "What color is the object?",
    }


def test_skips_text_only_chat() -> None:
    calls = 0

    def describe(_image_urls: list[str], _question: str) -> str:
        nonlocal calls
        calls += 1
        return "unused"

    chat = Chat(size=5)
    chat.add_item(
        RealtimeConversationItemUserMessage(
            type="message",
            role="user",
            content=[UserContent(type="input_text", text="hello")],
        )
    )

    assert VisionRouter(describe).process_chat(chat) == 0
    assert calls == 0
