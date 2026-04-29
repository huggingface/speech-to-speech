from __future__ import annotations

import json
import logging
from typing import Any, Literal, Union

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
from openai.types.realtime.realtime_conversation_item_system_message import Content as SystemContent
from openai.types.realtime.realtime_conversation_item_user_message import Content as UserContent
from openai.types.responses.response_input_image_param import ResponseInputImageParam
from openai.types.responses.response_input_message_content_list_param import (
    ResponseInputMessageContentListParam,
)
from openai.types.responses.response_input_param import (
    FunctionCallOutput,
    ResponseFunctionToolCallParam,
    ResponseInputItemParam,
    ResponseInputParam,
    ResponseOutputMessageParam,
)
from openai.types.responses.response_input_param import (
    Message as ResponseMessage,
)
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_output_text_param import ResponseOutputTextParam
from pydantic import BaseModel

from speech_to_speech.utils.utils import _generate_id

logger = logging.getLogger(__name__)


class ChatItemError(Exception):
    """Raised when a conversation item fails validation in :meth:`Chat.add_item`."""


def _ensure_id(value: str | None, prefix: str) -> str:
    if value is None:
        return _generate_id(prefix)
    if not value.startswith(f"{prefix}_"):
        raise ChatItemError(f"ID must start with '{prefix}_', got {value!r}")
    return value


SupportedItem = Union[
    RealtimeConversationItemSystemMessage,
    RealtimeConversationItemUserMessage,
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
]


class Chat:
    """Manages conversation history with bounded size to avoid OOM issues.

    The buffer stores ``ConversationItem`` objects (user messages, assistant
    messages, function calls, function call outputs).  System messages are
    stored separately in ``init_chat_message`` and never placed in the buffer.
    """

    def __init__(self, size: int) -> None:
        self.size = size
        self.init_chat_message: RealtimeConversationItemSystemMessage | None = None
        # ``size`` is the number of user turns to keep.  When exceeded the
        # oldest complete turn (everything up to the next user message)
        # is evicted.
        self.buffer: list[SupportedItem] = []
        self._pending_tool_calls: dict[str, RealtimeConversationItemFunctionCall] = {}
        self._user_turn_count: int = 0

    def _evict_oldest_turn(self) -> None:
        """Remove items from the front until the next user message boundary."""
        if not self.buffer:
            return
        first = self.buffer.pop(0)
        if isinstance(first, RealtimeConversationItemUserMessage):
            self._user_turn_count -= 1
        while self.buffer and not isinstance(self.buffer[0], RealtimeConversationItemUserMessage):
            self.buffer.pop(0)

    def _has_call_id_in_buffer(self, call_id: str) -> bool:
        for entry in self.buffer:
            if isinstance(entry, RealtimeConversationItemFunctionCall) and entry.call_id == call_id:
                return True
        return False

    def _mark_call_completed(
        self, call_id: str, status: Literal["completed", "incomplete", "in_progress"] | None = None
    ) -> None:
        """Set ``status`` to ``"completed"`` on the matching function_call."""
        for entry in self.buffer:
            if isinstance(entry, RealtimeConversationItemFunctionCall) and entry.call_id == call_id:
                entry.status = "completed" if status is None else status
                return

    def append_tool_output(self, call_id: str, output_item: RealtimeConversationItemFunctionCallOutput) -> None:
        """Append a ``function_call_output``, re-injecting its ``function_call`` if evicted.

        Also marks the paired ``function_call`` as ``"completed"`` if its
        status was ``None``.

        Raises :class:`ChatItemError` if *call_id* is unknown.
        """
        if self._has_call_id_in_buffer(call_id):
            self._pending_tool_calls.pop(call_id, None)
            self._mark_call_completed(call_id, output_item.status)
            self.buffer.append(output_item)
            return

        if call_id in self._pending_tool_calls:
            logger.info("Re-injecting evicted function_call for call_id=%s", call_id)
            fc = self._pending_tool_calls.pop(call_id)
            fc.status = "completed" if output_item.status is None else output_item.status
            self.buffer.append(fc)
            self.buffer.append(output_item)
            return

        raise ChatItemError(f"No function_call with call_id '{call_id}' found in conversation history.")

    def init_chat(self, message: RealtimeConversationItemSystemMessage) -> None:
        self.init_chat_message = message

    def add_item(self, item: SupportedItem) -> SupportedItem:
        """Validate and route a conversation item into the chat.

        Raises :class:`ChatItemError` if the item fails validation.
        """

        if isinstance(item, RealtimeConversationItemSystemMessage):
            item.id = _ensure_id(item.id, "sys")
            self.init_chat(item)
            logger.debug("Set system message via conversation item")

        elif isinstance(item, RealtimeConversationItemUserMessage):
            item.id = _ensure_id(item.id, "msg")
            item.content = [
                p
                for p in item.content
                if (p.type == "input_text" and p.text) or (p.type == "input_image" and p.image_url)
            ]
            if not item.content:
                raise ChatItemError("Message has no supported content. Supported modalities: input_text, input_image.")
            self.buffer.append(item)
            self._user_turn_count += 1
            logger.debug("Added user message to chat (%d parts)", len(item.content))

        elif isinstance(item, RealtimeConversationItemAssistantMessage):
            item.id = _ensure_id(item.id, "msg")
            item.content = [p for p in item.content if p.type == "output_text" and p.text]
            if not item.content:
                return item
            self.buffer.append(item)
            logger.debug("Added assistant message to chat (%d parts)", len(item.content))

        elif isinstance(item, RealtimeConversationItemFunctionCall):
            item.id = _ensure_id(item.id, "fc")
            item.call_id = _ensure_id(item.call_id, "call")
            self.buffer.append(item)
            self._pending_tool_calls[item.call_id] = item
            logger.debug("Added function_call to chat (call_id=%s)", item.call_id)

        elif isinstance(item, RealtimeConversationItemFunctionCallOutput):
            item.id = _ensure_id(item.id, "fco")
            self.append_tool_output(item.call_id, item)
            logger.debug("Added function_call_output to chat (call_id=%s)", item.call_id)

        else:
            raise ChatItemError(f"Unsupported item type: {getattr(item, 'type', None)}")

        while self._user_turn_count > self.size:
            self._evict_oldest_turn()

        return item

    def to_response_api_chat(self) -> ResponseInputParam:
        """Serialize the full chat (system prompt + buffer) for the OpenAI Responses API."""
        result: list[ResponseInputItemParam] = []
        if self.init_chat_message:
            result.append(
                ResponseMessage(
                    content=[
                        ResponseInputTextParam(text=p.text or "A helpful AI assistant.", type="input_text")
                        for p in self.init_chat_message.content
                    ],
                    role="system",
                    type="message",
                )
            )
        for item in self.buffer:
            assert item.id is not None and item.id != "", f"item.id is {item.id}"
            if isinstance(item, RealtimeConversationItemUserMessage):
                content: ResponseInputMessageContentListParam = []
                for user_part in item.content:
                    if user_part.type == "input_text" and user_part.text is not None:
                        content.append(ResponseInputTextParam(text=user_part.text or "", type="input_text"))
                    elif user_part.type == "input_image" and user_part.image_url is not None:
                        img = ResponseInputImageParam(type="input_image", detail=user_part.detail or "auto")
                        if user_part.image_url is not None:
                            img["image_url"] = user_part.image_url
                        content.append(img)
                if content:
                    result.append(ResponseMessage(content=content, role="user", type="message"))
            elif isinstance(item, RealtimeConversationItemAssistantMessage):
                assistant_content: list[ResponseOutputTextParam] = []
                for assistant_part in item.content:
                    if assistant_part.type == "output_text" and assistant_part.text is not None:
                        assistant_content.append(
                            ResponseOutputTextParam(text=assistant_part.text, type="output_text", annotations=[])
                        )
                if assistant_content:
                    result.append(
                        ResponseOutputMessageParam(
                            id=item.id,
                            content=assistant_content,
                            role="assistant",
                            status=item.status or "completed",
                            type="message",
                        )
                    )
            elif isinstance(item, RealtimeConversationItemFunctionCall) and item.call_id is not None:
                assert item.call_id is not None and item.call_id != ""
                function_call = ResponseFunctionToolCallParam(
                    arguments=item.arguments,
                    call_id=item.call_id,
                    name=item.name,
                    type="function_call",
                    id=item.id,
                )
                if item.id is not None:
                    function_call["id"] = item.id
                if item.status is not None:
                    function_call["status"] = item.status
                result.append(function_call)
            elif isinstance(item, RealtimeConversationItemFunctionCallOutput):
                function_call_output = FunctionCallOutput(
                    call_id=item.call_id,
                    output=item.output,
                    type="function_call_output",
                )
                if item.id is not None:
                    function_call_output["id"] = item.id
                if item.status is not None:
                    function_call_output["status"] = item.status
                result.append(function_call_output)
        return result

    def to_transformers_chat(self) -> list[dict[str, Any]]:
        """Serialize the full chat for HuggingFace transformers ``apply_chat_template``.

        User messages with only text produce a plain string ``content`` value.
        User messages containing images keep ``content`` as a list of dicts so
        VLM pipelines can process them.
        """
        messages: list[TransformersChatMessage] = []
        if self.init_chat_message:
            text = " ".join(p.text for p in self.init_chat_message.content if p.text)
            messages.append(TransformersSystemMessage(content=text))
        for item in self.buffer:
            if isinstance(item, RealtimeConversationItemUserMessage):
                has_images = any(p.type == "input_image" for p in item.content)
                if has_images:
                    messages.append(
                        TransformersUserMessage(content=[p.model_dump(exclude_none=True) for p in item.content])
                    )
                else:
                    text = " ".join(p.text for p in item.content if p.type == "input_text" and p.text)
                    messages.append(TransformersUserMessage(content=text))
            elif isinstance(item, RealtimeConversationItemAssistantMessage):
                text = " ".join(p.text for p in item.content if p.text)
                messages.append(TransformersAssistantMessage(content=text))
            elif isinstance(item, RealtimeConversationItemFunctionCall):
                assert item.call_id is not None and item.call_id != ""
                args: Any = item.arguments
                try:
                    args = json.loads(args) if isinstance(args, str) else args
                except (json.JSONDecodeError, TypeError):
                    args = {}
                messages.append(
                    TransformersFunctionCallMessage(
                        tool_calls=[
                            TransformersToolCall(
                                id=item.call_id,
                                function=TransformersToolCallFunction(name=item.name, arguments=args),
                            )
                        ]
                    )
                )
            elif isinstance(item, RealtimeConversationItemFunctionCallOutput):
                name = ""
                for prev in reversed(messages):
                    if isinstance(prev, TransformersFunctionCallMessage):
                        for tc in prev.tool_calls:
                            if tc.id == item.call_id:
                                name = tc.function.name
                                break
                        if name:
                            break
                messages.append(
                    TransformersToolMessage(
                        tool_call_id=item.call_id,
                        name=name,
                        content=item.output,
                    )
                )
        return [m.model_dump() for m in messages]

    def copy(self) -> Chat:
        """Return a shallow snapshot safe for concurrent read access."""
        clone = Chat(self.size)
        clone.init_chat_message = self.init_chat_message
        clone.buffer = list(self.buffer)
        clone._pending_tool_calls = dict(self._pending_tool_calls)
        clone._user_turn_count = self._user_turn_count
        return clone

    def reset(self) -> None:
        self.buffer = []
        self.init_chat_message = None
        self._pending_tool_calls = {}
        self._user_turn_count = 0

    def strip_images(self) -> None:
        """Remove all image content parts from user messages in the buffer.

        Called after appending the assistant response so images don't persist
        across turns.
        """
        for item in self.buffer:
            if isinstance(item, RealtimeConversationItemUserMessage):
                item.content = [p for p in item.content if p.type != "input_image"]


# ---------------------------------------------------------------------------
# Transformers chat message models
# ---------------------------------------------------------------------------


class TransformersToolCallFunction(BaseModel):
    name: str
    arguments: dict[str, Any]


class TransformersToolCall(BaseModel):
    type: Literal["function"] = "function"
    id: str
    function: TransformersToolCallFunction


class TransformersSystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str


class TransformersUserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[dict[str, Any]]


class TransformersAssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class TransformersFunctionCallMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    tool_calls: list[TransformersToolCall]


class TransformersToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    name: str
    content: str


TransformersChatMessage = Union[
    TransformersSystemMessage,
    TransformersUserMessage,
    TransformersAssistantMessage,
    TransformersFunctionCallMessage,
    TransformersToolMessage,
]


# ---------------------------------------------------------------------------
# Factory helpers -- hide verbose constructors behind simple calls
# ---------------------------------------------------------------------------


def make_user_message(text: str) -> RealtimeConversationItemUserMessage:
    return RealtimeConversationItemUserMessage(
        type="message",
        role="user",
        content=[UserContent(type="input_text", text=text)],
    )


def make_assistant_message(text: str) -> RealtimeConversationItemAssistantMessage:
    return RealtimeConversationItemAssistantMessage(
        type="message",
        role="assistant",
        content=[AssistantContent(type="output_text", text=text)],
    )


def make_system_message(text: str) -> RealtimeConversationItemSystemMessage:
    return RealtimeConversationItemSystemMessage(
        type="message",
        role="system",
        content=[SystemContent(type="input_text", text=text)],
    )
