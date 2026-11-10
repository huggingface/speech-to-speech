from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from typing import Any, cast

from openai import Stream
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.chat.chat_completion_named_tool_choice_param import Function as NamedToolChoiceFunction
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)
from openai.types.responses import ResponseFunctionToolCall
from openai.types.shared_params import FunctionDefinition

from speech_to_speech.LLM.base_openai_compatible_language_model import (
    WARMUP_MAX_RETRIES,
    AssistantMessage,
    BaseOpenAICompatibleHandler,
    ProviderEvent,
    ProviderResponseEnd,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    Usage,
)
from speech_to_speech.LLM.chat import Chat
from speech_to_speech.LLM.compaction_prompt import CompactGenerateFn
from speech_to_speech.LLM.tool_call.function_call import JSON_TOOL_CALL_END, JSON_TOOL_CALL_ENTER
from speech_to_speech.utils.utils import _generate_id

logger = logging.getLogger(__name__)


def _chat_reasoning_text(message: Any) -> str | None:
    # Compatible providers use either name. Never interpret opaque objects or
    # encrypted provider payloads as readable reasoning.
    for name in ("reasoning_content", "reasoning"):
        text = getattr(message, name, None)
        if isinstance(text, str) and text:
            return text
    return None


def _to_chat_tools(req_tools: Any) -> list[ChatCompletionToolParam] | None:
    """Convert Responses-API function tools to Chat-Completions tool format.

    Responses tools are flat ``{type:"function", name, description, parameters}``;
    Chat Completions nests them under a ``function`` key. Items already in the
    nested form (or non-function tools) are passed through untouched.
    """
    if not req_tools:
        return None
    chat_tools: list[ChatCompletionToolParam] = []
    for t in req_tools:
        d = t if isinstance(t, dict) else t.model_dump(exclude_none=True)
        if d.get("type") == "function" and "function" not in d:
            fn = FunctionDefinition(name=d["name"])
            if d.get("description") is not None:
                fn["description"] = d["description"]
            if d.get("parameters") is not None:
                fn["parameters"] = d["parameters"]
            chat_tools.append(ChatCompletionToolParam(type="function", function=fn))
        else:
            chat_tools.append(cast("ChatCompletionToolParam", d))
    return chat_tools


def _to_chat_tool_choice(tool_choice: Any) -> ChatCompletionToolChoiceOptionParam:
    """Convert a Responses-API tool_choice to Chat-Completions form.

    The string forms ("auto"/"required"/"none") are identical across both APIs;
    only the forced-function object differs (flat ``name`` vs nested ``function``).
    """
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function" and "name" in tool_choice:
        return ChatCompletionNamedToolChoiceParam(
            type="function", function=NamedToolChoiceFunction(name=tool_choice["name"])
        )
    if tool_choice is not None and not isinstance(tool_choice, (str, dict)):
        d = tool_choice.model_dump(exclude_none=True)
        if d.get("type") == "function" and "name" in d:
            return ChatCompletionNamedToolChoiceParam(type="function", function=NamedToolChoiceFunction(name=d["name"]))
        return cast("ChatCompletionToolChoiceOptionParam", d)
    return cast("ChatCompletionToolChoiceOptionParam", tool_choice)


def _build_chat_optional_kwargs(
    req_tools: Any,
    req_tool_choice: Any,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Build Chat Completions tool arguments from the Responses-style session config.

    A client-selected ``reasoning_effort`` replaces the startup ``extra_body``,
    matching the precedence of ``_build_extra_body`` (effort wins over the
    provider-specific ``enable_thinking`` flag).
    """
    optional_kwargs: dict[str, Any] = {}
    if reasoning_effort:
        optional_kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
    chat_tools = _to_chat_tools(req_tools)
    if chat_tools is not None:
        optional_kwargs["tools"] = chat_tools
    if req_tool_choice is not None:
        optional_kwargs["tool_choice"] = _to_chat_tool_choice(req_tool_choice)
    return optional_kwargs


def _to_chat_content_part(
    part: dict[str, Any],
    audio_content_type: str = "input_audio",
) -> ChatCompletionContentPartParam:
    """Convert one Realtime/transformers content part to Chat Completions shape."""
    ptype = part.get("type")
    if ptype == "input_text":
        return ChatCompletionContentPartTextParam(type="text", text=part.get("text") or "")
    if ptype == "input_image":
        raw_url: Any = part.get("image_url")
        if isinstance(raw_url, dict):
            image_url = cast("ImageURL", raw_url)
        else:
            image_url = ImageURL(url=raw_url)
            detail = part.get("detail")
            if detail is not None:
                image_url["detail"] = detail
        return ChatCompletionContentPartImageParam(type="image_url", image_url=image_url)
    if ptype == "input_audio":
        audio_b64 = part.get("audio") or ""
        if audio_content_type == "audio_url":
            return cast(
                "ChatCompletionContentPartParam",
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": f"data:audio/wav;base64,{audio_b64}",
                    },
                },
            )
        return cast(
            "ChatCompletionContentPartParam",
            {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_b64,
                    "format": "wav",
                },
            },
        )
    return cast("ChatCompletionContentPartParam", part)


def _chat_messages(
    chat: Chat,
    audio_content_type: str = "input_audio",
) -> list[dict[str, Any]]:
    """Serialise chat history, including media and tool messages, for Chat Completions."""
    messages = chat.to_transformers_chat()
    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            fn = tool_call.get("function")
            if fn is not None and not isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.dumps(fn.get("arguments") or {}, ensure_ascii=False)
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [
                _to_chat_content_part(part, audio_content_type=audio_content_type) for part in content
            ]
        if message.get("role") == "tool":
            message.pop("name", None)
    return messages


def _request_chat_completions(
    *,
    client: Any,
    model_name: str,
    messages: list[dict[str, Any]],
    stream: bool,
    extra_body: dict[str, Any] | None,
    timeout: Any,
    optional_kwargs: dict[str, Any],
) -> Any:
    """Issue a Chat Completions request with consistent streaming usage accounting.

    ``optional_kwargs`` may carry an ``extra_body`` that replaces the startup one
    for this request (a client-selected reasoning effort).
    """
    create_kwargs = dict(optional_kwargs)
    extra_body = create_kwargs.pop("extra_body", extra_body)
    if stream:
        create_kwargs["stream_options"] = {"include_usage": True}
    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=stream,
        extra_body=extra_body,
        timeout=timeout,
        **create_kwargs,
    )


def _tool_calls_from_accum(tool_accum: dict[int, dict[str, str]]) -> Iterator[ToolCall]:
    """Turn accumulated Chat Completions tool deltas into normalized tool-call events."""
    for index in sorted(tool_accum):
        entry = tool_accum[index]
        if not entry["name"]:
            continue
        yield ToolCall(
            item=ResponseFunctionToolCall(
                type="function_call",
                name=entry["name"],
                arguments=entry["args"] or "{}",
                call_id=_generate_id("call"),
                id=_generate_id("fc"),
                status="completed",
            )
        )


def _chat_response_end(finish_reason: str) -> ProviderResponseEnd:
    if finish_reason == "length":
        return ProviderResponseEnd(status="incomplete", reason="max_output_tokens")
    if finish_reason == "content_filter":
        return ProviderResponseEnd(status="incomplete", reason="content_filter")
    return ProviderResponseEnd()


_TOOL_CALL_TAGS = (JSON_TOOL_CALL_ENTER, JSON_TOOL_CALL_END)


class _ToolCallMarkupFilter:
    """Remove bare ``<tool_call>``/``</tool_call>`` tags that a provider streams as content.

    Some servers (e.g. vLLM with a Qwen XML tool parser) return structured
    ``tool_calls`` and also leak the opening tag as text. Tags may be split
    across chunks, so a tail that could still become a tag is held back, as is
    whitespace before any real text (it is dropped if a tag follows it).
    Whitespace after a tag is dropped; whitespace after real text is kept.
    """

    def __init__(self) -> None:
        self._pending = ""
        self._after_tag = False
        self._spoke = False
        self.stripped = False

    @property
    def pending(self) -> bool:
        return bool(self._pending)

    def feed(self, text: str) -> tuple[str, str]:
        """Return ``(held, new)``: cleaned output from previously held text and from ``text``."""
        buf = self._pending + text
        boundary = len(self._pending)
        self._pending = ""
        held: list[str] = []
        new: list[str] = []

        def emit(start: int, end: int) -> None:
            if start < boundary:
                held.append(buf[start : min(end, boundary)])
            if end > boundary:
                new.append(buf[max(start, boundary) : end])

        n = len(buf)
        i = 0
        while i < n:
            lead = i
            while i < n and buf[i].isspace():
                i += 1
            if self._spoke and not self._after_tag:
                emit(lead, i)
                lead = i
            if self._after_tag:
                lead = i
            if i == n:
                self._pending = buf[lead:]
                break
            rest = buf[i:]
            tag = next((t for t in _TOOL_CALL_TAGS if rest.startswith(t)), None)
            if tag is not None:
                i += len(tag)
                self._after_tag = True
                self.stripped = True
                continue
            if any(t.startswith(rest) for t in _TOOL_CALL_TAGS):
                self._pending = buf[lead:]
                break
            end = buf.find("<", i + 1)
            if end == -1:
                end = n
            emit(lead, end)
            i = end
            self._after_tag = False
            self._spoke = True
        return "".join(held), "".join(new)

    def flush(self) -> str:
        pending, self._pending = self._pending, ""
        return pending

    def keeps(self, segment: str) -> bool:
        """Whether a cleaned text segment is worth storing as an assistant message."""
        return bool(segment.strip()) if self.stripped else bool(segment)


def _iter_chat_stream_events(api_response: Stream[ChatCompletionChunk]) -> Iterator[ProviderEvent]:
    """Normalize a streaming Chat Completions response."""
    tool_accum: dict[int, dict[str, str]] = {}
    usage: Usage | None = None
    text_segment = ""
    ending: ProviderResponseEnd | None = None
    markup = _ToolCallMarkupFilter()

    def flush_text_segment() -> Iterator[ProviderEvent]:
        nonlocal text_segment
        if markup.keeps(text_segment):
            yield AssistantMessage(content=[AssistantContent(type="output_text", text=text_segment)])
        text_segment = ""

    def flush_tools() -> Iterator[ProviderEvent]:
        yield from flush_text_segment()
        yield from _tool_calls_from_accum(tool_accum)
        tool_accum.clear()

    def emit_text(text: str) -> Iterator[ProviderEvent]:
        nonlocal text_segment
        if not text:
            return
        if tool_accum:
            yield from flush_tools()
        text_segment += text
        yield TextDelta(text=text)

    def accumulate_tools(tool_calls: Any) -> None:
        for tool_call in tool_calls or []:
            entry = tool_accum.setdefault(tool_call.index, {"name": "", "args": "", "id": ""})
            if tool_call.id:
                entry["id"] = tool_call.id
            if tool_call.function is not None:
                if tool_call.function.name:
                    entry["name"] = tool_call.function.name
                if tool_call.function.arguments:
                    entry["args"] += tool_call.function.arguments

    deferred_tools: list[Any] = []

    def handle_tools(tool_calls: Any) -> None:
        if not tool_calls:
            return
        if markup.pending or deferred_tools:
            deferred_tools.extend(tool_calls)
        else:
            accumulate_tools(tool_calls)

    def release_deferred_tools() -> None:
        accumulate_tools(deferred_tools)
        deferred_tools.clear()

    def handle_text(text: str) -> Iterator[ProviderEvent]:
        held, new = markup.feed(text)
        yield from emit_text(held)
        release_deferred_tools()
        yield from emit_text(new)

    for chunk in api_response:
        if chunk.usage is not None:
            usage = Usage(
                input_tokens=chunk.usage.prompt_tokens or 0,
                output_tokens=chunk.usage.completion_tokens or 0,
            )
        if not chunk.choices:
            continue
        finish_reason = getattr(chunk.choices[0], "finish_reason", None)
        if finish_reason is not None:
            ending = _chat_response_end(finish_reason)
            yield ending
        delta = chunk.choices[0].delta
        reasoning = _chat_reasoning_text(delta)
        if reasoning:
            yield ReasoningDelta(text=reasoning)
        text_piece = delta.content or getattr(delta, "refusal", None)
        continuing_tool = bool(tool_accum)
        if continuing_tool:
            handle_tools(delta.tool_calls)
        if text_piece:
            yield from handle_text(text_piece)
        if not continuing_tool:
            handle_tools(delta.tool_calls)

    yield from emit_text(markup.flush())
    release_deferred_tools()
    if ending is None:
        logger.warning("Chat Completions stream ended without a finish_reason")
    if tool_accum and (ending is None or ending.status == "completed"):
        yield from flush_tools()
    yield from flush_text_segment()
    if usage is not None:
        yield usage


def _iter_chat_response_events(api_response: Any) -> Iterator[ProviderEvent]:
    """Normalize a non-streaming Chat Completions response."""
    usage = api_response.usage
    if usage:
        yield Usage(input_tokens=usage.prompt_tokens or 0, output_tokens=usage.completion_tokens or 0)
    if api_response.choices:
        finish_reason = getattr(api_response.choices[0], "finish_reason", None)
        if finish_reason is not None:
            yield _chat_response_end(finish_reason)
    message = api_response.choices[0].message if api_response.choices else None
    if message is None:
        return
    reasoning = _chat_reasoning_text(message)
    if reasoning:
        yield ReasoningDelta(text=reasoning)
    raw_content = message.content or getattr(message, "refusal", None)
    if raw_content:
        markup = _ToolCallMarkupFilter()
        content = "".join(markup.feed(raw_content)) + markup.flush()
        if markup.keeps(content):
            yield AssistantMessage(content=[AssistantContent(type="output_text", text=content)])
            yield TextDelta(text=content)
    tool_accum: dict[int, dict[str, str]] = {}
    for tool_call in message.tool_calls or []:
        tool_accum[len(tool_accum)] = {
            "name": tool_call.function.name or "",
            "args": tool_call.function.arguments or "",
            "id": tool_call.id or "",
        }
    yield from _tool_calls_from_accum(tool_accum)


class ChatCompletionsApiModelHandler(BaseOpenAICompatibleHandler):
    """LLM handler that talks to an OpenAI-compatible ``/v1/chat/completions`` server.

    Functionally mirrors :class:`ResponsesApiModelHandler` but uses the mature
    Chat Completions streaming tool-call protocol (``choices[].delta.tool_calls``)
    instead of ``/v1/responses``. This is the robust path for vLLM + Qwen tool
    calling. The conversation is serialised with :meth:`Chat.to_transformers_chat`,
    which already emits OpenAI chat messages including ``tool_calls``/``tool`` roles.
    """

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}")
        start = time.time()
        self.client.with_options(max_retries=WARMUP_MAX_RETRIES).chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ],
            extra_body=self._extra_body,
            timeout=self.request_timeout,
        )
        end = time.time()
        logger.info(f"{self.__class__.__name__}:  warmed up! time: {(end - start):.3f} s")

    def _build_compaction_generate_fn(self) -> CompactGenerateFn:
        """Return a generate fn that calls Chat Completions for compaction."""
        client = self.client
        model_name = self.model_name
        timeout = self.request_timeout
        extra_body = self._extra_body

        def generate(system: str, user: str) -> str:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                extra_body=extra_body,
                timeout=timeout,
            )
            return response.choices[0].message.content or ""

        return generate

    @staticmethod
    def _to_chat_content_part(
        part: dict[str, Any],
        audio_content_type: str = "input_audio",
    ) -> ChatCompletionContentPartParam:
        return _to_chat_content_part(part, audio_content_type=audio_content_type)

    @classmethod
    def _chat_messages(
        cls,
        chat: Chat,
        audio_content_type: str = "input_audio",
    ) -> list[dict[str, Any]]:
        return _chat_messages(chat, audio_content_type=audio_content_type)

    # ── base hooks ──────────────────────────────────────────────────────────--

    def _serialize(self, active_chat: Chat) -> list[dict[str, Any]]:
        return self._chat_messages(active_chat, audio_content_type=self.audio_content_type)

    def _build_optional_kwargs(
        self,
        req_tools: Any,
        req_tool_choice: Any,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        return _build_chat_optional_kwargs(req_tools, req_tool_choice, reasoning_effort)

    def _request(self, api_input: list[dict[str, Any]], optional_kwargs: dict[str, Any], *, client: Any = None) -> Any:
        return _request_chat_completions(
            client=client or self.client,
            model_name=self.model_name,
            messages=api_input,
            stream=self.stream,
            extra_body=self._extra_body,
            timeout=self.request_timeout,
            optional_kwargs=optional_kwargs,
        )

    def _iter_stream_events(self, api_response: Stream[ChatCompletionChunk]) -> Iterator[ProviderEvent]:
        yield from _iter_chat_stream_events(api_response)

    def _iter_response_events(self, api_response: Any) -> Iterator[ProviderEvent]:
        yield from _iter_chat_response_events(api_response)

    @staticmethod
    def _tool_calls_from_accum(tool_accum: dict[int, dict[str, str]]) -> Iterator[ToolCall]:
        yield from _tool_calls_from_accum(tool_accum)

    def on_session_end(self) -> None:
        logger.debug("Chat Completions API language model session state reset")
