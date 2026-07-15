from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any

from openai import Stream
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseTextDeltaEvent,
)

from speech_to_speech.LLM.base_openai_compatible_language_model import (
    WARMUP_MAX_RETRIES,
    AssistantMessage,
    BaseOpenAICompatibleHandler,
    ProviderEvent,
    TextDelta,
    ToolCall,
    Usage,
)
from speech_to_speech.LLM.chat import Chat
from speech_to_speech.LLM.compaction_prompt import CompactGenerateFn
from speech_to_speech.utils.utils import _generate_id

logger = logging.getLogger(__name__)


class ResponsesApiModelHandler(BaseOpenAICompatibleHandler):
    """LLM handler that talks to an OpenAI ``/v1/responses`` server."""

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}")
        start = time.time()
        self.client.with_options(max_retries=WARMUP_MAX_RETRIES).responses.create(
            model=self.model_name,
            input=[
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "You are a helpful assistant"}],
                },
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
            ],
            timeout=self.request_timeout,
        )
        end = time.time()
        logger.info(f"{self.__class__.__name__}:  warmed up! time: {(end - start):.3f} s")

    def _build_compaction_generate_fn(self) -> CompactGenerateFn:
        """Return a generate fn that calls the Responses API for compaction."""
        client = self.client
        model_name = self.model_name
        timeout = self.request_timeout

        def generate(system: str, user: str) -> str:
            response = client.responses.create(
                model=model_name,
                input=[
                    {
                        "type": "message",
                        "role": "system",
                        "content": [{"type": "input_text", "text": system}],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": user}],
                    },
                ],
                timeout=timeout,
            )
            return response.output_text

        return generate

    # ── base hooks ──────────────────────────────────────────────────────────--

    def _serialize(self, active_chat: Chat) -> Any:
        return active_chat.to_responses_api_chat()

    def _build_optional_kwargs(self, req_tools: Any, req_tool_choice: Any) -> dict[str, Any]:
        optional_kwargs: dict[str, Any] = {}
        if req_tools is not None:
            optional_kwargs["tools"] = req_tools
        if req_tool_choice is not None:
            optional_kwargs["tool_choice"] = req_tool_choice
        return optional_kwargs

    def _request(self, api_input: Any, optional_kwargs: dict[str, Any]) -> Any:
        return self.client.responses.create(
            model=self.model_name,
            input=api_input,
            stream=self.stream,
            extra_body=self._extra_body,
            timeout=self.request_timeout,
            **optional_kwargs,
        )

    @staticmethod
    def _assistant_content(content: Any) -> list[AssistantContent]:
        return [
            AssistantContent(type="output_text", text=c.text if c.type == "output_text" else c.refusal) for c in content
        ]

    def _iter_stream_events(self, api_response: Stream) -> Iterator[ProviderEvent]:
        for raw_event in api_response:
            if isinstance(raw_event, ResponseTextDeltaEvent):
                yield TextDelta(text=raw_event.delta)
            elif isinstance(raw_event, ResponseOutputItemDoneEvent):
                item = raw_event.item
                if isinstance(item, ResponseFunctionToolCall):
                    item.call_id = _generate_id("call")
                    item.id = _generate_id("fc")
                    yield ToolCall(item=item)
                elif isinstance(item, ResponseOutputMessage):
                    yield AssistantMessage(content=self._assistant_content(item.content))
            elif isinstance(raw_event, ResponseCompletedEvent):
                usage = getattr(raw_event.response, "usage", None)
                if usage:
                    yield Usage(input_tokens=usage.input_tokens or 0, output_tokens=usage.output_tokens or 0)

    def _iter_response_events(self, api_response: Any) -> Iterator[ProviderEvent]:
        usage = api_response.usage
        if usage:
            yield Usage(input_tokens=usage.input_tokens or 0, output_tokens=usage.output_tokens or 0)
        for message in api_response.output:
            if isinstance(message, ResponseFunctionToolCall):
                message.call_id = _generate_id("call")
                message.id = _generate_id("fc")
                yield ToolCall(item=message)
            elif isinstance(message, ResponseOutputMessage):
                yield AssistantMessage(content=self._assistant_content(message.content))
                # Text-only keeps every character; the base applies remove_unspeechable
                # for audio. Only output_text parts are spoken (refusals are stored).
                raw = "".join(c.text for c in message.content if c.type == "output_text")
                yield TextDelta(text=raw)
            else:
                logger.warning(f"Not supported message type: {message.type}")

    def on_session_end(self) -> None:
        logger.debug("OpenAI API language model session state reset")
