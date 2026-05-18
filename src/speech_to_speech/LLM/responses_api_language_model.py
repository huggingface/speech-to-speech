from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any, Optional

import httpx
from nltk import sent_tokenize
from openai import OpenAI, Stream
from openai.types.realtime.conversation_item import (
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemFunctionCall,
)
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.LLM.chat import Chat, make_system_message, make_user_message
from speech_to_speech.LLM.compaction_prompt import CompactGenerateFn, build_compactor
from speech_to_speech.LLM.utils import remove_unspeechable, resolve_auto_language
from speech_to_speech.LLM.voice_prompt import build_voice_system_prompt
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import LLMIn, LLMOut
from speech_to_speech.pipeline.messages import (
    EndOfResponse,
    LLMResponseChunk,
    TokenUsage,
)
from speech_to_speech.utils.utils import _generate_id

logger = logging.getLogger(__name__)


class ResponsesApiModelHandler(BaseHandler[LLMIn, LLMOut]):
    """
    Handles the language model part.
    """

    def setup(
        self,
        model_name: str = "gpt-5.4-mini",
        device: str = "cuda",
        gen_kwargs: dict[str, Any] = {},
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        stream: bool = True,
        user_role: str = "user",
        cancel_scope: CancelScope | None = None,
        disable_thinking: bool = True,
        request_timeout_s: float = 20.0,
        stream_batch_sentences: int = 3,
        enable_lang_prompt: bool = False,
        compact_history: bool = False,
        **_kwargs: Any,
    ) -> None:
        self.cancel_scope = cancel_scope
        self.model_name = model_name
        self.stream = stream
        self.stream_batch_sentences = max(1, stream_batch_sentences)
        self.enable_lang_prompt = enable_lang_prompt
        self.gen_kwargs = dict(gen_kwargs)
        self.request_timeout_s = float(request_timeout_s)
        self.request_timeout = httpx.Timeout(
            self.request_timeout_s,
            connect=min(10.0, self.request_timeout_s),
        )

        self.user_role = user_role
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._extra_body = (
            {"chat_template_kwargs": {"enable_thinking": False}}
            if disable_thinking
            and base_url is not None
            and base_url != "https://api.openai.com/v1"  # Only for other than OpenAI Official Server
            else None
        )
        self.compactor = build_compactor(self._build_compaction_generate_fn()) if compact_history else None
        self.warmup()

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}")
        start = time.time()
        self.client.responses.create(
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

    def _apply_config(
        self,
        chat: Chat,
        instructions: Optional[str],
    ) -> None:
        if instructions:
            full_instructions = build_voice_system_prompt(instructions)
            chat.add_item(make_system_message(full_instructions))

    def _generate(
        self,
        active_chat: Chat,
        original_chat: Chat,
        language_code: Optional[str],
        gen: int | None,
        runtime_config: Any,
        response: Any,
        optional_kwargs: dict[str, Any],
    ) -> Iterator[LLMOut]:
        api_response: Response | Stream[ResponseStreamEvent] | None = None
        tools: list[ResponseFunctionToolCall] = []
        clean_text = ""
        input_tokens = 0
        output_tokens = 0
        try:
            api_response = self.client.responses.create(
                model=self.model_name,
                input=active_chat.to_responses_api_chat(),
                stream=self.stream,
                extra_body=self._extra_body,
                timeout=self.request_timeout,
                **optional_kwargs,
            )
            if isinstance(api_response, Stream):
                cancelled = False
                printable_text = ""
                sentence_batch: list[str] = []
                for raw_event in api_response:
                    if gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen):
                        logger.info("LLM generation cancelled (interruption)")
                        cancelled = True
                        break
                    if isinstance(raw_event, ResponseTextDeltaEvent):
                        new_text = remove_unspeechable(raw_event.delta)
                        clean_text += new_text
                        printable_text += new_text
                        sentences = sent_tokenize(printable_text)
                        if len(sentences) > 1:
                            for s in sentences[:-1]:
                                sentence_batch.append(s)
                                if len(sentence_batch) >= self.stream_batch_sentences:
                                    yield LLMResponseChunk(
                                        text=" ".join(sentence_batch),
                                        language_code=language_code,
                                        runtime_config=runtime_config,
                                        response=response,
                                    )
                                    sentence_batch = []
                            printable_text = sentences[-1]
                    elif isinstance(raw_event, ResponseOutputItemDoneEvent):
                        if isinstance(raw_event.item, ResponseFunctionToolCall):
                            raw_event.item.call_id = _generate_id("call")
                            raw_event.item.id = _generate_id("fc")
                            tools.append(raw_event.item)
                            original_chat.add_item(
                                RealtimeConversationItemFunctionCall(
                                    type="function_call",
                                    name=raw_event.item.name,
                                    arguments=raw_event.item.arguments,
                                    call_id=raw_event.item.call_id,
                                    id=raw_event.item.id,
                                    status=raw_event.item.status,
                                )
                            )
                        elif isinstance(raw_event.item, ResponseOutputMessage):
                            content = [
                                AssistantContent(
                                    type="output_text",
                                    text=c.text if c.type == "output_text" else c.refusal,
                                )
                                for c in raw_event.item.content
                            ]
                            original_chat.add_item(
                                RealtimeConversationItemAssistantMessage(
                                    type="message", role="assistant", content=content
                                )
                            )
                    elif isinstance(raw_event, ResponseCompletedEvent):
                        usage = getattr(raw_event.response, "usage", None)
                        if usage:
                            input_tokens = usage.input_tokens or 0
                            output_tokens = usage.output_tokens or 0
                if not cancelled:
                    if printable_text.strip():
                        sentence_batch.append(printable_text.strip())
                    remaining = " ".join(sentence_batch)
                    if remaining or tools:
                        logger.debug(f"Clean text: {clean_text}")
                        logger.info(f"Tools: {tools}")
                        yield LLMResponseChunk(
                            text=remaining,
                            language_code=language_code,
                            tools=tools,
                            runtime_config=runtime_config,
                            response=response,
                        )
            elif isinstance(api_response, Response):
                if gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen):
                    logger.info("LLM generation cancelled (interruption)")
                else:
                    usage = api_response.usage
                    if usage:
                        input_tokens = usage.input_tokens or 0
                        output_tokens = usage.output_tokens or 0
                    for message in api_response.output:
                        if isinstance(message, ResponseFunctionToolCall):
                            item = original_chat.add_item(
                                RealtimeConversationItemFunctionCall(
                                    type="function_call",
                                    name=message.name,
                                    arguments=message.arguments,
                                    status="in_progress",
                                )
                            )
                            assert (hasattr(item, "call_id") and item.call_id is not None) and item.id is not None
                            message.call_id = item.call_id
                            message.id = item.id
                            tools.append(message)
                        elif isinstance(message, ResponseOutputMessage):
                            content = [
                                AssistantContent(
                                    type="output_text",
                                    text=c.text if c.type == "output_text" else c.refusal,
                                )
                                for c in message.content
                            ]
                            original_chat.add_item(
                                RealtimeConversationItemAssistantMessage(
                                    type="message", role="assistant", content=content
                                )
                            )
                            for chunk in message.content:
                                if chunk.type == "output_text":
                                    clean_text += remove_unspeechable(chunk.text)
                        else:
                            logger.warning(f"Not supported message type: {message.type}")
                    logger.debug(f"Clean text: {clean_text}")
                    logger.info(f"Tools: {tools}")
                    if clean_text.strip() or tools:
                        yield LLMResponseChunk(
                            text=clean_text.strip(),
                            language_code=language_code,
                            tools=tools,
                            runtime_config=runtime_config,
                            response=response,
                        )
        except httpx.ReadTimeout:
            logger.warning(
                "OpenAI API read timed out after %.1fs; ending the current response",
                self.request_timeout_s,
            )
            yield LLMResponseChunk(
                text="Wow I'm a bit slow today, could you repeat that?",
                runtime_config=runtime_config,
                response=response,
            )
        finally:
            if api_response is not None and hasattr(api_response, "close"):
                try:
                    api_response.close()
                except Exception:
                    pass

        original_chat.strip_images()
        original_chat.trim_if_needed(self.compactor)
        if input_tokens or output_tokens:
            yield TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
        yield EndOfResponse()

    def process(self, request: LLMIn) -> Iterator[LLMOut]:
        """
        Process a language model request and yield LLMResponseChunks.

        Args:
            request: The LLMIn request containing runtime configuration and response parameters.

        Yields:
            LLMResponseChunk: Chunks of text and tools from the language model response.
        """
        runtime_config = request.runtime_config
        response = request.response
        original_chat = runtime_config.chat
        active_chat = original_chat.copy()
        language_code = request.language_code
        instructions = (
            response.instructions if response and response.instructions else runtime_config.session.instructions
        ) or ""
        req_tools = response.tools if response and response.tools else runtime_config.session.tools
        req_tool_choice = (
            response.tool_choice if response and response.tool_choice else runtime_config.session.tool_choice
        )
        self._apply_config(active_chat, instructions)
        language_code, lang_name = resolve_auto_language(language_code)
        if lang_name and self.enable_lang_prompt:
            active_chat.add_item(make_user_message(f"Please reply to my message in {lang_name}."))

        optional_kwargs: dict[str, Any] = {}
        if req_tools is not None:
            optional_kwargs["tools"] = req_tools
        if req_tool_choice is not None:
            optional_kwargs["tool_choice"] = req_tool_choice

        # CancelScope.is_stale(gen) is checked when the stream iterator advances; a
        # blocked read inside httpx cannot be aborted by cancel_scope.cancel() from
        # the websocket router. Mitigations: request_timeout_s / ReadTimeout. A future
        # option is to run this API call in a child process and terminate() on session
        # end (IPC and lifecycle cost).
        gen = self.cancel_scope.generation if self.cancel_scope else None

        yield from self._generate(
            active_chat, original_chat, language_code, gen, runtime_config, response, optional_kwargs
        )

    def on_session_end(self) -> None:
        logger.debug("OpenAI API language model session state reset")

    @property
    def timing_log_level(self) -> int:
        return logging.INFO

    def should_log_timing(self, output: LLMOut) -> bool:
        return isinstance(output, LLMResponseChunk) and self.last_time > self.min_time_to_debug
