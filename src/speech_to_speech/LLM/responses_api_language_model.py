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
from speech_to_speech.LLM.chat import (
    Chat,
    ChatItemError,
    SupportedItem,
    build_active_chat,
    make_system_message,
    make_user_message,
)
from speech_to_speech.LLM.compaction_prompt import CompactGenerateFn, build_compactor
from speech_to_speech.LLM.text_prompt import build_text_system_prompt
from speech_to_speech.LLM.utils import remove_unspeechable, resolve_auto_language
from speech_to_speech.LLM.voice_prompt import build_voice_system_prompt
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import LLMIn, LLMOut
from speech_to_speech.pipeline.messages import (
    EndOfResponse,
    LLMResponseChunk,
    TokenUsage,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.utils.utils import _generate_id, is_out_of_band, response_wants_audio

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
        speculative_turns: SpeculativeTurnTracker | None = None,
        disable_thinking: bool = True,
        request_timeout_s: float = 20.0,
        stream_batch_sentences: int = 3,
        enable_lang_prompt: bool = False,
        compact_history: bool = False,
        **_kwargs: Any,
    ) -> None:
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
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

    def _turn_is_latest(self, turn_id: str | None, turn_revision: int | None) -> bool:
        return self.speculative_turns is None or self.speculative_turns.is_latest(turn_id, turn_revision)

    def _generation_is_stale(self, gen: int | None) -> bool:
        return gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen)

    def _turn_output_allowed(self, turn_id: str | None, turn_revision: int | None) -> bool:
        if self.speculative_turns is None:
            return True
        return self.speculative_turns.is_latest_after_reopen_grace(turn_id, turn_revision)

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
        wants_audio: bool = True,
    ) -> None:
        if instructions:
            builder = build_voice_system_prompt if wants_audio else build_text_system_prompt
            full_instructions = builder(instructions)
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
        turn_id: str | None,
        turn_revision: int | None,
        speech_stopped_at_s: float | None,
    ) -> Iterator[LLMOut]:
        api_response: Response | Stream[ResponseStreamEvent] | None = None
        tools: list[ResponseFunctionToolCall] = []
        pending_chat_items: list[SupportedItem] = []
        clean_text = ""
        input_tokens = 0
        output_tokens = 0
        # Text-only responses have no TTS to feed, so streaming buys no latency:
        # force a single non-streamed call (the Response branch yields one full-text
        # chunk). Audio responses keep streaming for low-latency synthesis.
        use_stream = self.stream and response_wants_audio(response)
        try:
            api_response = self.client.responses.create(
                model=self.model_name,
                input=active_chat.to_responses_api_chat(),
                stream=use_stream,
                extra_body=self._extra_body,
                timeout=self.request_timeout,
                **optional_kwargs,
            )
            if isinstance(api_response, Stream):
                cancelled = False
                printable_text = ""
                sentence_batch: list[str] = []
                for raw_event in api_response:
                    if (
                        gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen)
                    ) or not self._turn_is_latest(turn_id, turn_revision):
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
                                    if not self._turn_output_allowed(turn_id, turn_revision):
                                        logger.info("LLM generation cancelled (stale speculative turn)")
                                        cancelled = True
                                        break
                                    yield LLMResponseChunk(
                                        text=" ".join(sentence_batch),
                                        language_code=language_code,
                                        runtime_config=runtime_config,
                                        response=response,
                                        turn_id=turn_id,
                                        turn_revision=turn_revision,
                                        speech_stopped_at_s=speech_stopped_at_s,
                                        cancel_generation=gen,
                                    )
                                    sentence_batch = []
                            if cancelled:
                                break
                            printable_text = sentences[-1]
                    elif isinstance(raw_event, ResponseOutputItemDoneEvent):
                        if isinstance(raw_event.item, ResponseFunctionToolCall):
                            if printable_text.strip():
                                sentence_batch.append(printable_text.strip())
                                printable_text = ""
                            if sentence_batch:
                                if not self._turn_output_allowed(turn_id, turn_revision):
                                    logger.info("LLM generation cancelled (stale speculative turn)")
                                    cancelled = True
                                    break
                                yield LLMResponseChunk(
                                    text=" ".join(sentence_batch),
                                    language_code=language_code,
                                    runtime_config=runtime_config,
                                    response=response,
                                    turn_id=turn_id,
                                    turn_revision=turn_revision,
                                    speech_stopped_at_s=speech_stopped_at_s,
                                    cancel_generation=gen,
                                )
                                sentence_batch = []
                            raw_event.item.call_id = _generate_id("call")
                            raw_event.item.id = _generate_id("fc")
                            tools.append(raw_event.item)
                            pending_chat_items.append(
                                RealtimeConversationItemFunctionCall(
                                    type="function_call",
                                    name=raw_event.item.name,
                                    arguments=raw_event.item.arguments,
                                    call_id=raw_event.item.call_id,
                                    id=raw_event.item.id,
                                    status=raw_event.item.status,
                                )
                            )
                            if not self._turn_output_allowed(turn_id, turn_revision):
                                logger.info("LLM generation cancelled (stale speculative turn)")
                                cancelled = True
                                break
                            yield LLMResponseChunk(
                                text="",
                                language_code=language_code,
                                tools=[raw_event.item],
                                runtime_config=runtime_config,
                                response=response,
                                turn_id=turn_id,
                                turn_revision=turn_revision,
                                speech_stopped_at_s=speech_stopped_at_s,
                                cancel_generation=gen,
                            )
                        elif isinstance(raw_event.item, ResponseOutputMessage):
                            content = [
                                AssistantContent(
                                    type="output_text",
                                    text=c.text if c.type == "output_text" else c.refusal,
                                )
                                for c in raw_event.item.content
                            ]
                            pending_chat_items.append(
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
                    if remaining:
                        if self._generation_is_stale(gen):
                            logger.info("LLM generation cancelled (interruption)")
                        elif not self._turn_output_allowed(turn_id, turn_revision):
                            logger.info("LLM generation cancelled (stale speculative turn)")
                        else:
                            logger.debug(f"Clean text: {clean_text}")
                            logger.info(f"Tools: {tools}")
                            yield LLMResponseChunk(
                                text=remaining,
                                language_code=language_code,
                                runtime_config=runtime_config,
                                response=response,
                                turn_id=turn_id,
                                turn_revision=turn_revision,
                                speech_stopped_at_s=speech_stopped_at_s,
                                cancel_generation=gen,
                            )
            elif isinstance(api_response, Response):
                if (
                    gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen)
                ) or not self._turn_is_latest(turn_id, turn_revision):
                    logger.info("LLM generation cancelled (interruption)")
                else:
                    usage = api_response.usage
                    if usage:
                        input_tokens = usage.input_tokens or 0
                        output_tokens = usage.output_tokens or 0
                    for message in api_response.output:
                        if isinstance(message, ResponseFunctionToolCall):
                            message.call_id = _generate_id("call")
                            message.id = _generate_id("fc")
                            pending_chat_items.append(
                                RealtimeConversationItemFunctionCall(
                                    type="function_call",
                                    name=message.name,
                                    arguments=message.arguments,
                                    call_id=message.call_id,
                                    id=message.id,
                                    status="in_progress",
                                )
                            )
                            tools.append(message)
                            if self._generation_is_stale(gen):
                                logger.info("LLM generation cancelled (interruption)")
                            elif not self._turn_output_allowed(turn_id, turn_revision):
                                logger.info("LLM generation cancelled (stale speculative turn)")
                            else:
                                yield LLMResponseChunk(
                                    text="",
                                    language_code=language_code,
                                    tools=[message],
                                    runtime_config=runtime_config,
                                    response=response,
                                    turn_id=turn_id,
                                    turn_revision=turn_revision,
                                    speech_stopped_at_s=speech_stopped_at_s,
                                    cancel_generation=gen,
                                )
                        elif isinstance(message, ResponseOutputMessage):
                            content = [
                                AssistantContent(
                                    type="output_text",
                                    text=c.text if c.type == "output_text" else c.refusal,
                                )
                                for c in message.content
                            ]
                            pending_chat_items.append(
                                RealtimeConversationItemAssistantMessage(
                                    type="message", role="assistant", content=content
                                )
                            )
                            message_text = ""
                            for chunk in message.content:
                                if chunk.type == "output_text":
                                    message_text += remove_unspeechable(chunk.text)
                            clean_text += message_text
                            if message_text.strip():
                                if self._generation_is_stale(gen):
                                    logger.info("LLM generation cancelled (interruption)")
                                elif not self._turn_output_allowed(turn_id, turn_revision):
                                    logger.info("LLM generation cancelled (stale speculative turn)")
                                else:
                                    yield LLMResponseChunk(
                                        text=message_text.strip(),
                                        language_code=language_code,
                                        runtime_config=runtime_config,
                                        response=response,
                                        turn_id=turn_id,
                                        turn_revision=turn_revision,
                                        speech_stopped_at_s=speech_stopped_at_s,
                                        cancel_generation=gen,
                                    )
                        else:
                            logger.warning(f"Not supported message type: {message.type}")
                    logger.debug(f"Clean text: {clean_text}")
                    logger.info(f"Tools: {tools}")
        except httpx.ReadTimeout:
            logger.warning(
                "OpenAI API read timed out after %.1fs; ending the current response",
                self.request_timeout_s,
            )
            if not self._generation_is_stale(gen) and self._turn_output_allowed(turn_id, turn_revision):
                yield LLMResponseChunk(
                    text="Wow I'm a bit slow today, could you repeat that?",
                    runtime_config=runtime_config,
                    response=response,
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                    speech_stopped_at_s=speech_stopped_at_s,
                    cancel_generation=gen,
                )
        finally:
            if api_response is not None and hasattr(api_response, "close"):
                try:
                    api_response.close()
                except Exception:
                    pass

        if not self._generation_is_stale(gen) and self._turn_output_allowed(turn_id, turn_revision):
            # Out-of-band responses emit output and usage but never write back to the
            # default conversation (their context was a throwaway chat).
            if not is_out_of_band(response):
                for item in pending_chat_items:
                    original_chat.add_item(item)
                original_chat.strip_images()
                original_chat.trim_if_needed(self.compactor)
            if input_tokens or output_tokens:
                yield TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                )
        yield EndOfResponse(turn_id=turn_id, turn_revision=turn_revision, cancel_generation=gen)

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
        turn_id = request.turn_id
        turn_revision = request.turn_revision
        speech_stopped_at_s = request.speech_stopped_at_s
        if not self._turn_is_latest(turn_id, turn_revision):
            logger.info("Skipping stale LLM request for turn=%s rev=%s", turn_id, turn_revision)
            yield EndOfResponse(turn_id=turn_id, turn_revision=turn_revision)
            return

        original_chat = runtime_config.chat
        out_of_band = is_out_of_band(response)
        if out_of_band:
            try:
                active_chat = build_active_chat(original_chat, response)
            except ChatItemError as exc:
                logger.info("Out-of-band response rejected: %s", exc)
                yield EndOfResponse(turn_id=turn_id, turn_revision=turn_revision, error=str(exc))
                return
        else:
            active_chat = original_chat.copy()
        language_code = request.language_code
        instructions = (
            response.instructions if response and response.instructions else runtime_config.session.instructions
        ) or ""
        req_tools = response.tools if response and response.tools else runtime_config.session.tools
        req_tool_choice = (
            response.tool_choice if response and response.tool_choice else runtime_config.session.tool_choice
        )
        self._apply_config(active_chat, instructions, response_wants_audio(response))
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
            active_chat,
            original_chat,
            language_code,
            gen,
            runtime_config,
            response,
            optional_kwargs,
            turn_id,
            turn_revision,
            speech_stopped_at_s,
        )

    def on_session_end(self) -> None:
        logger.debug("OpenAI API language model session state reset")

    @property
    def timing_log_level(self) -> int:
        return logging.INFO

    def should_log_timing(self, output: LLMOut) -> bool:
        return isinstance(output, LLMResponseChunk) and self.last_time > self.min_time_to_debug
