from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from typing import Any, Optional

import httpx
from nltk import sent_tokenize
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk
from openai.types.realtime.conversation_item import (
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemFunctionCall,
)
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)
from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.LLM.chat import Chat, SupportedItem, make_system_message, make_user_message
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
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.utils.utils import _generate_id

logger = logging.getLogger(__name__)


def _to_chat_tools(req_tools: Any) -> list[dict[str, Any]] | None:
    """Convert Responses-API function tools to Chat-Completions tool format.

    Responses tools are flat ``{type:"function", name, description, parameters}``;
    Chat Completions nests them under a ``function`` key. Items already in the
    nested form (or non-function tools) are passed through untouched.
    """
    if not req_tools:
        return None
    chat_tools: list[dict[str, Any]] = []
    for t in req_tools:
        d = t if isinstance(t, dict) else t.model_dump(exclude_none=True)
        if d.get("type") == "function" and "function" not in d:
            fn = {k: d[k] for k in ("name", "description", "parameters") if k in d}
            chat_tools.append({"type": "function", "function": fn})
        else:
            chat_tools.append(d)
    return chat_tools


def _to_chat_tool_choice(tool_choice: Any) -> Any:
    """Convert a Responses-API tool_choice to Chat-Completions form.

    The string forms ("auto"/"required"/"none") are identical across both APIs;
    only the forced-function object differs (flat ``name`` vs nested ``function``).
    """
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function" and "name" in tool_choice:
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    if tool_choice is not None and not isinstance(tool_choice, (str, dict)):
        d = tool_choice.model_dump(exclude_none=True)
        if d.get("type") == "function" and "name" in d:
            return {"type": "function", "function": {"name": d["name"]}}
        return d
    return tool_choice


class ChatCompletionsApiModelHandler(BaseHandler[LLMIn, LLMOut]):
    """LLM handler that talks to an OpenAI-compatible ``/v1/chat/completions`` server.

    Functionally mirrors :class:`ResponsesApiModelHandler` but uses the mature
    Chat Completions streaming tool-call protocol (``choices[].delta.tool_calls``)
    instead of ``/v1/responses``. This is the robust path for vLLM + Qwen tool
    calling. The conversation is serialised with :meth:`Chat.to_transformers_chat`,
    which already emits OpenAI chat messages including ``tool_calls``/``tool`` roles.
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
        reasoning_effort: Optional[str] = None,
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
        self._extra_body = self._build_extra_body(base_url, disable_thinking, reasoning_effort)
        self.compactor = build_compactor(self._build_compaction_generate_fn()) if compact_history else None
        self.warmup()

    @staticmethod
    def _build_extra_body(
        base_url: Optional[str],
        disable_thinking: bool,
        reasoning_effort: Optional[str],
    ) -> Optional[dict[str, Any]]:
        """Build the provider-specific ``extra_body`` used to disable reasoning.

        Providers differ in how reasoning is turned off: vLLM/Qwen honour
        ``chat_template_kwargs.enable_thinking=false``, while others (e.g. GLM via
        the HF router) ignore that and require ``reasoning_effort='none'``. An
        explicit ``reasoning_effort`` therefore takes precedence; otherwise we fall
        back to the chat-template flag. None of this applies to the official
        OpenAI server, which rejects unknown extra_body keys.
        """
        if base_url is None or base_url == "https://api.openai.com/v1":
            return None
        if reasoning_effort is not None:
            return {"reasoning_effort": reasoning_effort}
        if disable_thinking:
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return None

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
        self.client.chat.completions.create(
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

    def _apply_config(
        self,
        chat: Chat,
        instructions: Optional[str],
    ) -> None:
        if instructions:
            full_instructions = build_voice_system_prompt(instructions)
            chat.add_item(make_system_message(full_instructions))

    @staticmethod
    def _chat_messages(chat: Chat) -> list[dict[str, Any]]:
        """Serialise the chat for the Chat Completions API.

        ``Chat.to_transformers_chat`` targets HuggingFace ``apply_chat_template``,
        which expects tool-call ``arguments`` as a parsed object. The OpenAI Chat
        Completions HTTP API instead requires ``arguments`` to be a JSON *string*,
        so re-encode any object back to a string here.
        """
        messages = chat.to_transformers_chat()
        for message in messages:
            for tool_call in message.get("tool_calls") or []:
                fn = tool_call.get("function")
                if fn is not None and not isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.dumps(fn.get("arguments") or {}, ensure_ascii=False)
        return messages

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
        api_response: Stream[ChatCompletionChunk] | Any = None
        tools: list[ResponseFunctionToolCall] = []
        pending_chat_items: list[SupportedItem] = []
        clean_text = ""
        input_tokens = 0
        output_tokens = 0

        # Accumulate streamed tool-call deltas, keyed by their stream index.
        tool_accum: dict[int, dict[str, str]] = {}

        def _flush_batch(sentence_batch: list[str]) -> Iterator[LLMOut]:
            if not sentence_batch:
                return
            if not self._turn_output_allowed(turn_id, turn_revision):
                return
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

        try:
            create_kwargs: dict[str, Any] = dict(optional_kwargs)
            if self.stream:
                create_kwargs["stream_options"] = {"include_usage": True}
            api_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self._chat_messages(active_chat),
                stream=self.stream,
                extra_body=self._extra_body,
                timeout=self.request_timeout,
                **create_kwargs,
            )

            if isinstance(api_response, Stream):
                cancelled = False
                printable_text = ""
                sentence_batch: list[str] = []
                for chunk in api_response:
                    if (
                        gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen)
                    ) or not self._turn_is_latest(turn_id, turn_revision):
                        logger.info("LLM generation cancelled (interruption)")
                        cancelled = True
                        break

                    # Usage-only trailing chunk (choices == []) when include_usage is set.
                    if chunk.usage is not None:
                        input_tokens = chunk.usage.prompt_tokens or 0
                        output_tokens = chunk.usage.completion_tokens or 0
                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta

                    if delta.content:
                        new_text = remove_unspeechable(delta.content)
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
                                    yield from _flush_batch(sentence_batch)
                                    sentence_batch = []
                            if cancelled:
                                break
                            printable_text = sentences[-1]

                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            entry = tool_accum.setdefault(tc.index, {"name": "", "args": "", "id": ""})
                            if tc.id:
                                entry["id"] = tc.id
                            if tc.function is not None:
                                if tc.function.name:
                                    entry["name"] = tc.function.name
                                if tc.function.arguments:
                                    entry["args"] += tc.function.arguments

                if not cancelled:
                    # Flush any trailing text before emitting tool calls.
                    if printable_text.strip():
                        sentence_batch.append(printable_text.strip())
                    if sentence_batch:
                        if self._generation_is_stale(gen):
                            logger.info("LLM generation cancelled (interruption)")
                        else:
                            logger.debug(f"Clean text: {clean_text}")
                            yield from _flush_batch(sentence_batch)
                    if clean_text.strip():
                        pending_chat_items.append(
                            RealtimeConversationItemAssistantMessage(
                                type="message",
                                role="assistant",
                                content=[AssistantContent(type="output_text", text=clean_text)],
                            )
                        )
                    yield from self._emit_tool_calls(
                        tool_accum,
                        tools,
                        pending_chat_items,
                        language_code,
                        gen,
                        runtime_config,
                        response,
                        turn_id,
                        turn_revision,
                        speech_stopped_at_s,
                    )
                    if tools:
                        logger.info(f"Tools: {tools}")
            else:
                if (
                    gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen)
                ) or not self._turn_is_latest(turn_id, turn_revision):
                    logger.info("LLM generation cancelled (interruption)")
                else:
                    usage = api_response.usage
                    if usage:
                        input_tokens = usage.prompt_tokens or 0
                        output_tokens = usage.completion_tokens or 0
                    message = api_response.choices[0].message
                    message_text = remove_unspeechable(message.content or "")
                    clean_text += message_text
                    if message_text.strip():
                        pending_chat_items.append(
                            RealtimeConversationItemAssistantMessage(
                                type="message",
                                role="assistant",
                                content=[AssistantContent(type="output_text", text=message.content or "")],
                            )
                        )
                        if not self._generation_is_stale(gen) and self._turn_output_allowed(turn_id, turn_revision):
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
                    for tc in message.tool_calls or []:
                        tool_accum[len(tool_accum)] = {
                            "name": tc.function.name or "",
                            "args": tc.function.arguments or "",
                            "id": tc.id or "",
                        }
                    yield from self._emit_tool_calls(
                        tool_accum,
                        tools,
                        pending_chat_items,
                        language_code,
                        gen,
                        runtime_config,
                        response,
                        turn_id,
                        turn_revision,
                        speech_stopped_at_s,
                    )
                    logger.debug(f"Clean text: {clean_text}")
                    if tools:
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

    def _emit_tool_calls(
        self,
        tool_accum: dict[int, dict[str, str]],
        tools: list[ResponseFunctionToolCall],
        pending_chat_items: list[SupportedItem],
        language_code: Optional[str],
        gen: int | None,
        runtime_config: Any,
        response: Any,
        turn_id: str | None,
        turn_revision: int | None,
        speech_stopped_at_s: float | None,
    ) -> Iterator[LLMOut]:
        """Turn accumulated tool-call deltas into ResponseFunctionToolCall events.

        IDs are regenerated (mirroring the Responses handler) so the rest of the
        pipeline pairs each call_id with its function_call_output consistently.
        """
        for index in sorted(tool_accum):
            entry = tool_accum[index]
            if not entry["name"]:
                continue
            call_id = _generate_id("call")
            fc_id = _generate_id("fc")
            item = ResponseFunctionToolCall(
                type="function_call",
                name=entry["name"],
                arguments=entry["args"] or "{}",
                call_id=call_id,
                id=fc_id,
                status="completed",
            )
            tools.append(item)
            pending_chat_items.append(
                RealtimeConversationItemFunctionCall(
                    type="function_call",
                    name=item.name,
                    arguments=item.arguments,
                    call_id=item.call_id,
                    id=item.id,
                    status=item.status,
                )
            )
            if self._generation_is_stale(gen) or not self._turn_output_allowed(turn_id, turn_revision):
                logger.info("LLM generation cancelled (stale speculative turn)")
                continue
            yield LLMResponseChunk(
                text="",
                language_code=language_code,
                tools=[item],
                runtime_config=runtime_config,
                response=response,
                turn_id=turn_id,
                turn_revision=turn_revision,
                speech_stopped_at_s=speech_stopped_at_s,
                cancel_generation=gen,
            )

    def process(self, request: LLMIn) -> Iterator[LLMOut]:
        """Process a language model request and yield LLMResponseChunks."""
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
        chat_tools = _to_chat_tools(req_tools)
        if chat_tools is not None:
            optional_kwargs["tools"] = chat_tools
            if req_tool_choice is not None:
                optional_kwargs["tool_choice"] = _to_chat_tool_choice(req_tool_choice)

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
        logger.debug("Chat Completions API language model session state reset")

    @property
    def timing_log_level(self) -> int:
        return logging.INFO

    def should_log_timing(self, output: LLMOut) -> bool:
        return isinstance(output, LLMResponseChunk) and self.last_time > self.min_time_to_debug
