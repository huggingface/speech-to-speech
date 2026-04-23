from __future__ import annotations

import logging
import time

import httpx
from nltk import sent_tokenize
from typing import Any, get_args
from openai import OpenAI, Stream
from openai.types.responses import Response, ResponseStreamEvent

from baseHandler import BaseHandler
from cancel_scope import CancelScope
from LLM.chat import Chat
from LLM.utils import remove_unspeechable, resolve_auto_language
from LLM.voice_prompt import build_voice_system_prompt
from pipeline_messages import (
    EndOfResponse,
    GenerateResponseRequest,
    LLMResponseChunk,
    TokenUsage,
    Transcription,
)

logger = logging.getLogger(__name__)


class OpenApiModelHandler(BaseHandler[Transcription | GenerateResponseRequest]):
    """
    Handles the language model part.
    """

    def setup(
        self,
        model_name="deepseek-chat",
        device="cuda",
        gen_kwargs={},
        base_url=None,
        api_key=None,
        stream=False,
        user_role="user",
        chat_size=1,
        init_chat_role="system",
        init_chat_prompt="You are a helpful AI assistant.",
        cancel_scope: CancelScope | None = None,
        disable_thinking=True,
        request_timeout_s=20.0,
        stream_batch_sentences=3,
    ):
        self.cancel_scope = cancel_scope
        self.model_name = model_name
        self.stream = stream
        self.stream_batch_sentences = max(1, stream_batch_sentences)
        self.gen_kwargs = dict(gen_kwargs)
        self.request_timeout_s = float(request_timeout_s)
        self.request_timeout = httpx.Timeout(
            self.request_timeout_s,
            connect=min(10.0, self.request_timeout_s),
        )

        # TODO: chat is not used in the realtime pipeline, but still need to be kept for backward compatibility. Remove it in the future.
        self.chat = Chat(chat_size)
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial promt needs to be specified when setting init_chat_role."
                )
            full_prompt = build_voice_system_prompt(init_chat_prompt)
            self.chat.init_chat({"role": init_chat_role, "content": full_prompt})
        
        self.user_role = user_role
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._extra_body = (
            {"chat_template_kwargs": {"enable_thinking": False}}
            if disable_thinking and base_url is not None # Only for other than OpenAI Official Server
            else None
        )
        self.warmup()

    def _prepare_chat_messages(self, chat: Chat) -> list[dict]:
        """Convert chat messages to OpenAI Responses API input format.

        Regular messages are wrapped with ``type: "message"``.
        Tool-related items (``function_call``, ``function_call_output``)
        are passed through as top-level input items per the Responses API spec.
        """
        result = []
        for msg in chat.to_list():
            if msg.get("type") in ("function_call", "function_call_output"):
                result.append(msg)
            else:
                result.append({"type": "message", "role": msg["role"], "content": msg["content"]})
        return result

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        start = time.time()
        self.client.responses.create(
            model=self.model_name,
            input=[
                {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "You are a helpful assistant"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
            ],
            timeout=self.request_timeout,
        )
        end = time.time()
        logger.info(
            f"{self.__class__.__name__}:  warmed up! time: {(end - start):.3f} s"
        )
    def _apply_config(
        self,
        chat: Chat,
        instructions: str | None,
    ) -> None:
        if instructions:
            full_instructions = build_voice_system_prompt(instructions)
            chat.init_chat({"role": "system", "content": [{"type": "input_text", "text": full_instructions}]})

    def process(self, request: Transcription | GenerateResponseRequest):
        language_code = None
        runtime_config = None
        response = None
        req_tools = None
        req_tool_choice = None

        if isinstance(request, GenerateResponseRequest):
            req = request
            runtime_config = req.runtime_config
            response = req.response
            original_chat = runtime_config.chat
            active_chat = original_chat.copy()
            language_code = req.language_code
            instructions = response.instructions if response and response.instructions else runtime_config.session.instructions
            req_tools = response.tools if response and response.tools else runtime_config.session.tools
            req_tool_choice = response.tool_choice if response and response.tool_choice else runtime_config.session.tool_choice
            self._apply_config(active_chat, instructions)
            language_code, lang_name = resolve_auto_language(language_code)
            if lang_name:
                active_chat.append({"role": self.user_role, "content": [{"type": "input_text", "text": f"Please reply to my message in {lang_name}."}]})
        elif isinstance(request, Transcription):
            original_chat = self.chat
            active_chat = original_chat
            logger.debug("call api language model...")
            language_code = request.language_code
            prompt_text = request.text
            language_code, lang_name = resolve_auto_language(language_code)
            if lang_name:
                prompt_text = f"Please reply to my message in {lang_name}. " + prompt_text
            active_chat.append({"role": self.user_role, "content": [{"type": "input_text", "text": prompt_text}]})
        else:
            raise TypeError(f"Unexpected request type: {type(request)}")

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
        api_response: Response | Stream[ResponseStreamEvent] | None = None
        tools: list[dict[str, str]] = []
        clean_text = ""
        input_tokens = 0
        output_tokens = 0
        try:
            api_response = self.client.responses.create(
                model=self.model_name,
                input=self._prepare_chat_messages(active_chat),  # type: ignore[arg-type]
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
                    if not isinstance(raw_event, get_args(ResponseStreamEvent)):
                        continue
                    if gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen):
                        logger.info("LLM generation cancelled (interruption)")
                        cancelled = True
                        break
                    if raw_event.type == "response.output_text.delta":
                        new_text = remove_unspeechable(raw_event.delta)
                        clean_text += new_text
                        printable_text += new_text
                        sentences = sent_tokenize(printable_text)
                        if len(sentences) > 1:
                            for s in sentences[:-1]:
                                sentence_batch.append(s)
                                if len(sentence_batch) >= self.stream_batch_sentences:
                                    yield LLMResponseChunk(text=" ".join(sentence_batch), language_code=language_code, runtime_config=runtime_config, response=response)
                                    sentence_batch = []
                            printable_text = sentences[-1]
                    elif raw_event.type == "response.output_item.done":
                        if raw_event.item.type == "function_call":
                            tools.append(raw_event.item.model_dump())
                        elif raw_event.item.type == "message":
                            original_chat.append({
                                "role": raw_event.item.role,
                                "content": raw_event.item.content,
                            })
                    elif raw_event.type == "response.completed":
                        usage = getattr(raw_event.response, "usage", None)
                        if usage:
                            input_tokens = usage.input_tokens or 0
                            output_tokens = usage.output_tokens or 0
                if not cancelled:
                    for tool in tools:
                        original_chat.append(tool)
                    if printable_text.strip():
                        sentence_batch.append(printable_text.strip())
                    remaining = " ".join(sentence_batch)
                    if remaining or tools:
                        logger.debug(f"Clean text: {clean_text}")
                        logger.info(f"Tools: {tools}")
                        yield LLMResponseChunk(text=remaining, language_code=language_code, tools=tools, runtime_config=runtime_config, response=response)
            elif isinstance(api_response, Response):
                if gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen):
                    logger.info("LLM generation cancelled (interruption)")
                else:
                    usage = api_response.usage
                    if usage:
                        input_tokens = usage.input_tokens or 0
                        output_tokens = usage.output_tokens or 0
                    for message in api_response.output:
                        if message.type == "function_call":
                            tools.append(message.model_dump())
                        elif message.type == "message":
                            original_chat.append({
                                "role": message.role,
                                "content": message.content,
                            })
                            for chunk in message.content:
                                if chunk.type == "output_text":
                                    clean_text += remove_unspeechable(chunk.text)
                        else:
                            logger.warning(f"Not supported message type: {message.type}")
                    for tool in tools:
                        original_chat.append(tool)
                    logger.debug(f"Clean text: {clean_text}")
                    logger.info(f"Tools: {tools}")
                    if clean_text.strip() or tools:
                        yield LLMResponseChunk(text=clean_text.strip(), language_code=language_code, tools=tools, runtime_config=runtime_config, response=response)
        except httpx.ReadTimeout:
            logger.warning(
                "OpenAI API read timed out after %.1fs; ending the current response",
                self.request_timeout_s,
            )
            yield LLMResponseChunk(text="Wow I'm a bit slow today, could you repeat that?", runtime_config=runtime_config, response=response)
        finally:
            if api_response is not None and hasattr(api_response, "close"):
                try:
                    api_response.close()
                except Exception:
                    pass

        original_chat.strip_images()
        if input_tokens or output_tokens:
            yield TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
        yield EndOfResponse()

    def on_session_end(self):
        logger.debug("OpenAI API language model session state reset")
