import logging
import time

import httpx
from nltk import sent_tokenize
from rich.console import Console
from openai import OpenAI, Stream
from openai.types.responses import Response, ResponseStreamEvent

from baseHandler import BaseHandler
from cancel_scope import CancelScope
from LLM.chat import Chat
from LLM.utils import remove_unspeechable
from api.openai_realtime.runtime_config import RuntimeConfig
from LLM.tool_call.qwen3coder_tool_parser import (
    Qwen3CoderToolParser,
    process_printable_text_qwen_xml,
    strip_qwen_tool_markup_for_chat,
)
from LLM.voice_prompt import build_voice_system_prompt
from pipeline_messages import MessageTag

logger = logging.getLogger(__name__)

console = Console()

WHISPER_LANGUAGE_TO_LLM_LANGUAGE = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
}

def _vllm_normalize_list_part(part: dict) -> dict:
    """
    Normalize input_image part to detail="auto"
    """
    t = part.get("type")
    if t == "input_image":
        part["detail"] = "auto"
    return part


def _vllm_normalize_content(content: list | str) -> list[dict]:
    """Normalize chat rows for strict vLLM Responses API responses validators."""
    if isinstance(content, list):
        return [
            _vllm_normalize_list_part(p) if isinstance(p, dict) else p
            for p in content
        ]
    else:
        return content


class OpenApiModelHandler(BaseHandler):
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
        runtime_config: RuntimeConfig | None = None,
        cancel_scope: CancelScope | None = None,
        disable_thinking=True,
        request_timeout_s=20.0,
    ):
        self.cancel_scope = cancel_scope
        self.model_name = model_name
        self.stream = stream
        self.gen_kwargs = dict(gen_kwargs)
        self.request_timeout_s = float(request_timeout_s)
        self.request_timeout = httpx.Timeout(
            self.request_timeout_s,
            connect=min(10.0, self.request_timeout_s),
        )
        self.chat = Chat(chat_size)
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial promt needs to be specified when setting init_chat_role."
                )
            full_prompt = build_voice_system_prompt(init_chat_prompt)
            self.chat.init_chat({"role": init_chat_role, "content": full_prompt})
        self.user_role = user_role
        self.runtime_config = runtime_config
        self._last_instructions = init_chat_prompt
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.tools = None
        self.tools_choice = None
        self._extra_body = (
            {"chat_template_kwargs": {"enable_thinking": False}}
            if disable_thinking and base_url is not None # Only for other than OpenAI Official Server
            else None
        )
        self.warmup()

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
    def _apply_runtime_config(self, override_tool_choice=None):
        if not self.runtime_config:
            return

        new_instructions = self.runtime_config.session.instructions
        if new_instructions and new_instructions != self._last_instructions:
            self._last_instructions = new_instructions
            full_instructions = build_voice_system_prompt(new_instructions)
            self.chat.init_chat({"type": "message", "role": "system", "content": [{"type": "input_text", "text": full_instructions}]})
            logger.info(f"LLM instructions updated ({len(new_instructions)} chars)")

        self.tools = self.runtime_config.session.tools
        self.tools_choice = override_tool_choice or self.runtime_config.session.tool_choice

    def process(self, prompt):
        if isinstance(prompt, tuple) and len(prompt) == 3 and prompt[0] == MessageTag.ADD_TO_CONTEXT:
            _, role, content = prompt
            self.chat.append({"type": "message", "role": role, "content": _vllm_normalize_content(content)})
            logger.debug("Added to LLM context (role=%s)", role)
            return

        if isinstance(prompt, tuple) and len(prompt) == 2 and prompt[0] == MessageTag.FUNCTION_RESULT:
            _, result_text = prompt
            self.chat.append({
                "type": "message",
                "role": self.user_role,
                "content": [{"type": "input_text", "text": result_text}],
            })
            logger.debug("Added function result to LLM context (%d chars)", len(result_text))
            return

        language_code = None

        if isinstance(prompt, tuple) and len(prompt) == 3 and prompt[0] == MessageTag.GENERATE_RESPONSE:
            _, override_instructions, override_tool_choice = prompt
            self._apply_runtime_config(override_tool_choice)
            if override_instructions:
                self.chat.append({
                    "type": "message",
                    "role": self.user_role,
                    "content": [{"type": "input_text", "text": override_instructions}],
                })
        else:
            self._apply_runtime_config()
            # Regular text from STT pipeline
            logger.debug("call api language model...")
            if isinstance(prompt, tuple):
                prompt, language_code = prompt
                if language_code[-5:] == "-auto":
                    language_code = language_code[:-5]
                    prompt = f"Please reply to my message in {WHISPER_LANGUAGE_TO_LLM_LANGUAGE[language_code]}. " + prompt
            self.chat.append({
                "type": "message",
                "role": self.user_role,
                "content": [{"type": "input_text", "text": prompt}],
            })

        optional_kwargs = {}
        parser = None
        if self.tools is not None:
            optional_kwargs["tools"] = self.tools
            parser = Qwen3CoderToolParser(tools=self.tools)
        if self.tools_choice is not None:
            optional_kwargs["tool_choice"] = self.tools_choice

        request_stream = self.stream and self.tools_choice != "required"
        gen = self.cancel_scope.generation if self.cancel_scope else None
        response: Response | Stream[ResponseStreamEvent] | None = None
        tools: list[dict[str, str]] = []
        clean_text = ""
        input_tokens = 0
        output_tokens = 0
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=self.chat.to_list(),
                stream=request_stream,
                extra_body=self._extra_body,
                timeout=self.request_timeout,
                **optional_kwargs,
            )
            if request_stream:
                cancelled = False
                printable_text = ""
                for event in response:
                    if gen is not None and self.cancel_scope.is_stale(gen):
                        logger.info("LLM generation cancelled (interruption)")
                        cancelled = True
                        break
                    if event.type == "response.output_text.delta":
                        new_text = event.delta
                        clean_text += new_text
                        printable_text += new_text
                        if parser is not None:
                            chunks, tools, printable_text = process_printable_text_qwen_xml(
                                printable_text, tools, parser,
                            )
                            for s in chunks:
                                yield remove_unspeechable(s), language_code, []
                        else:
                            sentences = sent_tokenize(printable_text)
                            if len(sentences) > 1:
                                for s in sentences[:-1]:
                                    yield remove_unspeechable(s), language_code, []
                                printable_text = sentences[-1]
                    elif event.type == "response.output_item.done":
                        if event.item.type == "function_call":
                            tools.append(event.item.model_dump())
                    elif event.type == "response.completed":
                        usage = getattr(event.response, "usage", None)
                        if usage:
                            input_tokens = usage.input_tokens or 0
                            output_tokens = usage.output_tokens or 0
                if not cancelled:
                    assistant_speech = remove_unspeechable(
                        strip_qwen_tool_markup_for_chat(clean_text),
                    )
                    if assistant_speech:
                        self.chat.append({
                            "type": "message",
                            "role": "assistant",
                            "content": assistant_speech,
                        })
                    printable_text = remove_unspeechable(
                        strip_qwen_tool_markup_for_chat(printable_text).strip(),
                    )
                    if printable_text or tools:
                        logger.debug(f"Clean text: {clean_text}")
                        logger.debug(f"Tools: {tools}")
                        yield printable_text, language_code, tools
            else:
                # Non-streaming Response (stream=False or tool_choice forces sync API).
                if gen is not None and self.cancel_scope.is_stale(gen):
                    logger.info("LLM generation cancelled (interruption)")
                else:
                    usage = getattr(response, "usage", None)
                    if usage:
                        input_tokens = usage.input_tokens or 0
                        output_tokens = usage.output_tokens or 0
                    for message in response.output:
                        if message.type == "function_call":
                            tools.append(message.model_dump())
                        elif message.type == "message":
                            for chunk in message.content:
                                if chunk.type == "output_text":
                                    clean_text += remove_unspeechable(chunk.text)
                        else:
                            logger.warning(f"Not supported message type: {message.type}")
                    if parser is not None:
                        chunks, tools, printable_text = process_printable_text_qwen_xml(
                            clean_text, tools, parser,
                        )
                        chunk_parts = [remove_unspeechable(s).strip() for s in chunks]
                        chunk_joined = " ".join(p for p in chunk_parts if p)
                        printable_text = remove_unspeechable(
                            strip_qwen_tool_markup_for_chat(printable_text).strip(),
                        )
                        combined = " ".join(
                            p for p in (chunk_joined, printable_text) if p
                        ).strip()
                        assistant_speech = remove_unspeechable(
                            strip_qwen_tool_markup_for_chat(clean_text),
                        )
                        if assistant_speech:
                            self.chat.append({
                                "type": "message",
                                "role": "assistant",
                                "content": assistant_speech,
                            })
                        logger.debug(f"Clean text: {clean_text}")
                        logger.info(f"Tools: {tools}")
                        if combined or tools:
                            yield combined, language_code, tools
                    else:
                        logger.debug(f"Clean text: {clean_text}")
                        logger.info(f"Tools: {tools}")
                        clean_text = remove_unspeechable(clean_text)
                        if clean_text.strip():
                            self.chat.append({
                                "type": "message",
                                "role": "assistant",
                                "content": clean_text.strip(),
                            })
                        if clean_text.strip() or tools:
                            yield clean_text.strip(), language_code, tools
        except httpx.ReadTimeout:
            logger.warning(
                "OpenAI API read timed out after %.1fs; ending the current response",
                self.request_timeout_s,
            )
            yield ("Wow I'm a bit slow today, could you repeat that?", None, None)
        finally:
            if response is not None and hasattr(response, "close"):
                try:
                    response.close()
                except Exception:
                    pass

        self.chat.strip_images()
        if input_tokens or output_tokens:
            yield (MessageTag.TOKEN_USAGE, input_tokens, output_tokens)
        yield (MessageTag.END_OF_RESPONSE, None, None)

    def on_session_end(self):
        # reset() also clears init_chat_message, so a previous session's
        # instructions cannot persist into the next one.
        self.chat.reset()
        self._last_instructions = None
        self.tools = None
        self.tools_choice = None
        logger.debug("OpenAI API language model session state reset (chat + tool cache)")
