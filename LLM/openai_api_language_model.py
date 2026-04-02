import logging
import time

from nltk import sent_tokenize
from rich.console import Console
from openai import OpenAI, Stream
from openai.types.responses import Response, ResponseStreamEvent

from baseHandler import BaseHandler
from cancel_scope import CancelScope
from LLM.chat import Chat
from LLM.utils import remove_unspeechable
from api.openai_realtime.runtime_config import RuntimeConfig
from LLM.voice_prompt import VOICE_SYSTEM_PROMPT

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
    ):
        self.cancel_scope = cancel_scope
        self.model_name = model_name
        self.stream = stream
        self.gen_kwargs = dict(gen_kwargs)
        self.chat = Chat(chat_size)
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial promt needs to be specified when setting init_chat_role."
                )
            full_prompt = f"{VOICE_SYSTEM_PROMPT}\n\n{init_chat_prompt}"
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
            ]
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
            full_instructions = f"{VOICE_SYSTEM_PROMPT}\n\n{new_instructions}"
            self.chat.init_chat({"type": "message", "role": "system", "content": [{"type": "input_text", "text": full_instructions}]})
            logger.info(f"LLM instructions updated ({len(new_instructions)} chars)")

        self.tools = self.runtime_config.session.tools
        self.tools_choice = override_tool_choice or self.runtime_config.session.tool_choice

    def process(self, prompt):
        # Context-only: add user/input text to chat without generating.
        # Generation is deferred until __GENERATE_RESPONSE__ (from response.create).
        if isinstance(prompt, tuple) and len(prompt) == 3 and prompt[0] == "__ADD_TO_CONTEXT__":
            _, role, content = prompt
            self.chat.append({"type": "message", "role": role, "content": content})
            logger.debug("Added to LLM context (role=%s)", role)
            return

        # Context-only: add function-call result to chat without generating.
        if isinstance(prompt, tuple) and len(prompt) == 2 and prompt[0] == "__FUNCTION_RESULT__":
            _, result_text = prompt
            self.chat.append({
                "type": "message",
                "role": self.user_role,
                "content": [{"type": "input_text", "text": result_text}],
            })
            logger.debug("Added function result to LLM context (%d chars)", len(result_text))
            return

        language_code = None

        if isinstance(prompt, tuple) and len(prompt) == 3 and prompt[0] == "__GENERATE_RESPONSE__":
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
        if self.tools is not None:
            optional_kwargs["tools"] = self.tools
        if self.tools_choice is not None:
            optional_kwargs["tool_choice"] = self.tools_choice

        gen = self.cancel_scope.generation if self.cancel_scope else None
        response: Response | Stream[ResponseStreamEvent] = self.client.responses.create(
            model=self.model_name,
            input=self.chat.to_list(),
            stream=self.stream,
            extra_body=self._extra_body,
            **optional_kwargs,
        )
        tools: list[dict[str, str]] = []
        clean_text = ""
        input_tokens = 0
        output_tokens = 0
        if self.stream:
            cancelled = False
            printable_text = ""
            for event in response:
                if gen is not None and self.cancel_scope.is_stale(gen):
                    logger.info("LLM generation cancelled (interruption)")
                    cancelled = True
                    break
                if event.type == "response.output_text.delta":
                    new_text = remove_unspeechable(event.delta)
                    clean_text += new_text
                    printable_text += new_text
                    sentences = sent_tokenize(printable_text)
                    if len(sentences) > 1:
                        for s in sentences[:-1]:
                            yield s, language_code, []
                        printable_text = sentences[-1]
                elif event.type == "response.output_item.done":
                    
                    if event.item.type == "function_call":
                        tools.append(event.item.model_dump())
                    elif event.item.type == "message":
                        self.chat.append({
                            "type": "message",
                            "role": event.item.role,
                            "content": event.item.content,
                        })
                elif event.type == "response.completed":
                    usage = getattr(event.response, "usage", None)
                    if usage:
                        input_tokens = usage.input_tokens or 0
                        output_tokens = usage.output_tokens or 0
            if not cancelled:
                if printable_text.strip() or tools:
                    logger.debug(f"Clean text: {clean_text}")
                    logger.info(f"Tools: {tools}")
                    yield printable_text.strip(), language_code, tools
        else:
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
                        self.chat.append({
                            "type": "message",
                            "role": message.role,
                            "content": message.content,
                        })
                        for chunk in message.content:
                            if chunk.type == "output_text":
                                clean_text += remove_unspeechable(chunk.text)
                    else:
                        logger.warning(f"Not supported message type: {message.type}")
                logger.debug(f"Clean text: {clean_text}")
                logger.info(f"Tools: {tools}")
                if clean_text.strip() or tools:
                    yield clean_text.strip(), language_code, tools

        self.chat.strip_images()
        if input_tokens or output_tokens:
            yield ("__TOKEN_USAGE__", input_tokens, output_tokens)
        yield ("__END_OF_RESPONSE__", None, None)

    def on_session_end(self):
        self.chat.reset()
        self._last_instructions = None
        self.tools = None
        self.tools_choice = None
        logger.debug("OpenAI API language model session state reset (chat + tool cache)")
