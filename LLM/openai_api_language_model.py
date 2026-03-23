import logging
import time
import re
from threading import Event

from nltk import sent_tokenize
from rich.console import Console
from openai import OpenAI, Stream
from openai.types.responses import Response, ResponseStreamEvent 

from baseHandler import BaseHandler
from LLM.chat import Chat

from api.openai_realtime.runtime_config import RuntimeConfig

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
        cancel_response: Event | None = None,
    ):
        self.cancel_response = cancel_response
        self.model_name = model_name
        self.stream = stream
        self.chat = Chat(chat_size)
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial promt needs to be specified when setting init_chat_role."
                )
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
        self.user_role = user_role
        self.runtime_config = runtime_config
        self._last_instructions = init_chat_prompt
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.tools = None
        self.tools_choice = None
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        start = time.time()
        self.client.responses.create(
            model=self.model_name,
            input=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ],
            stream=self.stream
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
            self.chat.init_chat({"role": "system", "content": new_instructions})
            logger.info(f"LLM instructions updated ({len(new_instructions)} chars)")

        self.tools = self.runtime_config.session.tools
        self.tools_choice = override_tool_choice or self.runtime_config.session.tool_choice

    def process(self, prompt):
            # Context-only: add user/input text to chat without generating.
            # Generation is deferred until __GENERATE_RESPONSE__ (from response.create).
            if isinstance(prompt, tuple) and len(prompt) == 3 and prompt[0] == "__ADD_TO_CONTEXT__":
                _, role, text = prompt
                self.chat.append({"role": role, "content": text})
                logger.debug("Added to LLM context (role=%s, %d chars)", role, len(text))
                return

            # Context-only: add function-call result to chat without generating.
            if isinstance(prompt, tuple) and len(prompt) == 2 and prompt[0] == "__FUNCTION_RESULT__":
                _, result_text = prompt
                self.chat.append({"role": self.user_role, "content": result_text})
                logger.debug("Added function result to LLM context (%d chars)", len(result_text))
                return

            language_code = None

            if isinstance(prompt, tuple) and len(prompt) == 3 and prompt[0] == "__GENERATE_RESPONSE__":
                _, override_instructions, override_tool_choice = prompt
                self._apply_runtime_config(override_tool_choice)
                if override_instructions:
                    self.chat.append({"role": self.user_role, "content": override_instructions})
            else:
                self._apply_runtime_config()
                # Regular text from STT pipeline
                logger.debug("call api language model...")
                if isinstance(prompt, tuple):
                    prompt, language_code = prompt
                    if language_code[-5:] == "-auto":
                        language_code = language_code[:-5]
                        prompt = f"Please reply to my message in {WHISPER_LANGUAGE_TO_LLM_LANGUAGE[language_code]}. " + prompt
                self.chat.append({"role": self.user_role, "content": prompt})

            response: Response | Stream[ResponseStreamEvent] = self.client.responses.create(
                model=self.model_name,
                input=self.chat.to_list(),
                stream=self.stream,
                tools=self.tools,
                tool_choice=self.tools_choice
            )
            tools: list[dict[str, str]] = []
            clean_text: str = ""
            if self.stream:
                printable_text = ""
                cancelled = False
                for event in response:
                    if self.cancel_response and self.cancel_response.is_set():
                        logger.info("LLM generation cancelled (interruption)")
                        cancelled = True
                        break
                    if event.type == "response.output_text.delta":
                        printable_text += event.delta
                        sentences = sent_tokenize(printable_text)
                        if len(sentences) > 1:
                            yield sentences[0], language_code, []
                            printable_text = sentences[-1]
                    elif event.type == "response.output_item.done":
                        if event.item.type == "function_call":
                            tools.append(event.item.model_dump())
                        elif event.item.type == "message":
                            self.chat.append({"role": event.item.role, "content": event.item.content})
                if not cancelled:
                    if printable_text.strip() or tools:
                        logger.info(f"Clean text: {printable_text}")
                        logger.info(f"Tools: {tools}")
                        yield printable_text, language_code, tools
            else:
                if self.cancel_response and self.cancel_response.is_set():
                    logger.info("LLM generation cancelled (interruption)")
                else:
                    for message in response.output:
                        if message.type == "function_call":
                            tools.append(message.model_dump())
                        elif message.type == "message":
                            self.chat.append({"role": message.role, "content": message.content})
                            for chunk in message.content:
                                if chunk.type == "output_text":
                                    clean_text += chunk.text
                        else:
                            logger.warning(f"Not supported message type: {message.type}")
                    logger.info(f"Clean text: {clean_text}")
                    logger.info(f"Tools: {tools}")
                    yield clean_text, language_code, tools

            yield ("__END_OF_RESPONSE__", None, None)

