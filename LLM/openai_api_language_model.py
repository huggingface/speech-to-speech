import logging
import time
import re

from nltk import sent_tokenize
from rich.console import Console
from openai import OpenAI

from baseHandler import BaseHandler
from LLM.chat import Chat

logger = logging.getLogger(__name__)

console = Console()

# Tool call pattern for extraction
TOOL_PATTERN = re.compile(r'\[TOOL:(\w+)(?:\|([^\]]+))?\]')

WHISPER_LANGUAGE_TO_LLM_LANGUAGE = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
}


def parse_tool_calls(text):
    """Extract tool calls from text and return cleaned text + tool calls."""
    tools = []

    for match in TOOL_PATTERN.finditer(text):
        tool_name = match.group(1)
        params_str = match.group(2) or ""

        # Parse params: key1:val1|key2:val2
        params = {}
        if params_str:
            for param in params_str.split('|'):
                if ':' in param:
                    key, val = param.split(':', 1)
                    params[key] = val

        tools.append({"name": tool_name, "parameters": params})

    # Remove tool markers from text
    clean_text = TOOL_PATTERN.sub('', text).strip()

    return clean_text, tools


def extract_stream_chunk_text(chunk):
    """Return streamed text content or an empty string for non-content chunks."""
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""

    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return ""

    content = getattr(delta, "content", None)
    return content if isinstance(content, str) else ""


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
        disable_thinking=True,
    ):
        self.model_name = model_name
        self.stream = stream
        self.gen_kwargs = dict(gen_kwargs)
        self.disable_thinking = disable_thinking
        self.chat = Chat(chat_size)
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial promt needs to be specified when setting init_chat_role."
                )
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
        self.user_role = user_role
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.warmup()

    def _build_request_kwargs(self, messages):
        request_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "stream": self.stream,
            **self.gen_kwargs,
        }

        if self.disable_thinking:
            extra_body = dict(request_kwargs.get("extra_body") or {})
            chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
            chat_template_kwargs["enable_thinking"] = False
            extra_body["chat_template_kwargs"] = chat_template_kwargs
            request_kwargs["extra_body"] = extra_body

        return request_kwargs

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        start = time.time()
        response = self.client.chat.completions.create(
            **self._build_request_kwargs([
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ])
        )
        end = time.time()
        logger.info(
            f"{self.__class__.__name__}:  warmed up! time: {(end - start):.3f} s"
        )

    def process(self, prompt):
        logger.debug("call api language model...")

        language_code = None
        if isinstance(prompt, tuple):
            prompt, language_code = prompt
            if language_code[-5:] == "-auto":
                language_code = language_code[:-5]
                prompt = f"Please reply to my message in {WHISPER_LANGUAGE_TO_LLM_LANGUAGE[language_code]}. " + prompt

        self.chat.append({"role": self.user_role, "content": prompt})

        response = self.client.chat.completions.create(
            **self._build_request_kwargs(self.chat.to_list())
        )
        if self.stream:
            generated_text, printable_text = "", ""
            for chunk in response:
                new_text = extract_stream_chunk_text(chunk)
                if not new_text:
                    continue

                generated_text += new_text
                printable_text += new_text
                sentences = sent_tokenize(printable_text)
                if len(sentences) > 1:
                    clean_text, tools = parse_tool_calls(sentences[0])
                    yield clean_text, language_code, tools
                    printable_text = new_text
            self.chat.append({"role": "assistant", "content": generated_text})
            # don't forget last sentence
            clean_text, tools = parse_tool_calls(printable_text)
            yield clean_text, language_code, tools
        else:
            generated_text = response.choices[0].message.content
            self.chat.append({"role": "assistant", "content": generated_text})
            clean_text, tools = parse_tool_calls(generated_text)
            yield clean_text, language_code, tools

    def on_session_end(self):
        self.chat.reset()
        logger.debug("OpenAI API language model chat state reset")
