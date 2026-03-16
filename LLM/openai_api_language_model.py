import base64
import logging
import mimetypes
import re
import time
from pathlib import Path

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


class OpenApiModelHandler(BaseHandler):
    """
    Handles the language model part.
    """
    def setup(
        self,
        model_name="deepseek-chat",
        device="cuda",
        gen_kwargs={},
        base_url =None,
        api_key=None,
        stream=False,
        user_role="user",
        chat_size=1,
        init_chat_role="system",
        init_chat_prompt="You are a helpful AI assistant.",
        image_paths=None,
        image_urls=None,
        image_detail="auto",
    ):
        self.model_name = model_name
        self.stream = stream
        self.chat = Chat(chat_size)
        self.image_detail = image_detail
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial promt needs to be specified when setting init_chat_role."
                )
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
        self.user_role = user_role
        self.default_images = self._normalize_default_images(image_paths, image_urls)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ],
            stream=self.stream
        )
        end = time.time()
        logger.info(
            f"{self.__class__.__name__}:  warmed up! time: {(end - start):.3f} s"
        )

    def _split_csv_arg(self, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return [item for item in value if item]

    def _path_to_data_url(self, image_path):
        path = Path(image_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")

        mime_type, _ = mimetypes.guess_type(path.name)
        if mime_type is None:
            mime_type = "image/jpeg"

        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _build_image_part(self, image_url, detail=None):
        return {
            "type": "image_url",
            "image_url": {
                "url": image_url,
                "detail": detail or self.image_detail,
            },
        }

    def _normalize_default_images(self, image_paths, image_urls):
        images = []

        for image_url in self._split_csv_arg(image_urls):
            images.append(self._build_image_part(image_url))

        for image_path in self._split_csv_arg(image_paths):
            images.append(self._build_image_part(self._path_to_data_url(image_path)))

        return images

    def _normalize_prompt(self, prompt):
        prompt_text = prompt
        language_code = None
        images = list(self.default_images)

        if isinstance(prompt, tuple):
            prompt_text, language_code = prompt

        if isinstance(prompt_text, dict):
            language_code = prompt_text.get("language_code", language_code)
            images.extend(prompt_text.get("images", []))
            prompt_text = prompt_text.get("text", "")

        if language_code and language_code.endswith("-auto"):
            language_code = language_code[:-5]
            prompt_text = (
                f"Please reply to my message in "
                f"{WHISPER_LANGUAGE_TO_LLM_LANGUAGE[language_code]}. {prompt_text}"
            )

        return prompt_text, language_code, images

    def _build_user_message(self, prompt_text, images):
        if not images:
            return {"role": self.user_role, "content": prompt_text}

        content = []
        if prompt_text:
            content.append({"type": "text", "text": prompt_text})
        content.extend(images)
        return {"role": self.user_role, "content": content}

    def process(self, prompt):
        logger.debug("call api language model...")

        prompt_text, language_code, images = self._normalize_prompt(prompt)
        self.chat.append(self._build_user_message(prompt_text, images))

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.chat.to_list(),
            stream=self.stream
        )
        if self.stream:
            generated_text, printable_text = "", ""
            for chunk in response:
                new_text = chunk.choices[0].delta.content or ""
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
