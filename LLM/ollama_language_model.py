from threading import Thread
from ollama import Client
import torch

from LLM.chat import Chat
from baseHandler import BaseHandler
from rich.console import Console
import logging
from nltk import sent_tokenize

logger = logging.getLogger(__name__)

console = Console()


WHISPER_LANGUAGE_TO_LLM_LANGUAGE = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "hi": "hindi",
}

class OllamaLanguageModelHandler(BaseHandler):
    """
    Handles the language model part.
    """

    def setup(
        self,
        model_name="hf.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
        device="",
        torch_dtype="",
        gen_kwargs={},
        api_endpoint=None,
        user_role="user",
        chat_size=1,
        init_chat_role=None,
        init_chat_prompt="You are a helpful AI assistant.",
    ):
        self.model_name = model_name
        self.client = Client(host=api_endpoint)

        self.gen_kwargs = gen_kwargs

        self.chat = Chat(chat_size)
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial promt needs to be specified when setting init_chat_role."
                )
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
        self.user_role = user_role

        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        self.client.chat(model=self.model_name, messages=[])

    def process(self, prompt):
        logger.debug("infering language model...")
        language_code = None

        if isinstance(prompt, tuple):
            prompt, language_code = prompt
            if language_code[-5:] == "-auto":
                language_code = language_code[:-5]
                prompt = f"Please reply to my message in {WHISPER_LANGUAGE_TO_LLM_LANGUAGE[language_code]}. " + prompt

        self.chat.append({"role": self.user_role, "content": prompt})

        stream = self.client.chat(
            model=self.model_name,
            messages=self.chat.to_list(),
            stream=True,
        )

        generated_text = ""
        for chunk in stream:
            chunk_text = chunk['message']['content']
            generated_text += chunk_text
            print(chunk_text, end='', flush=True)
        
        self.chat.append({"role": "assistant", "content": generated_text})
