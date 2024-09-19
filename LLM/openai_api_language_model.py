from openai import OpenAI
from LLM.chat import Chat
from baseHandler import BaseHandler
from rich.console import Console
import logging
import time
logger = logging.getLogger(__name__)

console = Console()
from nltk import sent_tokenize

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
    ):
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


def process(self, prompt):
        logger.debug("call api language model...")
        self.chat.append({"role": self.user_role, "content": prompt})

        language_code = None
        if isinstance(prompt, tuple):
            prompt, language_code = prompt

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": self.user_role, "content": prompt},
            ],
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
                    yield sentences[0], language_code
                    printable_text = new_text
            self.chat.append({"role": "assistant", "content": generated_text})
            # don't forget last sentence
            yield printable_text, language_code
        else:
            generated_text = response.choices[0].message.content
            self.chat.append({"role": "assistant", "content": generated_text})
            yield generated_text, language_code
