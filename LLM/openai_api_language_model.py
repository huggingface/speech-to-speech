from openai import OpenAI
from LLM.chat import Chat
from baseHandler import BaseHandler
from rich.console import Console
import logging
import time
logger = logging.getLogger(__name__)

console = Console()


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
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": self.user_role, "content": prompt},
            ],
            stream=self.stream
        )
        generated_text = response.choices[0].message.content
        self.chat.append({"role": "assistant", "content": generated_text})
        yield generated_text
