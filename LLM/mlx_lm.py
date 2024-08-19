import logging
from LLM.chat import Chat
from baseHandler import BaseHandler
from mlx_lm import load, stream_generate, generate
from rich.console import Console
import torch
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()

class MLXLanguageModelHandler(BaseHandler):
    """
    Handles the language model part.
    """

    def setup(
        self,
        model_name="microsoft/Phi-3-mini-4k-instruct",
        device="mps",
        torch_dtype="float16",
        gen_kwargs={},
        user_role="user",
        chat_size=1,
        init_chat_role=None,
        init_chat_prompt="You are a helpful AI assistant.",
    ):
        self.model_name = model_name
        model_id = 'microsoft/Phi-3-mini-4k-instruct'
        self.model, self.tokenizer = load(model_id)
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

        dummy_input_text = "Write me a poem about Machine Learning."
        dummy_chat = [{"role": self.user_role, "content": dummy_input_text}]
        
        n_steps = 2

        for _ in range(n_steps):
            prompt = self.tokenizer.apply_chat_template(dummy_chat, tokenize=False)
            generate(self.model, self.tokenizer, prompt=prompt, max_tokens=self.gen_kwargs["max_new_tokens"], verbose=False)


    def process(self, prompt):
        logger.debug("infering language model...")

        self.chat.append({"role": self.user_role, "content": prompt})
        prompt = self.tokenizer.apply_chat_template(self.chat.to_list(), tokenize=False, add_generation_prompt=True)
        output = ""
        curr_output = ""
        for t in stream_generate(self.model, self.tokenizer, prompt, max_tokens=self.gen_kwargs["max_new_tokens"]):
            output += t
            curr_output += t
            if curr_output.endswith(('.', '?', '!', '<|end|>')):
                yield curr_output.replace('<|end|>', '')
                curr_output = ""
        generated_text = output.replace('<|end|>', '')
        torch.mps.empty_cache()

        self.chat.append({"role": "assistant", "content": generated_text})
