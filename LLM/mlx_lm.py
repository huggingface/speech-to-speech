import logging
import uuid
import time
from langdetect import detect
from LLM.chat import Chat
from baseHandler import BaseHandler
from mlx_lm import load, stream_generate, generate
from rich.console import Console
from textblob import TextBlob
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
        debug_mode=False,
        context_id=None,
    ):
        self.model_name = model_name
        self.model, self.tokenizer = load(self.model_name)
        self.gen_kwargs = gen_kwargs
        self.debug_mode = debug_mode

        self.context_id = context_id or str(uuid.uuid4())
        self.chat = Chat(chat_size, context_id=self.context_id)

        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial prompt needs to be specified when setting init_chat_role."
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
            generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.gen_kwargs.get("max_new_tokens", 50),
                verbose=False,
            )

    def process(self, prompt):
        start_time = time.time()
        logger.debug("Starting language model inference...")

        # Detect the language of the prompt
        language = detect(prompt)
        if language != "en":
            logger.info(f"Detected language: {language}. Adjusting model and tokenizer.")
            self.tokenizer = load_language_specific_tokenizer(language)
            self.model = load_language_specific_model(language)

        self.chat.append({"role": self.user_role, "content": prompt, "context_id": self.context_id})

        # Remove system messages if using a Gemma model
        if "gemma" in self.model_name.lower():
            chat_messages = [msg for msg in self.chat.to_list() if msg["role"] != "system"]
        else:
            chat_messages = self.chat.to_list()

        if self.debug_mode:
            logger.debug(f"Model name: {self.model_name}")
            logger.debug(f"Prompt before tokenization: {prompt}")

        prompt = self.tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
        output = ""
        curr_output = ""

        for t in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=self.gen_kwargs.get("max_new_tokens", 50),
        ):
            output += t
            curr_output += t
            if curr_output.endswith((".", "?", "!", "<|end|>")):
                yield curr_output.replace("<|end|>", "")
                curr_output = ""

        generated_text = output.replace("<|end|>", "")

        # Quality control
        response_quality = TextBlob(generated_text).sentiment
        if response_quality.polarity < 0.1:  # Example threshold
            generated_text += " Please provide more positive feedback."

        torch.mps.empty_cache()

        self.chat.append({"role": "assistant", "content": generated_text})
        
        end_time = time.time()
        logger.info(f"Processing completed in {end_time - start_time:.2f} seconds.")
