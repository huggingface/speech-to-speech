import logging
import re
from LLM.chat import Chat
from baseHandler import BaseHandler
from mlx_lm import load, stream_generate, generate
from rich.console import Console
from utils.mlx_lock import MLXLockContext
import mlx.core as mx
import torch

logger = logging.getLogger(__name__)

console = Console()

# Emoji pattern for removal
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002700-\U000027BF"  # dingbats
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)

WHISPER_LANGUAGE_TO_LLM_LANGUAGE = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "hi": "hindi",
    "de": "german",
    "pt": "portuguese",
    "pl": "polish",
    "it": "italian",
    "nl": "dutch",
}

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
        self.model, self.tokenizer = load(self.model_name)
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

    def remove_emojis(self, text):
        """Remove emoji characters from text."""
        return EMOJI_PATTERN.sub('', text)

    def remove_special_tokens(self, text):
        """Remove special tokens like <|end|>, <|im_end|>, and <|im_start|>."""
        text = text.replace("<|end|>", "").replace("<|im_end|>", "")
        text = text.split("<|im_start|>")[0]  # Remove any hallucinated user turn
        return text

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        dummy_input_text = "Repeat the word 'home'."
        dummy_chat = [{"role": self.user_role, "content": dummy_input_text}]

        n_steps = 2

        for _ in range(n_steps):
            prompt = self.tokenizer.apply_chat_template(dummy_chat, tokenize=False)
            generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.gen_kwargs["max_new_tokens"],
                verbose=False,
            )

    def process(self, prompt):
        logger.debug("infering language model...")
        language_code = None

        if isinstance(prompt, tuple):
            prompt, language_code = prompt
            if language_code[-5:] == "-auto":
                language_code = language_code[:-5]
                prompt = f"Please reply to my message in {WHISPER_LANGUAGE_TO_LLM_LANGUAGE[language_code]}. " + prompt

        self.chat.append({"role": self.user_role, "content": prompt})

        # Remove system messages if using a Gemma model
        if "gemma" in self.model_name.lower():
            chat_messages = [
                msg for msg in self.chat.to_list() if msg["role"] != "system"
            ]
        else:
            chat_messages = self.chat.to_list()

        prompt = self.tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        output = ""
        curr_output = ""
        end_of_turn = False

        # Acquire global MLX lock to prevent concurrent access with STT/TTS
        with MLXLockContext(handler_name="MLX-LLM", timeout=10.0):
            for t in stream_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=self.gen_kwargs["max_new_tokens"],
            ):
                output += t.text
                curr_output += t.text

                # Check for end-of-turn tokens and stop generation
                # This prevents the LLM from hallucinating user messages
                if "<|im_end|>" in curr_output or "<|end|>" in curr_output or "<|im_start|>" in curr_output:
                    # Clean up and yield any remaining content
                    curr_output = self.remove_special_tokens(curr_output)
                    curr_output = self.remove_emojis(curr_output)
                    if len(curr_output.strip()) > 0:
                        yield (curr_output.strip(), language_code)
                    end_of_turn = True
                    break

                if curr_output.endswith((".", "?", "!")):
                    if len(curr_output) > 0:
                        yield (curr_output, language_code)
                    curr_output = ""

            # Yield any remaining content if we didn't hit end of turn
            if not end_of_turn and len(curr_output.strip()) > 0:
                yield (curr_output.strip(), language_code)

        # Clean up the full output for chat history
        generated_text = self.remove_special_tokens(output).strip()

        # Clear MLX cache to free memory
        try:
            mx.clear_cache()
        except ImportError:
            pass
        torch.mps.empty_cache()

        self.chat.append({"role": "assistant", "content": generated_text})
