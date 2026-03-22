import logging
import os
import re

from LLM.openai_api_language_model import OpenApiModelHandler

logger = logging.getLogger(__name__)

# Pattern to strip thinking tags from MiniMax M2.5/M2.7 responses
THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class MiniMaxModelHandler(OpenApiModelHandler):
    """
    MiniMax LLM handler built on top of the OpenAI-compatible API handler.

    Adds MiniMax-specific defaults (base URL, model, API key auto-detection)
    and temperature clamping to stay within MiniMax's accepted range.
    """

    def setup(
        self,
        model_name="MiniMax-M2.7",
        device="cuda",
        gen_kwargs={},
        base_url="https://api.minimax.io/v1",
        api_key=None,
        stream=False,
        user_role="user",
        chat_size=1,
        init_chat_role="system",
        init_chat_prompt="You are a helpful AI assistant.",
    ):
        if api_key is None:
            api_key = os.environ.get("MINIMAX_API_KEY")
            if not api_key:
                raise ValueError(
                    "MiniMax API key is required. Set it via --minimax_api_key or "
                    "the MINIMAX_API_KEY environment variable."
                )

        # Clamp temperature to MiniMax's accepted range [0.0, 1.0]
        gen_kwargs = dict(gen_kwargs)
        if "temperature" in gen_kwargs:
            gen_kwargs["temperature"] = max(0.0, min(1.0, gen_kwargs["temperature"]))

        super().setup(
            model_name=model_name,
            device=device,
            gen_kwargs=gen_kwargs,
            base_url=base_url,
            api_key=api_key,
            stream=stream,
            user_role=user_role,
            chat_size=chat_size,
            init_chat_role=init_chat_role,
            init_chat_prompt=init_chat_prompt,
            disable_thinking=False,
        )

    def process(self, prompt):
        for text, language_code, tools in super().process(prompt):
            # Strip thinking tags that MiniMax models may include
            if text:
                text = THINK_TAG_PATTERN.sub("", text).strip()
            yield text, language_code, tools
