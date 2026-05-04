from dataclasses import dataclass, field
from typing import Optional

from speech_to_speech.arguments_classes.language_model_base_arguments import LanguageModelBaseArguments


@dataclass
class OpenApiLanguageModelHandlerArguments(LanguageModelBaseArguments):
    model_name: str = field(
        default="deepseek-chat",
        metadata={"help": "The model to use with the OpenAI-compatible API. Default is 'deepseek-chat'."},
    )
    open_api_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "API key used to authenticate access to the OpenAI-compatible API. Default is None."},
    )
    open_api_base_url: Optional[str] = field(
        default=None,
        metadata={"help": "Base URL for the OpenAI-compatible API endpoint. Default is None (uses OpenAI)."},
    )
    open_api_stream: bool = field(
        default=False,
        metadata={"help": "Whether to stream responses from the API. Default is False."},
    )
    open_api_disable_thinking: bool = field(
        default=True,
        metadata={
            "help": "Disable provider-side thinking/reasoning when supported by the OpenAI-compatible backend. "
            "For Together Qwen3.5 models this sends chat_template_kwargs.enable_thinking=false."
        },
    )
