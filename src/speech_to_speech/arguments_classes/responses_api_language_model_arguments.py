from dataclasses import dataclass, field
from typing import Optional

from speech_to_speech.arguments_classes.language_model_base_arguments import LanguageModelBaseArguments


@dataclass
class ResponsesApiLanguageModelHandlerArguments(LanguageModelBaseArguments):
    model_name: str = field(
        default="gpt-5.4-mini",
        metadata={"help": "The model to use with the OpenAI-compatible API. Default is 'gpt-5.4-mini'."},
    )
    responses_api_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "API key used to authenticate access to the OpenAI-compatible API. Default is None."},
    )
    responses_api_base_url: Optional[str] = field(
        default=None,
        metadata={"help": "Base URL for the OpenAI-compatible API endpoint. Default is None (uses OpenAI)."},
    )
    responses_api_stream: bool = field(
        default=True,
        metadata={
            "help": "The stream parameter typically indicates whether data should be transmitted in a continuous flow rather"
            " than in a single, complete response, often used for handling large or real-time data.Default is True"
        },
    )
    responses_api_disable_thinking: bool = field(
        default=True,
        metadata={
            "help": "Disable provider-side thinking/reasoning when supported by the OpenAI-compatible backend. "
            "For Together Qwen3.5 models this sends chat_template_kwargs.enable_thinking=false."
        },
    )
    keenable_web_search: bool = field(
        default=False,
        metadata={
            "help": "Enable native live web search: the server advertises Keenable web_search/fetch_page tools "
            "to the model and executes them inside the pipeline, so clients need no tool handling. Default is False."
        },
    )
    keenable_api_key: Optional[str] = field(
        default=None,
        metadata={
            "help": "Keenable API key (keen_...). Falls back to the KEENABLE_API_KEY env var; with no key the "
            "keyless rate-limited free tier is used. Default is None."
        },
    )
    tool_call_max_rounds: int = field(
        default=3,
        metadata={
            "help": "Maximum LLM rounds per turn when server-executed tools (Keenable web search) are enabled; "
            "the final round forces a spoken answer. Default is 3."
        },
    )
