from dataclasses import dataclass, field
from typing import Literal, Optional

from speech_to_speech.arguments_classes.language_model_base_arguments import LanguageModelBaseArguments


@dataclass
class ResponsesApiLanguageModelHandlerArguments(LanguageModelBaseArguments):
    model_name: str = field(
        default="gpt-5.4-mini",
        metadata={
            "help": "The model to use with the OpenAI-compatible API. Default is 'gpt-5.4-mini', "
            "which is not audio-capable; --stt none requires an explicitly selected audio-input model."
        },
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
    responses_api_audio_max_tokens: int = field(
        default=256,
        metadata={"help": "Maximum chat completion tokens for audio-input LLM requests. Default is 256."},
    )
    responses_api_audio_temperature: float = field(
        default=0.0,
        metadata={"help": "Chat completion temperature for audio-input LLM requests. Default is 0.0."},
    )
    responses_api_audio_content_type: Literal["input_audio", "audio_url"] = field(
        default="input_audio",
        metadata={
            "help": "Chat Completions content representation for audio: 'input_audio' embeds WAV base64 directly; "
            "'audio_url' sends a base64 data URL. Default is 'input_audio'."
        },
    )
    responses_api_audio_history_turns: int = field(
        default=1,
        metadata={
            "help": "Number of recent completed audio user turns retained in history. Older audio is replaced with "
            "a role-preserving placeholder. Set to 0 to replace every completed audio turn. Default is 1."
        },
    )
