from dataclasses import dataclass, field
from typing import Optional

from speech_to_speech.arguments_classes.responses_api_language_model_arguments import (
    ResponsesApiLanguageModelHandlerArguments,
)


@dataclass
class ChatCompletionsLanguageModelHandlerArguments(ResponsesApiLanguageModelHandlerArguments):
    """Arguments for the ``chat-completions`` LLM backend.

    Inherits the OpenAI-compatible connection fields from the Responses-API
    arguments (``responses_api_base_url`` / ``responses_api_api_key`` /
    ``responses_api_stream`` / ``responses_api_disable_thinking``) so the same
    CLI flags and launcher env vars drive both backends, and adds the
    Chat-Completions-only ``reasoning_effort`` knob.
    """

    responses_api_reasoning_effort: Optional[str] = field(
        default=None,
        metadata={
            "help": "Provider-specific reasoning level sent as extra_body={'reasoning_effort': <value>} on the "
            "Chat Completions request. Use to disable reasoning on providers where "
            "chat_template_kwargs.enable_thinking is ignored (e.g. 'none' / 'low'). When unset, falls back to "
            "the disable_thinking behaviour (chat_template_kwargs.enable_thinking=false). Default is None."
        },
    )
