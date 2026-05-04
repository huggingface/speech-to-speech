from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OpenApiLanguageModelHandlerArguments:
    open_api_model_name: str = field(
        # default="HuggingFaceTB/SmolLM-360M-Instruct",
        default="gpt-5.4-mini",
        metadata={"help": "The OpenAI-compatible language model to use. Default is 'gpt-5.4-mini'."},
    )
    open_api_user_role: str = field(
        default="user",
        metadata={"help": "Role assigned to the user in the chat context. Default is 'user'."},
    )
    open_api_init_chat_role: str = field(
        default="system",
        metadata={"help": "Initial role for setting up the chat context. Default is 'system'."},
    )
    open_api_init_chat_prompt: str = field(
        # default="You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less than 20 words.",
        default="You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less than 20 words.",
        metadata={
            "help": "The initial chat prompt to establish context for the language model. Default is 'You are a helpful AI assistant.'"
        },
    )

    open_api_chat_size: int = field(
        default=30,
        metadata={"help": "Number of interactions assitant-user to keep for the chat. None for no limitations."},
    )
    open_api_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "Is a unique code used to authenticate and authorize access to an API.Default is None"},
    )
    open_api_base_url: Optional[str] = field(
        default=None,
        metadata={
            "help": "Is the root URL for all endpoints of an API, serving as the starting point for constructing API request.Default is Non"
        },
    )
    open_api_stream: bool = field(
        default=True,
        metadata={
            "help": "The stream parameter typically indicates whether data should be transmitted in a continuous flow rather"
            " than in a single, complete response, often used for handling large or real-time data.Default is True"
        },
    )
    open_api_disable_thinking: bool = field(
        default=True,
        metadata={
            "help": "Disable provider-side thinking/reasoning when supported by the OpenAI-compatible backend. "
            "For Together Qwen3.5 models this sends chat_template_kwargs.enable_thinking=false."
        },
    )
    open_api_stream_batch_sentences: int = field(
        default=3,
        metadata={
            "help": "Number of sentences to accumulate before yielding a batch during streaming. "
            "Set to 1 for sentence-by-sentence streaming. Default is 3."
        },
    )
    open_api_enable_lang_prompt: bool = field(
        default=False,
        metadata={
            "help": "When True, append a user message instructing the model to reply in the detected/selected "
            "language (e.g. 'Please reply to my message in French.'). Default is False."
        },
    )
