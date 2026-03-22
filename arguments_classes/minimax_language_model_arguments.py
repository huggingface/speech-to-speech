from dataclasses import dataclass, field


@dataclass
class MiniMaxLanguageModelHandlerArguments:
    minimax_model_name: str = field(
        default="MiniMax-M2.7",
        metadata={
            "help": "The MiniMax model to use. Default is 'MiniMax-M2.7'. "
            "Also available: 'MiniMax-M2.5', 'MiniMax-M2.5-highspeed'."
        },
    )
    minimax_user_role: str = field(
        default="user",
        metadata={
            "help": "Role assigned to the user in the chat context. Default is 'user'."
        },
    )
    minimax_init_chat_role: str = field(
        default="system",
        metadata={
            "help": "Initial role for setting up the chat context. Default is 'system'."
        },
    )
    minimax_init_chat_prompt: str = field(
        default="You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less than 20 words.",
        metadata={
            "help": "The initial chat prompt to establish context for the language model."
        },
    )
    minimax_chat_size: int = field(
        default=5,
        metadata={
            "help": "Number of interactions assistant-user to keep for the chat. None for no limitations."
        },
    )
    minimax_api_key: str = field(
        default=None,
        metadata={
            "help": "MiniMax API key. If not set, falls back to MINIMAX_API_KEY environment variable."
        },
    )
    minimax_base_url: str = field(
        default="https://api.minimax.io/v1",
        metadata={
            "help": "MiniMax API base URL. Default is 'https://api.minimax.io/v1'."
        },
    )
    minimax_stream: bool = field(
        default=False,
        metadata={
            "help": "Whether to use streaming mode for MiniMax API responses. Default is False."
        },
    )
