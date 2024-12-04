from dataclasses import dataclass, field


@dataclass
class OllamaLanguageModelHandlerArguments:
    ollama_model_name: str = field(
        default="hf.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
        metadata={
            "help": "The pretrained language model to use. Default is 'hf.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF'."
        },
    )
    ollama_user_role: str = field(
        default="user",
        metadata={
            "help": "Role assigned to the user in the chat context. Default is 'user'."
        },
    )
    ollama_init_chat_role: str = field(
        default="system",
        metadata={
            "help": "Initial role for setting up the chat context. Default is 'system'."
        },
    )
    ollama_init_chat_prompt: str = field(
        default="You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less than 20 words.",
        metadata={
            "help": "The initial chat prompt to establish context for the language model. Default is 'You are a helpful AI assistant.'"
        },
    )
    ollama_gen_max_new_tokens: int = field(
        default=128,
        metadata={
            "help": "Maximum number of new tokens to generate in a single completion. Default is 128."
        },
    )
    ollama_gen_temperature: float = field(
        default=0.0,
        metadata={
            "help": "Controls the randomness of the output. Set to 0.0 for deterministic (repeatable) outputs. Default is 0.0."
        },
    )
    ollama_api_endpoint: str = field(
        default="http://localhost:11434",
        metadata={
            "help": "Ollama endpoint. Default is 'http://localhost:11434'"
        },
    )
    ollama_chat_size: int = field(
        default=2,
        metadata={
            "help": "Number of interactions assitant-user to keep for the chat. None for no limitations."
        },
    )
