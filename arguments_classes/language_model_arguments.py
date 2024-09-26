from dataclasses import dataclass, field


@dataclass
class LanguageModelHandlerArguments:
    lm_model_name: str = field(
        default="HuggingFaceTB/SmolLM-360M-Instruct",
        metadata={
            "help": "The pretrained language model to use. Default is 'HuggingFaceTB/SmolLM-360M-Instruct'."
        },
    )
    lm_device: str = field(
        default="cuda",
        metadata={
            "help": "The device type on which the model will run. Default is 'cuda' for GPU acceleration."
        },
    )
    lm_torch_dtype: str = field(
        default="float16",
        metadata={
            "help": "The PyTorch data type for the model and input tensors. One of `float32` (full-precision), `float16` or `bfloat16` (both half-precision)."
        },
    )
    user_role: str = field(
        default="user",
        metadata={
            "help": "Role assigned to the user in the chat context. Default is 'user'."
        },
    )
    init_chat_role: str = field(
        default="system",
        metadata={
            "help": "Initial role for setting up the chat context. Default is 'system'."
        },
    )
    init_chat_prompt: str = field(
        default="You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less than 20 words.",
        metadata={
            "help": "The initial chat prompt to establish context for the language model. Default is 'You are a helpful AI assistant.'"
        },
    )
    lm_gen_max_new_tokens: int = field(
        default=128,
        metadata={
            "help": "Maximum number of new tokens to generate in a single completion. Default is 128."
        },
    )
    lm_gen_min_new_tokens: int = field(
        default=0,
        metadata={
            "help": "Minimum number of new tokens to generate in a single completion. Default is 0."
        },
    )
    lm_gen_temperature: float = field(
        default=0.0,
        metadata={
            "help": "Controls the randomness of the output. Set to 0.0 for deterministic (repeatable) outputs. Default is 0.0."
        },
    )
    lm_gen_do_sample: bool = field(
        default=False,
        metadata={
            "help": "Whether to use sampling; set this to False for deterministic outputs. Default is False."
        },
    )
    chat_size: int = field(
        default=2,
        metadata={
            "help": "Number of interactions assitant-user to keep for the chat. None for no limitations."
        },
    )
