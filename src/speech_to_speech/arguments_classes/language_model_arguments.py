from dataclasses import dataclass, field

from speech_to_speech.arguments_classes.language_model_base_arguments import LanguageModelBaseArguments


@dataclass
class LanguageModelHandlerArguments(LanguageModelBaseArguments):
    llm_device: str = field(
        default="cuda",
        metadata={"help": "The device type on which the model will run. Default is 'cuda' for GPU acceleration."},
    )
    llm_torch_dtype: str = field(
        default="float16",
        metadata={
            "help": "The PyTorch data type for the model and input tensors. One of `float32` (full-precision), `float16` or `bfloat16` (both half-precision)."
        },
    )
    llm_gen_max_new_tokens: int = field(
        default=1024,
        metadata={"help": "Maximum number of new tokens to generate in a single completion. Default is 1024."},
    )
    llm_gen_min_new_tokens: int = field(
        default=0,
        metadata={"help": "Minimum number of new tokens to generate in a single completion. Default is 0."},
    )
    llm_gen_temperature: float = field(
        default=0.0,
        metadata={
            "help": "Controls the randomness of the output. Set to 0.0 for deterministic (repeatable) outputs. Default is 0.0."
        },
    )
    llm_gen_do_sample: bool = field(
        default=False,
        metadata={"help": "Whether to use sampling; set this to False for deterministic outputs. Default is False."},
    )
    llm_is_vlm: bool = field(
        default=False,
        metadata={
            "help": "Set to True when using a Vision Language Model (VLM) that accepts image inputs. "
            "Loads AutoProcessor + AutoModelForImageTextToText instead of the default text-only model. "
            "Default is False."
        },
    )
