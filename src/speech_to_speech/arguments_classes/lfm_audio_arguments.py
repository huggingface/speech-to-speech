from dataclasses import dataclass, field

from speech_to_speech.arguments_classes.language_model_base_arguments import LanguageModelBaseArguments


@dataclass
class LFMAudioModelHandlerArguments(LanguageModelBaseArguments):
    model_name: str = field(
        default="LiquidAI/LFM3-Audio-2B",
        metadata={"help": "The LFM3-Audio checkpoint to use. Default is 'LiquidAI/LFM3-Audio-2B'."},
    )
    llm_device: str = field(
        default="auto",
        metadata={"help": "The device type on which the model will run. 'auto' selects an available GPU or CPU."},
    )
    llm_torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "The PyTorch data type for the model: `float32` or `bfloat16`. Default is `bfloat16`."},
    )
    llm_gen_max_new_tokens: int = field(
        default=1024,
        metadata={
            "help": "Maximum number of text tokens and speech frames (80 ms each) in one reply. Default is 1024."
        },
    )
