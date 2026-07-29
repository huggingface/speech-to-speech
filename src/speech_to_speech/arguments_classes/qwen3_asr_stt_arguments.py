from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Qwen3ASRSTTHandlerArguments:
    qwen3_asr_model_name: str = field(
        default="Qwen/Qwen3-ASR-0.6B-hf",
        metadata={"help": "The Qwen3-ASR model to use. Default is 'Qwen/Qwen3-ASR-0.6B-hf'."},
    )
    qwen3_asr_device: str = field(
        default="auto",
        metadata={
            "help": "Device to run Qwen3-ASR on. 'auto' will use CUDA when available, then MPS, then CPU. "
            "Options: 'auto', 'cuda', 'mps', 'cpu'. Default is 'auto'."
        },
    )
    qwen3_asr_torch_dtype: str = field(
        default="bfloat16",
        metadata={
            "help": "PyTorch dtype for Qwen3-ASR. One of 'float32', 'float16', or 'bfloat16'. Default is 'bfloat16'."
        },
    )
    qwen3_asr_language: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional transcription language hint, such as 'English' or 'en'. "
            "If omitted, Qwen3-ASR auto-detects the language."
        },
    )
    qwen3_asr_prompt: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional context or hotwords prompt to bias Qwen3-ASR transcription. Default is None."
        },
    )
    qwen3_asr_compile_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional torch.compile mode for Qwen3-ASR. Use values like 'default', "
            "'reduce-overhead', or 'max-autotune'. Default is None."
        },
    )
    qwen3_asr_gen_max_new_tokens: int = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to generate for Qwen3-ASR transcription. Default is 256."},
    )
    qwen3_asr_gen_do_sample: bool = field(
        default=False,
        metadata={"help": "Whether to sample during Qwen3-ASR generation. Default is False."},
    )
