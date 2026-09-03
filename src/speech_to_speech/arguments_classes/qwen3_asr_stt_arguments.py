from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Qwen3ASRSTTHandlerArguments:
    qwen3_asr_model_name: str = field(
        default="Qwen/Qwen3-ASR-0.6B-hf",
        metadata={
            "help": (
                "The Qwen3-ASR checkpoint to load through Transformers. "
                "'Qwen/Qwen3-ASR-1.7B-hf' is more accurate and needs about 4 GB of VRAM. "
                "Default is 'Qwen/Qwen3-ASR-0.6B-hf'."
            )
        },
    )
    qwen3_asr_device: str = field(
        default="auto",
        metadata={"help": "The device to run on. 'auto' picks CUDA, then MPS, then CPU. Default is 'auto'."},
    )
    qwen3_asr_torch_dtype: str = field(
        default="auto",
        metadata={
            "help": (
                "The model dtype. 'auto' picks bfloat16 on CUDA, float16 on MPS and float32 on CPU. "
                "Also accepts 'float32', 'float16' or 'bfloat16'. Default is 'auto'."
            )
        },
    )
    qwen3_asr_language: str = field(
        default="auto",
        metadata={
            "help": (
                "The language of the speech as an ISO code ('en', 'fr', 'zh', ...) or a name ('English'). "
                "'auto' lets the model identify the language of each final turn and forwards it to the LLM "
                "and TTS. Default is 'auto'."
            )
        },
    )
    qwen3_asr_prompt: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional context or hotwords that bias the transcription, "
                "for example 'Vocabulary: Quilter, apostle.'. Default is None."
            )
        },
    )
    qwen3_asr_gen_max_new_tokens: int = field(
        default=256,
        metadata={"help": "The maximum number of tokens to generate per turn. Default is 256."},
    )
