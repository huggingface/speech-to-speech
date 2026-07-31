from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Qwen3ASRSTTHandlerArguments:
    """
    Arguments for the Qwen3-ASR Speech-to-Text handler.

    Qwen3-ASR is an audio-LLM ASR family from Alibaba (bidirectional audio encoder +
    Qwen3 causal LM with audio-token injection), run through the `qwen-asr` package.
    """

    qwen3_asr_model_name: str = field(
        default="Qwen/Qwen3-ASR-1.7B",
        metadata={
            "help": "The Qwen3-ASR model to use (HuggingFace Hub ID or local path). "
            "Default is 'Qwen/Qwen3-ASR-1.7B'. 'Qwen/Qwen3-ASR-0.6B' is also available for lower latency."
        },
    )
    qwen3_asr_device: str = field(
        default="cuda",
        metadata={"help": "Device to run the model on. Options: 'cuda', 'cpu', 'mps'. Default is 'cuda'."},
    )
    qwen3_asr_torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Torch dtype for inference. Options: 'bfloat16', 'float16', 'float32'. Default is 'bfloat16'."},
    )
    qwen3_asr_language: Optional[str] = field(
        default=None,
        metadata={
            "help": "Target language code for transcription (ISO 639-1, e.g. 'en', 'zh'). If not specified or set "
            "to 'auto', the model auto-detects the language among its 30 supported languages."
        },
    )
    qwen3_asr_max_new_tokens: int = field(
        default=256,
        metadata={
            "help": "Maximum number of tokens to generate per utterance. Raise this for long audio input. Default is 256."
        },
    )
    qwen3_asr_max_inference_batch_size: int = field(
        default=1,
        metadata={
            "help": "Batch size limit for inference. -1 means unlimited. Default is 1 (one VAD-segmented utterance at a time)."
        },
    )
