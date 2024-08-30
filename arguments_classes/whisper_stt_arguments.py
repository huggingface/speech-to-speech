from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WhisperSTTHandlerArguments:
    language: Optional[str] = field(
        default=None,
        metadata={
            "help": "The language for the conversation. Default is None."
        },
    )
    stt_model_name: str = field(
        default="distil-whisper/distil-large-v3",
        metadata={
            "help": "The pretrained Whisper model to use. Default is 'distil-whisper/distil-large-v3'."
        },
    )
    stt_device: str = field(
        default="cuda",
        metadata={
            "help": "The device type on which the model will run. Default is 'cuda' for GPU acceleration."
        },
    )
    stt_torch_dtype: str = field(
        default="float16",
        metadata={
            "help": "The PyTorch data type for the model and input tensors. One of `float32` (full-precision), `float16` or `bfloat16` (both half-precision)."
        },
    )
    stt_compile_mode: str = field(
        default=None,
        metadata={
            "help": "Compile mode for torch compile. Either 'default', 'reduce-overhead' and 'max-autotune'. Default is None (no compilation)"
        },
    )
    stt_gen_max_new_tokens: int = field(
        default=128,
        metadata={
            "help": "The maximum number of new tokens to generate. Default is 128."
        },
    )
    stt_gen_num_beams: int = field(
        default=1,
        metadata={
            "help": "The number of beams for beam search. Default is 1, implying greedy decoding."
        },
    )
    stt_gen_return_timestamps: bool = field(
        default=False,
        metadata={
            "help": "Whether to return timestamps with transcriptions. Default is False."
        },
    )
    stt_gen_task: str = field(
        default="transcribe",
        metadata={
            "help": "The task to perform, typically 'transcribe' for transcription. Default is 'transcribe'."
        },
    )
    language: Optional[str] = field(
        default=None,
        metadata={
            "help": "The language for the conversation. Default is None."
        },
    )