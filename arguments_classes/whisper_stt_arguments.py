from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WhisperSTTHandlerArguments:
    whisper_model_name: str = field(
        default="distil-whisper/distil-large-v3",
        metadata={
            "help": "The pretrained Whisper model to use. Default is 'distil-whisper/distil-large-v3'."
        },
    )
    whisper_device: str = field(
        default="cuda",
        metadata={
            "help": "The device type on which the model will run. Default is 'cuda' for GPU acceleration."
        },
    )
    whisper_torch_dtype: str = field(
        default="float16",
        metadata={
            "help": "The PyTorch data type for the model and input tensors. One of `float32` (full-precision), `float16` or `bfloat16` (both half-precision)."
        },
    )
    whisper_compile_mode: str = field(
        default=None,
        metadata={
            "help": "Compile mode for torch compile. Either 'default', 'reduce-overhead' and 'max-autotune'. Default is None (no compilation)"
        },
    )
    whisper_gen_max_new_tokens: int = field(
        default=128,
        metadata={
            "help": "The maximum number of new tokens to generate. Default is 128."
        },
    )
    whisper_gen_num_beams: int = field(
        default=1,
        metadata={
            "help": "The number of beams for beam search. Default is 1, implying greedy decoding."
        },
    )
    whisper_gen_return_timestamps: bool = field(
        default=False,
        metadata={
            "help": "Whether to return timestamps with transcriptions. Default is False."
        },
    )
    whisper_gen_task: str = field(
        default="transcribe",
        metadata={
            "help": "The task to perform, typically 'transcribe' for transcription. Default is 'transcribe'."
        },
    )
    language: Optional[str] = field(
        default='en',
        metadata={
            "help": """The language for the conversation. 
            Choose between 'en' (english), 'fr' (french), 'es' (spanish), 
            'zh' (chinese), 'ko' (korean), 'ja' (japanese), or 'None'.
            If using 'auto', the language is automatically detected and can
            change during the conversation. Default is 'en'."""
        },
    )