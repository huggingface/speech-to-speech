from dataclasses import dataclass, field


@dataclass
class FasterWhisperSTTHandlerArguments:
    faster_whisper_stt_model_name: str = field(
        default="tiny.en",
        metadata={
            "help": """The pretrained Faster Whisper model to use.
            One of ('tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'distil-small.en', 'medium', 'medium.en', 'distil-medium.en', 'large-v1', 'large-v2', 'large-v3', 'large', 'distil-large-v2', 'distil-large-v3').
            Default is 'small'."""
        },
    )
    faster_whisper_stt_device: str = field(
        default="auto",
        metadata={
            "help": """The device type on which the model will run.
            One of ('cpu', 'cuda', 'auto').
            Default is 'auto'."""
        },
    )
    faster_whisper_stt_compute_type: str = field(
        default="auto",
        metadata={
            "help": """The data type to use for computation.
            One of ('default', 'auto', 'int8', 'int8_float32', 'int8_float16', 'int8_bfloat16', 'int16', 'float16', 'float32', 'bfloat16')
            Default is 'auto'.
            Refer to 'https://opennmt.net/CTranslate2/quantization.html#quantize-on-model-loading'"""
        },
    )
    faster_whisper_stt_gen_max_new_tokens: int = field(
        default=128,
        metadata={
            "help": "The maximum number of new tokens to generate. Default is 128."
        },
    )
    faster_whisper_stt_gen_beam_size: int = field(
        default=1,
        metadata={
            "help": "The number of beams for beam search. Default is 1, implying greedy decoding."
        },
    )
    faster_whisper_stt_gen_return_timestamps: bool = field(
        default=False,
        metadata={
            "help": "Whether to return timestamps with transcriptions. Default is False."
        },
    )
    faster_whisper_stt_gen_task: str = field(
        default="transcribe",
        metadata={
            "help": "The task to perform, typically 'transcribe' for transcription. Default is 'transcribe'."
        },
    )
    faster_whisper_stt_gen_language: str = field(
        default="en",
        metadata={
            "help": "The language of the speech to transcribe. Default is 'en' for English."
        },
    )
