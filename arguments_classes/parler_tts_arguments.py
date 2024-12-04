from dataclasses import dataclass, field


@dataclass
class ParlerTTSHandlerArguments:
    tts_model_name: str = field(
        default="parler-tts/parler-mini-v1-jenny",
        metadata={
            "help": "The pretrained TTS model to use. Default is 'parler-tts/parler-mini-v1-jenny'."
        },
    )
    tts_device: str = field(
        default="cuda",
        metadata={
            "help": "The device type on which the model will run. Default is 'cuda' for GPU acceleration."
        },
    )
    tts_torch_dtype: str = field(
        default="float16",
        metadata={
            "help": "The PyTorch data type for the model and input tensors. One of `float32` (full-precision), `float16` or `bfloat16` (both half-precision)."
        },
    )
    tts_compile_mode: str = field(
        default=None,
        metadata={
            "help": "Compile mode for torch compile. Either 'default', 'reduce-overhead' and 'max-autotune'. Default is None (no compilation)"
        },
    )
    tts_gen_min_new_tokens: int = field(
        default=64,
        metadata={
            "help": "Maximum number of new tokens to generate in a single completion. Default is 64, which corresponds to ~0.74 secs"
        },
    )
    tts_gen_max_new_tokens: int = field(
        default=1024,
        metadata={
            "help": "Maximum number of new tokens to generate in a single completion. Default is 1024, which corresponds to ~12 secs"
        },
    )
    description: str = field(
        default=(
            "Jenny speaks at a slightly slow pace with an animated delivery with clear audio quality."
        ),
        metadata={
            "help": "Description of the speaker's voice and speaking style to guide the TTS model."
        },
    )
    play_steps_s: float = field(
        default=1.0,
        metadata={
            "help": "The time interval in seconds for playing back the generated speech in steps. Default is 1.0 seconds."
        },
    )
    max_prompt_pad_length: int = field(
        default=8,
        metadata={
            "help": "When using compilation, the prompt as to be padded to closest power of 2. This parameters sets the maximun power of 2 possible."
        },
    )
    use_default_speakers_list: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the default list of speakers or not."
        },
    )
