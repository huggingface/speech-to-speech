from dataclasses import dataclass, field


@dataclass
class ParlerTTSHandlerArguments:
    tts_model_name: str = field(
        default="ylacombe/parler-tts-mini-jenny-30H",
        metadata={
            "help": "The pretrained TTS model to use. Default is 'ylacombe/parler-tts-mini-jenny-30H'."
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
            "help": "Maximum number of new tokens to generate in a single completion. Default is 64, which corresponds to ~0.64 secs"
        },
    )
    tts_gen_max_new_tokens: int = field(
        default=512,
        metadata={
            "help": "Maximum number of new tokens to generate in a single completion. Default is 512, which corresponds to ~12 secs"
        },
    )
    description: str = field(
        default=(
            "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. "
            "She speaks very fast."
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
