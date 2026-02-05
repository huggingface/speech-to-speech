from dataclasses import dataclass, field


@dataclass
class KokoroTTSHandlerArguments:
    kokoro_model_name: str = field(
        default="hexgrad/Kokoro-82M",
        metadata={
            "help": "The Kokoro TTS model to use. Default is 'hexgrad/Kokoro-82M'. For MLX use 'mlx-community/Kokoro-82M-bf16'."
        },
    )
    kokoro_device: str = field(
        default="cuda",
        metadata={
            "help": "The device to run Kokoro TTS on. Options: 'cuda', 'cpu', 'mps'. Default is 'cuda'."
        },
    )
    kokoro_voice: str = field(
        default="bm_fable",
        metadata={
            "help": "The voice to use for synthesis. See VOICES.md in the Kokoro repo for options. Default is 'bm_fable' (British male)."
        },
    )
    kokoro_lang_code: str = field(
        default="b",
        metadata={
            "help": "Language code: 'a' for American English, 'b' for British English, 'j' for Japanese, etc. Default is 'b'."
        },
    )
    kokoro_speed: float = field(
        default=1.0,
        metadata={
            "help": "Speech speed multiplier. Values > 1.0 speed up, < 1.0 slow down. Default is 1.0."
        },
    )
    kokoro_blocksize: int = field(
        default=512,
        metadata={
            "help": "The audio chunk size in samples for streaming output. Default is 512."
        },
    )


@dataclass
class KokoroMLXTTSHandlerArguments:
    kokoro_mlx_model_name: str = field(
        default="mlx-community/Kokoro-82M-bf16",
        metadata={
            "help": "The MLX Kokoro TTS model to use. Default is 'mlx-community/Kokoro-82M-bf16'."
        },
    )
    kokoro_mlx_voice: str = field(
        default="bm_fable",
        metadata={
            "help": "The voice to use for synthesis. See VOICES.md in the Kokoro repo for options. Default is 'bm_fable' (British male)."
        },
    )
    kokoro_mlx_lang_code: str = field(
        default="b",
        metadata={
            "help": "Language code: 'a' for American English, 'b' for British English, 'j' for Japanese, etc. Default is 'b'."
        },
    )
    kokoro_mlx_speed: float = field(
        default=1.0,
        metadata={
            "help": "Speech speed multiplier. Values > 1.0 speed up, < 1.0 slow down. Default is 1.0."
        },
    )
    kokoro_mlx_blocksize: int = field(
        default=512,
        metadata={
            "help": "The audio chunk size in samples for streaming output. Default is 512."
        },
    )
