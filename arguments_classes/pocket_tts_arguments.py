from dataclasses import dataclass, field


@dataclass
class PocketTTSHandlerArguments:
    pocket_tts_device: str = field(
        default="cpu",
        metadata={
            "help": "The device type on which the Pocket TTS model will run. Options: 'cpu', 'cuda', 'mps'. Default is 'cpu'."
        },
    )
    pocket_tts_voice: str = field(
        default="jean",
        metadata={
            "help": "Voice to use for Pocket TTS. Can be a preset name ('alba', 'marius', 'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma'), a local audio file path, or a HuggingFace path like 'hf://kyutai/tts-voices/...'. Default is 'jean'."
        },
    )
    pocket_tts_sample_rate: int = field(
        default=16000,
        metadata={
            "help": "Output sample rate in Hz for Pocket TTS. Pocket TTS uses 24kHz internally but will be resampled to this rate to match the audio output. Default is 16000 to match the pipeline's audio streamer."
        },
    )
    pocket_tts_blocksize: int = field(
        default=512,
        metadata={
            "help": "Size of audio blocks to yield for streaming in Pocket TTS. Default is 512."
        },
    )
    pocket_tts_max_tokens: int = field(
        default=50,
        metadata={
            "help": "Maximum number of tokens to generate per sentence in Pocket TTS. Default is 50."
        },
    )
