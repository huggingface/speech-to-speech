from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LightningWhisperSTTHandlerArguments:
    stt_model_name: str = field(
        default="distil-large-v3",
        metadata={"help": "The Lightning Whisper MLX model to use."},
    )
    stt_device: str = field(
        default="mps",
        metadata={"help": "Device used for cache cleanup. Lightning Whisper inference runs on MLX."},
    )
    language: Optional[str] = field(
        default="en",
        metadata={"help": "Transcription language, or 'auto' to detect it. Default is 'en'."},
    )
