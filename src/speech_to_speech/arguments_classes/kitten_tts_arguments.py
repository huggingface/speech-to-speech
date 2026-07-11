from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KittenTTSHandlerArguments:
    kitten_model_name: Optional[str] = field(
        default="KittenML/kitten-tts-mini-0.8",
        metadata={
            "help": "The KittenTTS model to use. Default is 'KittenML/kitten-tts-mini-0.8'."
        },
    )
    kitten_device: str = field(
        default="cpu",
        metadata={
            "help": "The device to run KittenTTS on. Options: 'cuda', 'cpu'. Default is 'cpu'."
        },
    )
    kitten_voice: str = field(
        default="Bruno",
        metadata={
            "help": "The voice to use for synthesis. Default is 'Bruno'."
        },
    )
