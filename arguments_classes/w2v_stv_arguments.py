"""This file contains the arguments for the Wav2Vec2STVHandler."""
from dataclasses import dataclass, field

@dataclass
class Wav2Vec2STVHandlerArguments:
    stv_model_name: str = field(
        default="bookbot/wav2vec2-ljspeech-gruut",
        metadata={
            "help": "The pretrained language model to use. Default is 'bookbot/wav2vec2-ljspeech-gruut'."
        },
    )
    stv_device: str = field(
        default="cuda",
        metadata={
            "help": "The device type on which the model will run. Default is 'cuda' for GPU acceleration."
        },
    )
    stv_blocksize: int = field(
        default=512,
        metadata={
            "help": "The blocksize of the model. Default is 512."
        },
    )
    stv_skip: bool = field(
        default=False,
        metadata={
            "help": "If True, skips the STV generation. Default is False."
        },
    )
