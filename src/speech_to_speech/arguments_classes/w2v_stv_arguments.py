"""Arguments for the optional Wav2Vec2 speech-to-viseme stage."""

from dataclasses import dataclass, field


@dataclass
class Wav2Vec2STVHandlerArguments:
    enable_visemes: bool = field(default=False, metadata={"help": "Generate timed visemes for assistant audio."})
    stv_model_name: str = field(
        default="bookbot/wav2vec2-ljspeech-gruut",
        metadata={"help": "Phoneme-recognition checkpoint used for visemes. The default is English."},
    )
    stv_device: str = field(default="auto", metadata={"help": "Viseme model device: auto, cpu, cuda, or mps."})
