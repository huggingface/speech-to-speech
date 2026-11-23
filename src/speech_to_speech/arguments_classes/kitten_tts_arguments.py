from dataclasses import dataclass, field


@dataclass
class KittenTTSHandlerArguments:
    kitten_model_name: str = field(
        default="KittenML/kitten-tts-mini-0.8",
        metadata={
            "help": "The KittenTTS ONNX model repository or local repository directory to use. Default is 'KittenML/kitten-tts-mini-0.8'."
        },
    )
    kitten_device: str = field(
        default="cpu",
        metadata={"help": "KittenTTS runs on CPU.", "choices": ["cpu"]},
    )
    kitten_voice: str = field(
        default="Bruno",
        metadata={"help": "The voice to use for synthesis. Default is 'Bruno'."},
    )
