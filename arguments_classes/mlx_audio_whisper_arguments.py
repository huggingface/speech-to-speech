from dataclasses import dataclass, field


@dataclass
class MLXAudioWhisperSTTHandlerArguments:
    mlx_audio_whisper_model_name: str = field(
        default="mlx-community/whisper-large-v3-turbo",
        metadata={
            "help": "The pretrained MLX Audio Whisper model to use. Default is 'mlx-community/whisper-large-v3-turbo'."
        },
    )
    mlx_audio_whisper_gen_kwargs: dict = field(
        default_factory=dict,
        metadata={
            "help": "Additional generation kwargs to pass to the model. Default is an empty dict."
        },
    )
