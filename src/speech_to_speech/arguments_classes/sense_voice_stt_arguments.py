from dataclasses import dataclass, field


@dataclass
class SenseVoiceSTTHandlerArguments:
    sense_voice_stt_model_name: str = field(
        default="iic/SenseVoiceSmall",
        metadata={
            "help": "The pretrained SenseVoice model to use. Default is 'iic/SenseVoiceSmall'. See https://github.com/FunAudioLLM/SenseVoice"
        },
    )
    sense_voice_stt_device: str = field(
        default="cuda",
        metadata={"help": "The device type on which the model will run. Default is 'cuda' for GPU acceleration."},
    )
    sense_voice_stt_language: str = field(
        default="auto",
        metadata={"help": "Decoding language: 'auto', 'zh', 'en', 'yue', 'ja', 'ko'. Default is 'auto' (auto-detect)."},
    )
