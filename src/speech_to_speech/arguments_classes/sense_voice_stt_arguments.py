from dataclasses import dataclass, field


@dataclass
class SenseVoiceSTTHandlerArguments:
    sense_voice_stt_model_name: str = field(
        default="FunAudioLLM/SenseVoiceSmall",
        metadata={
            "help": "The Hugging Face model ID or local path for SenseVoice. "
            "Default is 'FunAudioLLM/SenseVoiceSmall'. See https://huggingface.co/FunAudioLLM/SenseVoiceSmall"
        },
    )
    sense_voice_stt_device: str = field(
        default="cuda",
        metadata={"help": "The device type on which the model will run. Default is 'cuda'."},
    )
    sense_voice_stt_language: str = field(
        default="auto",
        metadata={"help": "Decoding language: auto, zh, en, yue, ja, or ko. Default is 'auto'."},
    )
