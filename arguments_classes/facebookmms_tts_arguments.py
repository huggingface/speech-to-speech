from dataclasses import field, dataclass

@dataclass
class FacebookMMSTTSHandlerArguments:
    model_name: str = field(
        default="facebook/mms-tts-hin",
        metadata={
            "help": "The model name to use. Default is 'facebook/mms-tts-hin'."
        },
    )
    tts_language: str = field(  # Renamed to avoid conflict
        default="en",
        metadata={
            "help": "The language code for the TTS model. Default is 'en' for English."
        },
    )
    facebook_mms_device: str = field(
        default="cuda",
        metadata={
            "help": "The device to use for the TTS model. Default is 'cuda'."
        },
    )
    facebook_mms_torch_dtype: str = field(
        default="float32",
        metadata={
            "help": "The torch data type to use for the TTS model. Default is 'float32'."
        },
    )
    
