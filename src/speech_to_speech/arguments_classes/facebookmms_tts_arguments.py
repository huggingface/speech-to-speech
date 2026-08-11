from dataclasses import dataclass, field


@dataclass
class FacebookMMSTTSHandlerArguments:
    facebook_mms_model_name: str | None = field(
        default=None,
        metadata={"help": "Optional model override. By default, select the Facebook MMS model from --tts_language."},
    )
    tts_language: str = field(
        default="en",
        metadata={"help": "The language code for the TTS model. Default is 'en' for English."},
    )
    facebook_mms_device: str = field(
        default="cuda",
        metadata={"help": "The device to use for the TTS model. Default is 'cuda'."},
    )
    facebook_mms_torch_dtype: str = field(
        default="float32",
        metadata={"help": "The torch data type to use for the TTS model. Default is 'float32'."},
    )
