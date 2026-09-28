from dataclasses import dataclass, field


@dataclass
class TelnyxTTSHandlerArguments:
    telnyx_tts_api_key: str = field(
        default="",
        metadata={"help": "Telnyx API key. Defaults to $TELNYX_API_KEY env var."},
    )
    telnyx_tts_voice: str = field(
        default="Telnyx.NaturalHD.astra",
        metadata={
            "help": "Voice identifier. Telnyx.NaturalHD.*, AWS.Polly.*, Azure.*, ElevenLabs.*, MiniMax.*, ResembleAI.*, Inworld.*, Rime.*"
        },
    )
    telnyx_tts_sample_rate: int = field(
        default=16000,
        metadata={"help": "Output sample rate. Default 16000 to match pipeline audio output."},
    )
    telnyx_tts_blocksize: int = field(
        default=512,
        metadata={"help": "Size of audio blocks to yield. Default 512."},
    )
    telnyx_tts_gen_kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "Reserved for pipeline compatibility."},
    )
