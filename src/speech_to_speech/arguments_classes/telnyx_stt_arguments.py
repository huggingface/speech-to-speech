from dataclasses import dataclass, field


@dataclass
class TelnyxSTTHandlerArguments:
    telnyx_stt_api_key: str = field(
        default="",
        metadata={"help": "Telnyx API key. Defaults to $TELNYX_API_KEY env var."},
    )
    telnyx_stt_engine: str = field(
        default="Telnyx",
        metadata={"help": "STT engine: Telnyx, Deepgram, Google, or Azure."},
    )
    telnyx_stt_language: str = field(
        default="en",
        metadata={"help": "Language code or 'auto'."},
    )
    telnyx_stt_model: str = field(
        default="",
        metadata={"help": "Engine-specific model name (e.g. 'nova-3' for Deepgram)."},
    )
    telnyx_stt_partial_results: bool = field(
        default=True,
        metadata={"help": "Request partial transcript results for live transcription."},
    )
    telnyx_stt_gen_kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "Reserved for pipeline compatibility."},
    )
