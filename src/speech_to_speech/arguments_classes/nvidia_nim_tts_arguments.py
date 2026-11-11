from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NvidiaNimTTSHandlerArguments:
    nvidia_nim_tts_base_url: str = field(
        default="http://localhost:9000",
        metadata={
            "help": "Base URL of the NVIDIA Speech NIM HTTP API, without /v1. Default is 'http://localhost:9000'."
        },
    )
    nvidia_nim_tts_voice: str = field(
        default="Magpie-Multilingual.EN-US.Aria",
        metadata={
            "help": (
                "NIM voice name, formatted as Magpie-Multilingual.{LOCALE}.{Speaker}[.{Emotion}]. "
                "Default is 'Magpie-Multilingual.EN-US.Aria'."
            )
        },
    )
    nvidia_nim_tts_language: str = field(
        default="en-US",
        metadata={"help": "Language code sent with every request, e.g. 'en-US'. Default is 'en-US'."},
    )
    nvidia_nim_tts_sample_rate: int = field(
        default=22050,
        metadata={"help": "Sample rate in Hz requested from the NIM. Default is 22050."},
    )
    nvidia_nim_tts_stream: bool = field(
        default=True,
        metadata={
            "help": (
                "Stream raw PCM from /v1/audio/synthesize_online. "
                "If false, fetch a WAV file from /v1/audio/synthesize. Default is true."
            )
        },
    )
    nvidia_nim_tts_timeout_s: float = field(
        default=30.0,
        metadata={"help": "Total timeout in seconds for one synthesis request. Must be > 0. Default is 30."},
    )
    nvidia_nim_tts_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "Optional bearer token sent as 'Authorization: Bearer ...', for NVIDIA-hosted endpoints."},
    )
    nvidia_nim_tts_voice_override: Optional[str] = field(
        default=None,
        metadata={"help": "If set, always send this voice to the NIM and ignore voices requested by realtime clients."},
    )
    nvidia_nim_tts_fix_date_commas: bool = field(
        default=True,
        metadata={"help": "Work around NIM text normalisers that speak the comma in 'Month D, YYYY' as 'sil'."},
    )
    nvidia_nim_tts_max_chars: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Split text longer than this many characters into separate requests. "
                "Default is auto: 450 for Chatterbox voices, otherwise 1800."
            )
        },
    )
