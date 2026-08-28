from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OpenAIRealtimeSTTHandlerArguments:
    """Connection settings for an OpenAI Realtime transcription session."""

    openai_realtime_stt_base_url: str = field(
        default="wss://api.openai.com/v1",
        metadata={"help": "Base URL for the OpenAI Realtime API, including /v1."},
    )
    openai_realtime_stt_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "Optional bearer token. OPENAI_API_KEY is used for the official endpoint when unset."},
    )
    openai_realtime_stt_model: str = field(
        default="gpt-live-transcribe",
        metadata={"help": "Realtime transcription model identifier."},
    )
    openai_realtime_stt_language: Optional[str] = field(
        default=None,
        metadata={"help": "Optional language hint for the transcription session."},
    )
    openai_realtime_stt_audio_sample_rate: int = field(
        default=24000,
        metadata={"help": "PCM sample rate advertised to the Realtime transcription session."},
    )
    openai_realtime_stt_connect_timeout: float = field(
        default=10.0,
        metadata={"help": "WebSocket connection and session-setup timeout in seconds."},
    )
    openai_realtime_stt_final_timeout: float = field(
        default=60.0,
        metadata={"help": "Maximum wait for the committed final transcript in seconds."},
    )
