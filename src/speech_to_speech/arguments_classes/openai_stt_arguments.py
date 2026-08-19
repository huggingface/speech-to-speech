from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OpenAICompatibleSTTHandlerArguments:
    """Connection and admission settings for an OpenAI-compatible STT server."""

    openai_stt_base_url: str = field(
        default="http://localhost:8000/v1",
        metadata={"help": "Base URL of the OpenAI-compatible server, including /v1."},
    )
    openai_stt_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "Optional bearer token. If unset, OPENAI_API_KEY is used when available."},
    )
    openai_stt_model: Optional[str] = field(
        default="nvidia/parakeet-tdt-0.6b-v3",
        metadata={
            "help": "Optional model identifier sent to POST /audio/transcriptions. "
            "May be omitted when the server selects a model from the language."
        },
    )
    openai_stt_language: Optional[str] = field(
        default=None,
        metadata={"help": "Optional ISO language hint sent with each transcription request."},
    )
    openai_stt_response_format: str = field(
        default="json",
        metadata={"help": "Transcription response format. The adapter supports json and text."},
    )
    openai_stt_timeout: float = field(
        default=60.0,
        metadata={"help": "HTTP request timeout in seconds."},
    )
    openai_stt_max_concurrency: int = field(
        default=1,
        metadata={"help": "Aggregate in-flight request limit shared by pipelines using this endpoint."},
    )
    openai_stt_max_queue_size: int = field(
        default=8,
        metadata={"help": "Aggregate bounded admission queue size for this endpoint."},
    )
    openai_stt_progressive_min_interval: float = field(
        default=0.75,
        metadata={"help": "Minimum seconds between progressive dispatches for the same turn."},
    )
