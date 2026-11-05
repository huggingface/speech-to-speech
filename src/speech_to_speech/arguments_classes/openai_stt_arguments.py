from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OpenAICompatibleSTTHandlerArguments:
    """Connection settings for an OpenAI-compatible STT server."""

    openai_stt_base_url: str = field(
        default="http://localhost:8000/v1",
        metadata={"help": "Base URL of the OpenAI-compatible server, including /v1."},
    )
    openai_stt_api_key: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional bearer token. For https://api.openai.com/v1 only, "
            "OPENAI_API_KEY is used when this flag is unset."
        },
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
    openai_stt_max_window_s: Optional[float] = field(
        default=None,
        metadata={
            "help": "Maximum audio duration per HTTP upload (seconds, at least 1). Unset preserves whole-turn uploads."
        },
    )
    openai_stt_profile: str = field(
        default="transcription",
        metadata={"help": "transcription uses the Audio API; nemotron uses native vLLM audio chat with annotations."},
    )
    openai_stt_recognition_context: str = field(
        default="",
        metadata={"help": "Optional recognition context for the nemotron profile; not assistant instructions."},
    )
    openai_stt_context_terms: list[str] = field(
        default_factory=list,
        metadata={"help": "Optional vocabulary spellings for the nemotron profile, listed once in the prompt."},
    )
    openai_stt_max_output_tokens: int = field(
        default=384,
        metadata={"help": "Output token budget for the nemotron profile. Truncated transcripts fail the request."},
    )
