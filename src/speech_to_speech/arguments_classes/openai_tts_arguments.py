from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OpenAICompatibleTTSHandlerArguments:
    """Connection and audio settings for an OpenAI-compatible TTS server."""

    openai_tts_base_url: str = field(
        default="http://localhost:8091/v1",
        metadata={"help": "Base URL of the OpenAI-compatible server, including /v1."},
    )
    openai_tts_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "Optional bearer token. If unset, OPENAI_API_KEY is used when available."},
    )
    openai_tts_model: str = field(
        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        metadata={"help": "Model identifier sent to POST /audio/speech."},
    )
    openai_tts_voice: str = field(
        default="aiden",
        metadata={"help": "Voice name sent with speech requests."},
    )
    openai_tts_language: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional vLLM-Omni language extension. Use 'Auto' to forward the STT language when available."
        },
    )
    openai_tts_task_type: Optional[str] = field(
        default=None,
        metadata={"help": "Optional vLLM-Omni task type such as CustomVoice, VoiceDesign, or Base."},
    )
    openai_tts_instructions: Optional[str] = field(
        default=None,
        metadata={"help": "Optional voice style instructions for compatible servers."},
    )
    openai_tts_response_format: str = field(
        default="pcm",
        metadata={"help": "Audio response format. The adapter currently supports pcm and wav."},
    )
    openai_tts_sample_rate: int = field(
        default=24000,
        metadata={"help": "Sample rate of raw PCM returned by the server. Qwen3-TTS uses 24000."},
    )
    openai_tts_speed: float = field(
        default=1.0,
        metadata={"help": "Speech speed for compatible non-streaming endpoints."},
    )
    openai_tts_stream: bool = field(
        default=True,
        metadata={"help": "Request vLLM-Omni raw audio streaming extensions."},
    )
    openai_tts_timeout: float = field(
        default=300.0,
        metadata={"help": "HTTP request timeout in seconds."},
    )
    openai_tts_blocksize: int = field(
        default=512,
        metadata={"help": "Pipeline output chunk size in 16 kHz samples."},
    )
