from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VLLMRealtimeSTTHandlerArguments:
    """Connection settings for vLLM's experimental Realtime transcription API."""

    vllm_realtime_stt_base_url: str = field(
        default="ws://localhost:8000/v1",
        metadata={"help": "Base URL for vLLM's Realtime API, including /v1."},
    )
    vllm_realtime_stt_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "Optional bearer token for the vLLM endpoint."},
    )
    vllm_realtime_stt_model: str = field(
        default="Qwen/Qwen3-ASR-1.7B",
        metadata={"help": "vLLM realtime-capable transcription model identifier."},
    )
    vllm_realtime_stt_audio_sample_rate: int = field(
        default=16000,
        metadata={"help": "PCM sample rate; current vLLM Realtime requires 16000."},
    )
    vllm_realtime_stt_connect_timeout: float = field(
        default=10.0,
        metadata={"help": "WebSocket connection and session-setup timeout in seconds."},
    )
    vllm_realtime_stt_final_timeout: float = field(
        default=60.0,
        metadata={"help": "Maximum wait for transcription.done after the final commit in seconds."},
    )
