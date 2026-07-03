from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ModuleArguments:
    device: Optional[str] = field(
        default=None,
        metadata={"help": "If specified, overrides the device for all handlers."},
    )
    mode: Optional[Literal["local", "socket", "websocket", "realtime"]] = field(
        default="realtime",
        metadata={
            "help": "The mode to run the pipeline in. Either 'local', 'socket', 'websocket', or 'realtime'. Default is 'realtime'."
        },
    )
    local_mac_optimal_settings: bool = field(
        default=False,
        metadata={
            "help": "If specified, sets the optimal settings for Mac OS. Sets Parakeet TDT for STT, MLX LM for language model, and Qwen3-TTS for TTS, with MPS device and local mode."
        },
    )
    stt: Optional[
        Literal["whisper", "whisper-mlx", "mlx-audio-whisper", "faster-whisper", "parakeet-tdt", "paraformer"]
    ] = field(
        default="parakeet-tdt",
        metadata={
            "help": "The STT to use. Either 'whisper', 'whisper-mlx', 'mlx-audio-whisper', 'faster-whisper', 'parakeet-tdt', or 'paraformer'. Default is 'parakeet-tdt'."
        },
    )
    llm_backend: Optional[Literal["transformers", "mlx-lm", "responses-api", "chat-completions"]] = field(
        default="responses-api",
        metadata={
            "help": "The LLM backend to use. Either 'transformers', 'mlx-lm', 'responses-api', or "
            "'chat-completions' (OpenAI-compatible /v1/chat/completions). Default is 'responses-api'."
        },
    )
    tts: Optional[Literal["chatTTS", "facebookMMS", "pocket", "kokoro", "qwen3"]] = field(
        default="qwen3",
        metadata={
            "help": "The TTS to use. Either 'chatTTS', 'facebookMMS', 'pocket', 'kokoro', or 'qwen3'. Default is 'qwen3'."
        },
    )
    log_level: str = field(
        default="info",
        metadata={"help": "Provide logging level. Example --log_level debug, default=info."},
    )
    enable_live_transcription: bool = field(
        default=True,
        metadata={
            "help": "Enable live transcription display while user is speaking (works with parakeet-tdt). Default is true."
        },
    )
    live_transcription_update_interval: float = field(
        default=0.5,
        metadata={"help": "Update interval for live transcription in seconds (default: 0.5s = 500ms)"},
    )
    live_transcription_min_silence_ms: int = field(
        default=500,
        metadata={
            "help": "Minimum silence duration (ms) before ending speech when live transcription is enabled (default: 500ms)"
        },
    )
    num_pipelines: int = field(
        default=1,
        metadata={
            "help": "Number of isolated realtime pipelines in the pool. One uvicorn server listens on "
            "--ws_port and routes each incoming websocket to the next free pipeline (each has its own "
            "VAD/STT/LM/TTS handlers and conversation state). Max concurrent websocket sessions equals "
            "num_pipelines; further connections are rejected. Only valid for --mode realtime. Default is 1."
        },
    )
