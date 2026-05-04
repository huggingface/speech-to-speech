from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ModuleArguments:
    device: Optional[str] = field(
        default=None,
        metadata={"help": "If specified, overrides the device for all handlers."},
    )
    mode: Optional[Literal["local", "socket", "websocket", "realtime"]] = field(
        default="local",
        metadata={
            "help": "The mode to run the pipeline in. Either 'local', 'socket', 'websocket', or 'realtime'. Default is 'local'."
        },
    )
    local_mac_optimal_settings: bool = field(
        default=False,
        metadata={
            "help": "If specified, sets the optimal settings for Mac OS. Sets Parakeet TDT for STT, MLX LM for language model, and Qwen3-TTS for TTS, with MPS device and local mode."
        },
    )
    stt: Optional[Literal["whisper", "whisper-mlx", "mlx-audio-whisper", "faster-whisper", "parakeet-tdt", "paraformer"]] = field(
        default="whisper",
        metadata={
            "help": "The STT to use. Either 'whisper', 'whisper-mlx', 'mlx-audio-whisper', 'faster-whisper', 'parakeet-tdt', or 'paraformer'. Default is 'whisper'."
        },
    )
    llm_backend: Optional[Literal["transformers", "mlx-lm", "openai-api"]] = field(
        default="transformers",
        metadata={"help": "The LLM backend to use. Either 'transformers', 'mlx-lm', or 'openai-api'. Default is 'transformers'."},
    )
    tts: Optional[Literal["melo", "chatTTS", "facebookMMS", "pocket", "kokoro", "qwen3"]] = field(
        default=None,
        metadata={
            "help": "The TTS to use. Either 'melo', 'chatTTS', 'facebookMMS', 'pocket', 'kokoro', or 'qwen3'. Default is 'qwen3'."
        },
    )
    log_level: str = field(
        default="info",
        metadata={"help": "Provide logging level. Example --log_level debug, default=info."},
    )
    enable_live_transcription: bool = field(
        default=False,
        metadata={
            "help": "Enable live transcription display while user is speaking (only works with parakeet-tdt on MLX/MPS)"
        },
    )
    live_transcription_update_interval: float = field(
        default=0.25,
        metadata={"help": "Update interval for live transcription in seconds (default: 0.25s = 250ms)"},
    )
    live_transcription_min_silence_ms: int = field(
        default=500,
        metadata={
            "help": "Minimum silence duration (ms) before ending speech when live transcription is enabled (default: 500ms)"
        },
    )
