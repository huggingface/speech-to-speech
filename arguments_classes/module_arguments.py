from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModuleArguments:
    device: Optional[str] = field(
        default=None,
        metadata={"help": "If specified, overrides the device for all handlers."},
    )
    mode: Optional[str] = field(
        default="socket",
        metadata={
            "help": "The mode to run the pipeline in. Either 'local', 'socket', or 'websocket'. Default is 'socket'."
        },
    )
    local_mac_optimal_settings: bool = field(
        default=False,
        metadata={
            "help": "If specified, sets the optimal settings for Mac OS. Hence whisper-mlx, MLX LM and MeloTTS will be used."
        },
    )
    stt: Optional[str] = field(
        default="whisper",
        metadata={
            "help": "The STT to use. Either 'whisper', 'whisper-mlx', 'mlx-audio-whisper', 'faster-whisper', 'parakeet-tdt', 'moonshine', or 'paraformer'. Default is 'whisper'."
        },
    )
    llm: Optional[str] = field(
        default="transformers",
        metadata={
            "help": "The LLM to use. Either 'transformers' or 'mlx-lm'. Default is 'transformers'"
        },
    )
    tts: Optional[str] = field(
        default="parler",
        metadata={
            "help": "The TTS to use. Either 'parler', 'melo', 'chatTTS', 'facebookMMS', 'pocket', 'kokoro', or 'kokoro-mlx'. Default is 'parler'"
        },
    )
    log_level: str = field(
        default="info",
        metadata={
            "help": "Provide logging level. Example --log_level debug, default=info."
        },
    )
    enable_live_transcription: bool = field(
        default=False,
        metadata={
            "help": "Enable live transcription display while user is speaking (only works with parakeet-tdt on MLX/MPS)"
        },
    )
    live_transcription_update_interval: float = field(
        default=0.25,
        metadata={
            "help": "Update interval for live transcription in seconds (default: 0.25s = 250ms)"
        },
    )
    live_transcription_min_silence_ms: int = field(
        default=500,
        metadata={
            "help": "Minimum silence duration (ms) before ending speech when live transcription is enabled (default: 500ms)"
        },
    )
