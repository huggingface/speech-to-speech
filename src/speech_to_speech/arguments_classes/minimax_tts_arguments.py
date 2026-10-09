from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MiniMaxTTSHandlerArguments:
    minimax_tts_api_key: Optional[str] = field(
        default=None,
        metadata={"help": ("MiniMax API key for TTS. Defaults to the MINIMAX_API_KEY environment variable.")},
    )
    minimax_tts_base_url: str = field(
        default="https://api.minimax.io",
        metadata={
            "help": (
                "Base URL for the MiniMax API. "
                "Use 'https://api.minimaxi.com' for the mainland China endpoint. "
                "Default is 'https://api.minimax.io'."
            )
        },
    )
    minimax_tts_model: str = field(
        default="speech-2.8-hd",
        metadata={
            "help": (
                "MiniMax TTS model to use. "
                "Options: 'speech-2.8-hd' (default), "
                "'speech-2.8-turbo' (faster). "
                "Default is 'speech-2.8-hd'."
            )
        },
    )
    minimax_tts_voice: str = field(
        default="English_Graceful_Lady",
        metadata={"help": ("MiniMax system or custom voice ID. Default is 'English_Graceful_Lady'.")},
    )
    minimax_tts_speed: float = field(
        default=1.0,
        metadata={"help": "Speech speed for MiniMax TTS. Range: [0.5, 2.0]. Default is 1.0."},
    )
    minimax_tts_vol: float = field(
        default=1.0,
        metadata={"help": "Volume for MiniMax TTS. Range: (0, 10]. Default is 1.0."},
    )
    minimax_tts_pitch: int = field(
        default=0,
        metadata={"help": "Pitch adjustment for MiniMax TTS. Range: [-12, 12]. Default is 0."},
    )
    minimax_tts_blocksize: int = field(
        default=512,
        metadata={"help": ("Pipeline audio chunk size in samples. Default is 512.")},
    )

    minimax_tts_timeout: float = field(
        default=30.0,
        metadata={"help": "Total timeout in seconds for one MiniMax synthesis request. Must be > 0."},
    )
