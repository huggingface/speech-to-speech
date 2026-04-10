from dataclasses import dataclass, field
from typing import Optional

from TTS.minimax_tts_handler import MINIMAX_TTS_VOICES


@dataclass
class MiniMaxTTSHandlerArguments:
    minimax_tts_api_key: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "MiniMax API key for TTS. Defaults to the MINIMAX_API_KEY environment variable."
            )
        },
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
                "Options: 'speech-2.8-hd' (default, highest quality), "
                "'speech-2.8-turbo' (faster). "
                "Default is 'speech-2.8-hd'."
            )
        },
    )
    minimax_tts_voice: str = field(
        default="English_Graceful_Lady",
        metadata={
            "help": (
                f"Voice ID for MiniMax TTS. Available voices: {', '.join(MINIMAX_TTS_VOICES)}. "
                "Default is 'English_Graceful_Lady'."
            )
        },
    )
    minimax_tts_speed: float = field(
        default=1.0,
        metadata={
            "help": "Speech speed for MiniMax TTS. Range: [0.5, 2.0]. Default is 1.0."
        },
    )
    minimax_tts_vol: float = field(
        default=1.0,
        metadata={
            "help": "Volume for MiniMax TTS. Range: (0, 10]. Default is 1.0."
        },
    )
    minimax_tts_pitch: int = field(
        default=0,
        metadata={
            "help": "Pitch adjustment for MiniMax TTS. Range: [-12, 12]. Default is 0."
        },
    )
    minimax_tts_blocksize: int = field(
        default=512,
        metadata={
            "help": (
                "Audio chunk size in samples for streaming output. "
                "Must match LocalAudioStreamer blocksize. Default is 512."
            )
        },
    )
