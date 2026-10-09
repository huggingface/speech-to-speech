from dataclasses import dataclass, field


@dataclass
class SupertonicTTSHandlerArguments:
    supertonic_tts_voice: str = field(
        default="M1",
        metadata={"help": "Voice style for Supertonic TTS (M1-M5 or F1-F5). Default is M1."},
    )
    supertonic_tts_lang: str = field(
        default="na",
        metadata={"help": "Language code for Supertonic TTS (default: 'na' for auto/language-agnostic)."},
    )
    supertonic_tts_speed: float = field(
        default=1.0,
        metadata={"help": "Speed modifier for Supertonic TTS (0.7 to 2.0). Default is 1.0."},
    )
    supertonic_tts_blocksize: int = field(
        default=512,
        metadata={"help": "Audio chunk size in samples for streaming output. Default is 512."},
    )
