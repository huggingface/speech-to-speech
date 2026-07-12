from dataclasses import dataclass, field


@dataclass
class SupertonicTTSHandlerArguments:
    supertonic_voice: str = field(
        default="M1",
        metadata={
            "help": "Voice style for Supertonic TTS (e.g., M1, M2, F1, F2)."
        },
    )
    supertonic_lang: str = field(
        default="na",
        metadata={
            "help": "Language code for Supertonic TTS (default: 'na' for auto/language-agnostic)."
        },
    )
    supertonic_speed: float = field(
        default=1.0,
        metadata={
            "help": "Speed modifier for Supertonic TTS (e.g. 0.7 to 2.0)."
        },
    )
