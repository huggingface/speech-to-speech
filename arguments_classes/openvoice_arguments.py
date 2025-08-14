from dataclasses import dataclass, field


@dataclass
class OpenVoiceArguments:
    openvoice_reference_audio: str = field(
        default=os.path.join(OPENVOICE_CACHE_DIR, "assets/resources/example_reference.mp3"),
        metadata={
            "help": "Path to the reference audio file for voice cloning."
        },
    )
    openvoice_base_speaker: str = field(
        default="EN-US",
        metadata={
            "help": "The base speaker ID from MeloTTS to use. e.g., 'EN-US', 'EN-BR', 'ZH', 'ES', 'FR'."
        },
    )
    openvoice_language: str = field(
        default="English",
        metadata={
            "help": "The language for the TTS generation. e.g., 'English', 'Chinese'."
        },
    )
    openvoice_speed: float = field(
        default=1.0,
        metadata={"help": "Speed of the generated speech."},
    )
