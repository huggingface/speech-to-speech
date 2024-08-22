from dataclasses import dataclass, field


@dataclass
class MeloTTSHandlerArguments:
    melo_language: str = field(
        default="EN_NEWEST",
        metadata={
            "help": "The language of the text to be synthesized. Default is 'EN_NEWEST'."
        },
    )
    melo_device: str = field(
        default="auto",
        metadata={
            "help": "The device to be used for speech synthesis. Default is 'auto'."
        },
    )
    melo_speaker_to_id: str = field(
        default="EN-Newest",
        metadata={
            "help": "Mapping of speaker names to speaker IDs. Default is ['EN-Newest']."
        },
    )
