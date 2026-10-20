from dataclasses import dataclass, field


@dataclass
class PhononTTSHandlerArguments:
    phonon_tts_model_name: str | None = field(
        default="gradium/phonon-40m-multilingual",
        metadata={
            "help": "Phonon checkpoint directory, config.json path, or Hugging Face repo ID. Defaults to gradium/phonon-40m-multilingual (Hub access required)."
        },
    )
    phonon_tts_revision: str | None = field(
        default=None,
        metadata={"help": "Hugging Face checkpoint revision (branch, tag, or commit)."},
    )
    phonon_tts_voice: str | None = field(
        default=None,
        metadata={"help": "Voice name from the checkpoint. Omit to use the checkpoint's default voice."},
    )
    phonon_tts_language: str = field(
        default="en",
        metadata={
            "help": "Initial text normalization language: en, fr, de, es, pt, or none. Multilingual checkpoints follow session language selections."
        },
    )
    phonon_tts_quant: str = field(
        default="q8",
        metadata={
            "help": "Phonon weight format. Default q8 loads model.q8.gguf; the checkpoint must supply the selected format."
        },
    )
    phonon_tts_blocksize: int = field(
        default=512,
        metadata={"help": "Number of 16 kHz PCM16 samples in each output block. Default 512."},
    )
    phonon_tts_num_threads: int | None = field(
        default=None,
        metadata={
            "help": "Positive CPU thread count for Phonon's process-wide native pool, set before model load. Omit to keep the runtime default."
        },
    )
