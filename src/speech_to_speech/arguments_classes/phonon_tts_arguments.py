from dataclasses import dataclass, field


@dataclass
class PhononTTSHandlerArguments:
    phonon_tts_model_name: str | None = field(
        default="gradium/phonon-31m-en",
        metadata={
            "help": "Phonon checkpoint directory, config.json path, or Hugging Face repo ID. Defaults to gradium/phonon-31m-en (public, Apache-2.0)."
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
            "help": "Fixed text normalization language: en, fr, de, es, pt, or none. This does not select a model."
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
    phonon_tts_num_threads: int = field(
        default=4,
        metadata={
            "help": "Positive CPU thread count for Phonon's process-wide native pool, set before model load. Default 4."
        },
    )
    phonon_tts_max_trailing_silence_ms: int | None = field(
        default=None,
        metadata={
            "help": "Opt-in quiet-tail limit per Phonon utterance in milliseconds (minimum 40). Retains speech and internal pauses; omitted preserves native audio. Try 120 for sentence-streamed speech."
        },
    )
