from dataclasses import dataclass, field


@dataclass
class ParakeetUnifiedSTTHandlerArguments:
    parakeet_unified_model_name: str = field(
        default="nvidia/parakeet-unified-en-0.6b",
        metadata={"help": "The NeMo Parakeet Unified ASR checkpoint. Default is 'nvidia/parakeet-unified-en-0.6b'."},
    )
    parakeet_unified_device: str = field(
        default="auto",
        metadata={"help": "The device to run on. 'auto' picks CUDA when available, otherwise CPU. Default is 'auto'."},
    )
    parakeet_unified_language: str = field(
        default="en",
        metadata={"help": "The language code reported with each final transcription. Default is 'en'."},
    )
