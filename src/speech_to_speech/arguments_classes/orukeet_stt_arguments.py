from dataclasses import dataclass, field


@dataclass
class OrukeetSTTHandlerArguments:
    orukeet_model_name: str = field(
        default="oruk/orukeet",
        metadata={"help": "The Orukeet NeMo ASR repository. Default is 'oruk/orukeet'."},
    )
    orukeet_checkpoint_filename: str = field(
        default="orukeet-v0.1.0.nemo",
        metadata={"help": "The NeMo checkpoint filename in the repository. Default is 'orukeet-v0.1.0.nemo'."},
    )
    orukeet_checkpoint_revision: str = field(
        default="555136b50265a132d4cea0d35560c26fc4f657ab",
        metadata={
            "help": "The Hugging Face revision that contains the NeMo checkpoint. Default is '555136b50265a132d4cea0d35560c26fc4f657ab'."
        },
    )
    orukeet_device: str = field(
        default="auto",
        metadata={
            "help": "The device to run on. Options: 'auto' (CUDA, then NPU, then CPU), 'cuda', 'npu', 'cpu'. Default is 'auto'."
        },
    )
    orukeet_language: str = field(
        default="auto",
        metadata={
            "help": "Fallback language code when text language detection does not return a code. Default is 'auto' (reports 'en' until the first detection)."
        },
    )
