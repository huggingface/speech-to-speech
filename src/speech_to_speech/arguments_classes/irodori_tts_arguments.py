from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IrodoriTTSHandlerArguments:
    irodori_model_name: str = field(
        default="Aratako/Irodori-TTS-v4.1-Small-MF",
        metadata={
            "help": "Irodori-TTS Hugging Face repository, or a local model.safetensors path. "
            "Default is 'Aratako/Irodori-TTS-v4.1-Small-MF', the four-step MeanFlow checkpoint."
        },
    )
    irodori_device: str = field(
        default="auto",
        metadata={"help": "Device for the model and its codec: 'auto', 'cuda', 'xpu', 'mps', or 'cpu'."},
    )
    irodori_caption: Optional[str] = field(
        default=None,
        metadata={
            "help": "Japanese description of the voice and speaking style, for checkpoints with caption conditioning. "
            "When neither this nor --irodori_ref_audio is set, a calm young female voice is used."
        },
    )
    irodori_ref_audio: Optional[str] = field(
        default=None,
        metadata={"help": "Reference audio path for zero-shot voice cloning. Can be combined with --irodori_caption."},
    )
    irodori_num_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Sampling steps. Default is the checkpoint's own: 4 for MeanFlow, 40 for Rectified Flow."},
    )
    irodori_blocksize: int = field(
        default=512,
        metadata={"help": "Audio output block size in 16 kHz samples. Default is 512."},
    )
