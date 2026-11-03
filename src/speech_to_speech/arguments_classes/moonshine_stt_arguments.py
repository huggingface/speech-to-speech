from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MoonshineSTTHandlerArguments:
    moonshine_stt_model_name: str = field(
        default="moonshine-ai/moonshine-base",
        metadata={
            "help": (
                "The English Moonshine checkpoint to load through Transformers: 'moonshine-ai/moonshine-tiny' (27M), "
                "'moonshine-ai/moonshine-base' (61M), or 'moonshine-ai/moonshine-streaming-{tiny,small,medium}' "
                "(34M/123M/245M). Default is 'moonshine-ai/moonshine-base'."
            )
        },
    )
    moonshine_stt_device: str = field(
        default="auto",
        metadata={
            "help": "The device to run on: 'auto' (first available of CUDA, NPU, XPU, MPS, CPU), 'cuda', 'npu', "
            "'xpu', 'mps' or 'cpu'. Default is 'auto'."
        },
    )
    moonshine_stt_torch_dtype: str = field(
        default="auto",
        metadata={
            "help": "The model dtype. 'auto' picks bfloat16 on supported CUDA devices (float16 otherwise), "
            "float16 on MPS and float32 elsewhere. Default is 'auto'."
        },
    )
    moonshine_stt_tokens_per_second: float = field(
        default=6.5,
        metadata={
            "help": "Decoding stops after this many tokens per second of audio, which bounds hallucination loops. "
            "Default is 6.5, as recommended by the model card."
        },
    )
    moonshine_stt_num_threads: Optional[int] = field(
        default=None,
        metadata={"help": "Optional PyTorch CPU thread count. Default leaves PyTorch's setting unchanged."},
    )
