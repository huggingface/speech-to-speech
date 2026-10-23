from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GraniteSpeechCTCSTTHandlerArguments:
    granite_ctc_model_name: str = field(
        default="ibm-granite/granite-speech-5.0-470m-turboctc",
        metadata={
            "help": "The Granite Speech TurboCTC checkpoint to load through Transformers. "
            "Default is 'ibm-granite/granite-speech-5.0-470m-turboctc'."
        },
    )
    granite_ctc_revision: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional checkpoint revision. '947f59af40db9791170a0628cf0f3f4812d720f1' is the earlier "
            "English-only release without punctuation. Default is the latest revision."
        },
    )
    granite_ctc_device: str = field(
        default="auto",
        metadata={
            "help": "The device to run on: 'auto' (first available of CUDA, NPU, XPU, MPS, CPU), 'cuda', 'npu', "
            "'xpu', 'mps' or 'cpu'. Default is 'auto'."
        },
    )
    granite_ctc_torch_dtype: str = field(
        default="auto",
        metadata={
            "help": "The model dtype. 'auto' picks bfloat16 on CUDA, float16 on MPS and float32 on CPU. Default is 'auto'."
        },
    )
    granite_ctc_language: str = field(
        default="en",
        metadata={
            "help": "Language reported with each transcript: en, fr, de, es, pt, it or ja. CTC decoding does not "
            "identify the language, so this is not detected. Default is 'en'."
        },
    )
    granite_ctc_num_threads: Optional[int] = field(
        default=None,
        metadata={"help": "Optional PyTorch CPU thread count. Default leaves PyTorch's setting unchanged."},
    )
