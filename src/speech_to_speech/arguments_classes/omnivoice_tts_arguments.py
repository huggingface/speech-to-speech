from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class OmniVoiceTTSHandlerArguments:
    omnivoice_model_name: str = field(
        default="k2-fsa/OmniVoice",
        metadata={"help": "OmniVoice Hugging Face model ID or local path. Default is 'k2-fsa/OmniVoice'."},
    )
    omnivoice_device: str = field(
        default="auto",
        metadata={
            "help": "Device passed to OmniVoice: 'auto', 'cuda', 'cuda:0', 'mps', 'xpu', or 'cpu'. Default is 'auto'."
        },
    )
    omnivoice_dtype: Literal["float16", "bfloat16", "float32"] = field(
        default="float16",
        metadata={
            "help": "Torch dtype for OmniVoice inference: 'float16', 'bfloat16', or 'float32'. Default is 'float16'."
        },
    )
    omnivoice_ref_audio: Optional[str] = field(
        default=None,
        metadata={"help": "Reference audio path for voice cloning. Requires --omnivoice_ref_text."},
    )
    omnivoice_ref_text: Optional[str] = field(
        default=None,
        metadata={"help": "Transcript of --omnivoice_ref_audio for voice cloning."},
    )
    omnivoice_voice_clone_prompt: Optional[str] = field(
        default=None,
        metadata={
            "help": "Saved OmniVoice VoiceClonePrompt path. Replaces --omnivoice_ref_audio and --omnivoice_ref_text."
        },
    )
    omnivoice_instruct: Optional[str] = field(
        default=None,
        metadata={"help": "Voice-design instruction. Leave unset for auto voice or voice cloning."},
    )
    omnivoice_language: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional target language name or code. When unset, the per-utterance pipeline language is used."
        },
    )
    omnivoice_num_steps: int = field(
        default=32,
        metadata={"help": "Number of OmniVoice diffusion decoding steps. Default is 32."},
    )
    omnivoice_speed: float = field(
        default=1.0,
        metadata={"help": "Speaking-speed factor (>1 faster, <1 slower). Default is 1.0."},
    )
    omnivoice_blocksize: int = field(
        default=512,
        metadata={"help": "Audio output block size in 16 kHz samples. Default is 512."},
    )
