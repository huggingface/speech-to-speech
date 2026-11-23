from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Drama3TTSHandlerArguments:
    drama3_base_url: str = field(
        default="http://localhost:8000/v1",
        metadata={"help": "SGLang Omni server URL, including /v1, serving Fish Audio Drama-3."},
    )
    drama3_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "Optional bearer token. Defaults to DRAMA3_API_KEY."},
    )
    drama3_model: str = field(
        default="fishaudio/drama-3",
        metadata={"help": "Drama-3 model identifier served by SGLang Omni."},
    )
    drama3_ref_audio: Optional[str] = field(
        default=None,
        metadata={"help": "Local reference audio file for voice cloning. Requires --drama3_ref_text."},
    )
    drama3_ref_text: Optional[str] = field(
        default=None,
        metadata={"help": "Transcript of --drama3_ref_audio."},
    )
    drama3_timeout: float = field(
        default=300.0,
        metadata={"help": "Total timeout in seconds for each speech request, including startup warmup."},
    )
    drama3_blocksize: int = field(
        default=512,
        metadata={"help": "Pipeline output chunk size in 16 kHz samples."},
    )
    drama3_gen_temperature: float = field(default=1.0, metadata={"help": "Drama-3 sampling temperature."})
    drama3_gen_top_p: float = field(default=0.9, metadata={"help": "Drama-3 nucleus sampling probability."})
    drama3_gen_top_k: int = field(
        default=4097,
        metadata={
            "help": "Sampling candidate count. The default retains every semantic code, matching the official Space."
        },
    )
