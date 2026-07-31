from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VisionResolverArguments:
    vision_model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The vision language model to use for vision resolution (OpenAI-compatible). "
            "If None, vision resolution is disabled and images go to the main model if it supports vision."
        },
    )
    vision_base_url: Optional[str] = field(
        default=None,
        metadata={"help": "Base URL for the OpenAI-compatible vision API endpoint. Default is None."},
    )
    vision_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "API key for the OpenAI-compatible vision API. Default is None."},
    )
    vision_max_tokens: int = field(
        default=300,
        metadata={"help": "Maximum tokens for vision resolver completion response. Default is 300."},
    )
    vision_timeout_s: float = field(
        default=10.0,
        metadata={"help": "Timeout in seconds for vision resolver calls. Default is 10.0."},
    )
