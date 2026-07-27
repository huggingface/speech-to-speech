from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VoiceStoreArguments:
    """Configuration for the cloned-voice library (realtime mode, Qwen3-TTS).

    Kept separate from the Qwen3 TTS arguments because the store is
    TTS-agnostic; only the route gating is Qwen-specific today.
    """

    voice_store_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Local directory holding the cloned-voice library (one folder per voice). Defaults to ~/.cache/speech-to-speech/voices."
        },
    )
    voice_store_hub_repo: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional Hugging Face Hub dataset repo id (e.g. 'org/voices') used as the durable, fleet-consistent source of truth for the voice library. Authentication uses the standard HF_TOKEN environment variable. When unset, voices only persist in the local directory (single-instance deployments)."
        },
    )
