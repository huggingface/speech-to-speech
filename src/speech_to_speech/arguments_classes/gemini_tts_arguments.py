from dataclasses import dataclass, field
from typing import Optional

DEFAULT_GEMINI_TTS_PROMPT = (
    "Przeczytaj dostarczony tekst dokładnie, naturalnym polskim głosem, "
    "bez dodawania, usuwania ani parafrazowania treści."
)


@dataclass
class GeminiTTSHandlerArguments:
    gemini_tts_model_name: str = field(
        default="gemini-3.1-flash-tts-preview",
        metadata={"help": "Gemini TTS model name. Default is 'gemini-3.1-flash-tts-preview'."},
    )
    gemini_tts_voice: str = field(
        default="Kore",
        metadata={"help": "Gemini prebuilt voice. Default is 'Kore'."},
    )
    gemini_tts_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "Gemini API key. Falls back to GEMINI_API_KEY when unset."},
    )
    gemini_tts_prompt: str = field(
        default=DEFAULT_GEMINI_TTS_PROMPT,
        metadata={"help": "Instruction prepended to each Gemini TTS request."},
    )
    gemini_tts_timeout_s: float = field(
        default=20.0,
        metadata={"help": "Gemini TTS request timeout in seconds. Default is 20."},
    )
    gemini_tts_blocksize: int = field(
        default=512,
        metadata={"help": "Audio chunk size in 16 kHz output samples. Default is 512."},
    )
