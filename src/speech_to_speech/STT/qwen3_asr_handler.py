from __future__ import annotations

import logging
from typing import Any, Iterator

from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logger = logging.getLogger(__name__)
console = Console()

# ISO 639-1 (plus 'yue' for Cantonese, not part of the standard) -> the language
# name Qwen3-ASR expects/returns, per https://huggingface.co/Qwen/Qwen3-ASR-1.7B
LANGUAGE_NAME_BY_CODE = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "hu": "Hungarian",
    "mk": "Macedonian",
    "ro": "Romanian",
}
CODE_BY_LANGUAGE_NAME = {name.lower(): code for code, name in LANGUAGE_NAME_BY_CODE.items()}


class Qwen3ASRSTTHandler(BaseSTTHandler):
    """
    Handles Speech To Text generation using Qwen3-ASR (an audio-LLM ASR model).
    """

    def setup(
        self,
        model_name: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        language: str | None = None,
        max_new_tokens: int = 256,
        max_inference_batch_size: int = 1,
        gen_kwargs: dict[str, Any] = {},
    ) -> None:
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Qwen3-ASR STT requires the optional 'qwen3-asr' extra. "
                "Install it with `pip install speech-to-speech[qwen3-asr]`."
            ) from exc

        self.start_language = language
        self.last_language_code = language if language != "auto" else None
        self.language_name = self._code_to_language_name(self.last_language_code)

        self.model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=getattr(torch, torch_dtype),
            device_map=device,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )
        self.warmup()

    def _code_to_language_name(self, language_code: str | None) -> str | None:
        if language_code is None:
            return None
        language_name = LANGUAGE_NAME_BY_CODE.get(language_code)
        if language_name is None:
            raise ValueError(
                f"Unsupported --qwen3_asr_language {language_code!r}. "
                f"Supported codes: {sorted(LANGUAGE_NAME_BY_CODE)}, or 'auto'."
            )
        return language_name

    def warmup(self) -> None:
        import numpy as np

        logger.info(f"Warming up {self.__class__.__name__}")
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1s of silence at 16kHz
        self.model.transcribe(audio=(dummy_audio, 16000), language=self.language_name)

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        logger.debug("infering Qwen3-ASR...")

        results = self.model.transcribe(audio=(vad_audio.audio, 16000), language=self.language_name)
        result = results[0]
        pred_text = result.text
        language_code = CODE_BY_LANGUAGE_NAME.get(result.language.lower(), result.language) if result.language else None

        if language_code is not None:
            self.last_language_code = language_code

        logger.debug("finished Qwen3-ASR inference")
        console.print(f"[yellow]USER: {pred_text}")
        logger.debug(f"Language Code Qwen3-ASR: {language_code}")

        if self.start_language == "auto" and language_code is not None:
            language_code += "-auto"

        yield Transcription(
            text=pred_text,
            language_code=language_code,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )
