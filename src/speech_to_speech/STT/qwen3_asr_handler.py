from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version
from sys import platform
from typing import Any, Iterator, Optional

import numpy as np
import torch
from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logger = logging.getLogger(__name__)
console = Console()

MIN_TRANSFORMERS_VERSION = (5, 13, 0)
QWEN3_ASR_LANGUAGE_TO_CODE = {
    "arabic": "ar",
    "cantonese": "yue",
    "chinese": "zh",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "filipino": "fil",
    "finnish": "fi",
    "french": "fr",
    "german": "de",
    "greek": "el",
    "hindi": "hi",
    "hungarian": "hu",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "macedonian": "mk",
    "malay": "ms",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "spanish": "es",
    "swedish": "sv",
    "thai": "th",
    "turkish": "tr",
    "vietnamese": "vi",
}


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = []
    for part in value.split(".")[:3]:
        digits = ""
        for char in part:
            if char.isdigit():
                digits += char
            else:
                break
        parts.append(int(digits or 0))
    return tuple([*parts, 0, 0, 0][:3])


def _require_qwen3_asr_transformers() -> None:
    try:
        installed = version("transformers")
    except PackageNotFoundError as exc:
        raise ModuleNotFoundError(
            "Qwen3-ASR STT requires Transformers. Install it with "
            "`pip install speech-to-speech[qwen3-asr]`."
        ) from exc

    if _version_tuple(installed) < MIN_TRANSFORMERS_VERSION:
        raise ImportError(
            "Qwen3-ASR STT requires transformers>=5.13.0 because older versions do not recognize "
            "`model_type: qwen3_asr`. Install it with `pip install speech-to-speech[qwen3-asr]` "
            f"or upgrade Transformers. Found transformers=={installed}."
        )


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if platform == "darwin" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _language_to_code(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    normalized = language.strip().lower()
    return QWEN3_ASR_LANGUAGE_TO_CODE.get(normalized, normalized)


class Qwen3ASRSTTHandler(BaseSTTHandler):
    """Final-transcript STT handler for Qwen3-ASR Transformers checkpoints."""

    def setup(
        self,
        model_name: str = "Qwen/Qwen3-ASR-0.6B-hf",
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        compile_mode: Optional[str] = None,
        gen_kwargs: dict[str, Any] = {},
    ) -> None:
        _require_qwen3_asr_transformers()

        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.device = _resolve_device(device)
        self.torch_dtype = getattr(torch, torch_dtype)
        self.language = language
        self.prompt = prompt
        self.gen_kwargs = gen_kwargs

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForMultimodalLM.from_pretrained(model_name, dtype=self.torch_dtype).to(self.device)
        self.model.eval()

        if compile_mode:
            self.model.forward = torch.compile(self.model.forward, mode=compile_mode)

    def _prepare_inputs(self, audio: np.ndarray) -> Any:
        kwargs: dict[str, Any] = {"audio": audio}
        if self.language:
            kwargs["language"] = self.language
        if self.prompt:
            kwargs["prompt"] = self.prompt

        inputs = self.processor.apply_transcription_request(**kwargs)
        return inputs.to(self.device, self.torch_dtype)

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        if vad_audio.mode == "progressive":
            return

        logger.debug("inferring qwen3-asr...")
        inputs = self._prepare_inputs(vad_audio.audio)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **self.gen_kwargs)

        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        pred_text = self.processor.decode(generated_ids, return_format="transcription_only")[0]
        parsed = self.processor.decode(generated_ids, return_format="parsed")[0]
        parsed_language = parsed.get("language") if isinstance(parsed, dict) else None
        language_code = _language_to_code(parsed_language)

        logger.debug("finished qwen3-asr inference")
        console.print(f"[yellow]USER: {pred_text}")

        yield Transcription(
            text=pred_text,
            language_code=language_code,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )
