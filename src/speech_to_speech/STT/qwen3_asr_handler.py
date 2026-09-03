from __future__ import annotations

import logging
from sys import platform
from time import perf_counter
from typing import Any, Iterator, Optional

import numpy as np
import torch
import transformers
from rich.console import Console
from transformers.models.qwen3_asr.processing_qwen3_asr import LANGUAGE_CODE_TO_NAME

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B-hf"
DEFAULT_LANGUAGE = "en"
SAMPLE_RATE = 16000

# The processor owns the language table: it validates forced languages against it and the
# model emits these names before ``<asr_text>`` when it identifies the language itself.
SUPPORTED_LANGUAGES = list(LANGUAGE_CODE_TO_NAME)
_LANGUAGE_NAME_TO_CODE = {name.lower(): code for code, name in LANGUAGE_CODE_TO_NAME.items()}


def language_to_code(language: Optional[str]) -> Optional[str]:
    """Map a language name or ISO code to the ISO code Qwen3-ASR supports. Unknown values give ``None``."""
    if not language:
        return None
    normalized = language.strip().lower()
    if normalized in LANGUAGE_CODE_TO_NAME:
        return normalized
    return _LANGUAGE_NAME_TO_CODE.get(normalized)


def resolve_device(device: str) -> str:
    """Turn ``auto`` into a concrete device the same way the other local handlers do."""
    if device != "auto":
        return device
    if platform == "darwin":
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_torch_dtype(torch_dtype: str, device: str) -> torch.dtype:
    """Turn ``auto`` into the dtype that runs well on ``device``; keep an explicit choice as is."""
    if torch_dtype != "auto":
        return getattr(torch, torch_dtype)
    if device.startswith("cuda"):
        return torch.bfloat16
    if device.startswith("mps"):
        return torch.float16
    return torch.float32


class Qwen3ASRSTTHandler(BaseSTTHandler):
    """Speech to text with a Qwen3-ASR checkpoint through Transformers.

    Language policy: with ``language="auto"`` the model identifies the language on
    every final turn and the code is reported with the ``-auto`` suffix. Progressive
    windows are short and fool the language ID, so they reuse the language of the
    last final turn instead. A forced language is passed on every request.
    """

    def setup(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        torch_dtype: str = "auto",
        language: Optional[str] = "auto",
        prompt: Optional[str] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        logger.info("Loading Qwen3-ASR STT model: %s", model_name)
        self.device = resolve_device(device)
        self.torch_dtype = resolve_torch_dtype(torch_dtype, self.device)
        self.prompt = prompt or None
        self.gen_kwargs = dict(gen_kwargs or {})
        self.configure_language(language)

        self.processor = transformers.AutoProcessor.from_pretrained(model_name)
        model = transformers.AutoModelForMultimodalLM.from_pretrained(model_name, dtype=self.torch_dtype)
        self.model = model.to(self.device).eval()
        self.warmup()

    def configure_language(self, language: Optional[str]) -> None:
        """Set the forced language, or ``None`` for per-turn detection."""
        requested = (language or "").strip()
        if requested.lower() in ("", "auto"):
            self.forced_language: Optional[str] = None
        else:
            # Pass unknown codes through: the processor validates what the checkpoint supports.
            self.forced_language = language_to_code(requested) or requested
        self.last_language: Optional[str] = self.forced_language

    def warmup(self) -> None:
        logger.info("Warming up %s", self.__class__.__name__)
        start = perf_counter()
        self._transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), self.forced_language)
        logger.info("%s: warmed up! time: %.3f s", self.__class__.__name__, perf_counter() - start)

    def _transcribe(self, audio: np.ndarray, language: Optional[str]) -> tuple[str, Optional[str]]:
        """Run one generation. Return the text and the language name the model identified (auto mode only)."""
        inputs = self.processor.apply_transcription_request(audio=audio, language=language, prompt=self.prompt)
        inputs = inputs.to(self.device, self.torch_dtype)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **self.gen_kwargs)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]

        text = self.processor.decode(generated_ids, return_format="transcription_only")[0].strip()
        if language is not None:
            return text, None
        parsed = self.processor.decode(generated_ids, return_format="parsed")[0]
        detected = parsed.get("language") if isinstance(parsed, dict) else None
        return text, detected

    def _final_language_code(self, detected: Optional[str]) -> str:
        if self.forced_language is not None:
            return self.forced_language
        code = language_to_code(detected)
        if code is not None:
            self.last_language = code
        elif detected:
            logger.warning("Qwen3-ASR detected unsupported language: %s", detected)
        return f"{self.last_language or DEFAULT_LANGUAGE}-auto"

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        progressive = vad_audio.mode == "progressive"
        request_language = self.forced_language
        if request_language is None and progressive:
            request_language = self.last_language
        audio = np.asarray(vad_audio.audio, dtype=np.float32)

        start = perf_counter()
        text, detected = self._transcribe(audio, request_language)
        logger.debug(
            "Qwen3-ASR %s transcription took %.3f s for %.2f s of audio",
            vad_audio.mode,
            perf_counter() - start,
            len(audio) / SAMPLE_RATE,
        )

        if progressive:
            yield PartialTranscription(
                text=text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
            )
            return

        language_code = self._final_language_code(detected)
        console.print(f"[yellow]USER: {text}")
        logger.debug("Language Code Qwen3-ASR: %s", language_code)
        yield Transcription(
            text=text,
            language_code=language_code,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )
