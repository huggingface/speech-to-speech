from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy as np
import torch
from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logger = logging.getLogger(__name__)
console = Console()

SAMPLE_RATE = 16000


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _extract_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    text = getattr(result, "text", None)
    if text is not None:
        return str(text)
    if isinstance(result, dict) and "text" in result:
        return str(result["text"])
    return str(result)


class NemoASRSTTHandler(BaseSTTHandler):
    """Speech to text with a NeMo ASR checkpoint through ASRModel.transcribe."""

    def setup(
        self,
        model_name: str,
        device: str = "auto",
        language: str = "en",
        gen_kwargs: dict | None = None,
    ) -> None:
        logger.info("Loading NeMo ASR STT model: %s", model_name)
        self.device = resolve_device(device)
        self.language = language
        self.model_name = model_name
        self.gen_kwargs = dict(gen_kwargs or {})

        from nemo.collections.asr.models import ASRModel

        self.model = ASRModel.from_pretrained(model_name=model_name)
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        self.warmup()

    def warmup(self) -> None:
        logger.info("Warming up %s", self.__class__.__name__)
        try:
            self._transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
        except Exception:
            logger.warning("%s: warmup failed", self.__class__.__name__, exc_info=True)

    def _transcribe(self, audio: np.ndarray) -> str:
        results = self.model.transcribe([audio])
        if not results:
            return ""
        return _extract_text(results[0]).strip()

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        audio = np.asarray(vad_audio.audio, dtype=np.float32)
        text = self._transcribe(audio)
        if vad_audio.mode == "progressive":
            yield PartialTranscription(
                text=text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
            )
            return
        console.print(f"[yellow]USER: {text}")
        yield Transcription(
            text=text,
            language_code=self.language,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )
