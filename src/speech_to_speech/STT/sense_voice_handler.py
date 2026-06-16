from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy as np
import torch
from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


class SenseVoiceSTTHandler(BaseSTTHandler):
    """
    Handles Speech To Text using a FunASR SenseVoice model.

    SenseVoice is a non-autoregressive multilingual speech understanding model
    (50+ languages). Being non-autoregressive it is markedly faster than
    autoregressive models such as Whisper, which suits low-latency voice agents.
    This handler returns the transcript; SenseVoice can additionally tag
    language / emotion / audio events.
    """

    def setup(
        self,
        model_name: str = "iic/SenseVoiceSmall",
        device: str = "cuda",
        language: str = "auto",
        gen_kwargs: dict[str, Any] = {},
    ) -> None:
        self.device = device
        self.language = language
        try:
            from funasr import AutoModel
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SenseVoice STT requires the optional 'sensevoice' extra. "
                "Install it with `pip install speech-to-speech[sensevoice]`."
            ) from exc
        self._postprocess = rich_transcription_postprocess
        self.model = AutoModel(model=model_name, device=device, disable_update=True)
        self.warmup()

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}")
        dummy_input = np.zeros(16000, dtype=np.float32)
        _ = self.model.generate(dummy_input, cache={}, language=self.language, use_itn=True)

    def _empty_cache(self) -> None:
        if isinstance(self.device, str) and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self.device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        logger.debug("infering sensevoice...")

        res = self.model.generate(vad_audio.audio, cache={}, language=self.language, use_itn=True)
        pred_text = self._postprocess(res[0]["text"]).strip()
        self._empty_cache()

        logger.debug("finished sensevoice inference")
        console.print(f"[yellow]USER: {pred_text}")

        if vad_audio.mode == "progressive":
            yield PartialTranscription(
                text=pred_text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
            )
        else:
            yield Transcription(
                text=pred_text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
                speech_stopped_at_s=vad_audio.created_at_s,
            )
