from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy as np
import torch
from funasr import AutoModel
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import Transcription

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


class ParaformerSTTHandler(BaseHandler[STTIn, STTOut]):
    """
    Handles the Speech To Text generation using a Paraformer model.
    The default for this model is set to Chinese.
    This model was contributed by @wuhongsheng.
    """

    def setup(
        self,
        model_name: str = "paraformer-zh",
        device: str = "cuda",
        gen_kwargs: dict[str, Any] = {},
    ) -> None:
        print(model_name)
        if len(model_name.split("/")) > 1:
            model_name = model_name.split("/")[-1]
        self.device = device
        self.model = AutoModel(model=model_name, device=device)
        self.warmup()

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}")

        # 2 warmup steps for no compile or compile mode with CUDA graphs capture
        n_steps = 1
        dummy_input = np.array([0] * 512, dtype=np.float32)
        for _ in range(n_steps):
            _ = self.model.generate(dummy_input)[0]["text"].strip().replace(" ", "")

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        logger.debug("infering paraformer...")

        pred_text = self.model.generate(vad_audio.audio)[0]["text"].strip().replace(" ", "")
        torch.mps.empty_cache()

        logger.debug("finished paraformer inference")
        console.print(f"[yellow]USER: {pred_text}")

        yield Transcription(text=pred_text)
