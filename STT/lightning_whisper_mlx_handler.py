import logging
from time import perf_counter
from baseHandler import BaseHandler
from lightning_whisper_mlx import LightningWhisperMLX
import numpy as np
from rich.console import Console
from copy import copy
import torch

logger = logging.getLogger(__name__)

console = Console()

SUPPORTED_LANGUAGES = [
    "en",
    "fr",
    "es",
    "zh",
    "ja",
    "ko",
]


class LightningWhisperSTTHandler(BaseHandler):
    """
    Handles the Speech To Text generation using a Whisper model.
    """

    def setup(
        self,
        model_name="distil-large-v3",
        device="mps",
        torch_dtype="float16",
        compile_mode=None,
        language=None,
        gen_kwargs={},
    ):
        if len(model_name.split("/")) > 1:
            model_name = model_name.split("/")[-1]
        self.device = device
        self.model = LightningWhisperMLX(model=model_name, batch_size=6, quant=None)
        if language == 'auto':
            language = None
        self.last_language = language
        if self.last_language is not None:
            self.gen_kwargs["language"] = self.last_language

        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        # 2 warmup steps for no compile or compile mode with CUDA graphs capture
        n_steps = 1
        dummy_input = np.array([0] * 512)

        for _ in range(n_steps):
            _ = self.model.transcribe(dummy_input)["text"].strip()

    def process(self, spoken_prompt):
        logger.debug("infering whisper...")

        global pipeline_start
        pipeline_start = perf_counter()

        # language_code = self.processor.tokenizer.decode(pred_ids[0, 1])[2:-2]  # remove "<|" and "|>"

        language_code = self.model.transcribe(spoken_prompt)["language"]

        if language_code not in SUPPORTED_LANGUAGES:  # reprocess with the last language
            logger.warning("Whisper detected unsupported language:", language_code)
            gen_kwargs = copy(self.gen_kwargs) #####
            gen_kwargs['language'] = self.last_language
            language_code = self.last_language
            # pred_ids = self.model.generate(input_features, **gen_kwargs)
        else:
            self.last_language = language_code

        pred_text = self.model.transcribe(spoken_prompt)["text"].strip()
        torch.mps.empty_cache()

        # language_code = self.processor.tokenizer.decode(pred_ids[0, 1])[2:-2] # remove "<|" and "|>"
        language_code = self.model.transcribe(spoken_prompt)["language"]

        logger.debug("finished whisper inference")
        console.print(f"[yellow]USER: {pred_text}")
        logger.debug(f"Language Code Whisper: {language_code}")

        yield (pred_text, language_code)
