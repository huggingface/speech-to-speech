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
    "hi",
    "de",
    "pt",
    "pl",
    "it",
    "nl",
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
        self.start_language = language
        self.last_language = language

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

        if self.start_language != 'auto':
            transcription_dict = self.model.transcribe(spoken_prompt, language=self.start_language)
        else:
            transcription_dict = self.model.transcribe(spoken_prompt)
            language_code = transcription_dict["language"]
            if language_code not in SUPPORTED_LANGUAGES:
                logger.warning(f"Whisper detected unsupported language: {language_code}")
                if self.last_language in SUPPORTED_LANGUAGES:  # reprocess with the last language
                    transcription_dict = self.model.transcribe(spoken_prompt, language=self.last_language)
                else:
                    transcription_dict = {"text": "", "language": "en"}
            else:
                self.last_language = language_code

        pred_text = transcription_dict["text"].strip()
        language_code = transcription_dict["language"]
        torch.mps.empty_cache()

        logger.debug("finished whisper inference")
        console.print(f"[yellow]USER: {pred_text}")
        logger.debug(f"Language Code Whisper: {language_code}")

        if self.start_language == "auto":
            language_code += "-auto"
                    
        yield (pred_text, language_code)
