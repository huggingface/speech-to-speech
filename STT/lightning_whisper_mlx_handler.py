import logging
from time import perf_counter
from baseHandler import BaseHandler
from lightning_whisper_mlx import LightningWhisperMLX
import numpy as np
from rich.console import Console

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


class LightningWhisperSTTHandler(BaseHandler):
    """
    Handles the Speech To Text generation using a Whisper model.
    """

    def setup(
        self,
        model_name="distil-whisper/distil-large-v3",
        device="cuda",
        torch_dtype="float16",
        compile_mode=None,
        gen_kwargs={},
    ):
        self.device = device
        self.model = LightningWhisperMLX(
            model="distil-medium.en", batch_size=12, quant=None
        )
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

        pred_text = self.model.transcribe(spoken_prompt)["text"].strip()

        logger.debug("finished whisper inference")
        console.print(f"[yellow]USER: {pred_text}")

        yield pred_text
