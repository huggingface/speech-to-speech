import logging
from time import perf_counter

from baseHandler import BaseHandler
from funasr import AutoModel
import numpy as np
from rich.console import Console
import torch

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


class ParaformerSTTHandler(BaseHandler):
    """
    Handles the Speech To Text generation using a Paraformer model.
    The default for this model is set to Chinese.
    This model was contributed by @wuhongsheng.
    """

    def setup(
        self,
        model_name="paraformer-zh",
        device="cuda",
        gen_kwargs={},
    ):
        print(model_name)
        if len(model_name.split("/")) > 1:
            model_name = model_name.split("/")[-1]
        self.device = device
        self.model = AutoModel(model=model_name, device=device)
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        # 2 warmup steps for no compile or compile mode with CUDA graphs capture
        n_steps = 1
        dummy_input = np.array([0] * 512, dtype=np.float32)
        for _ in range(n_steps):
            _ = self.model.generate(dummy_input)[0]["text"].strip().replace(" ", "")

    def process(self, spoken_prompt):
        logger.debug("infering paraformer...")

        global pipeline_start
        pipeline_start = perf_counter()

        pred_text = (
            self.model.generate(spoken_prompt)[0]["text"].strip().replace(" ", "")
        )
        torch.mps.empty_cache()

        logger.debug("finished paraformer inference")
        console.print(f"[yellow]USER: {pred_text}")

        yield pred_text
