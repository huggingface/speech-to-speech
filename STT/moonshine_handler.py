import os
os.environ['KERAS_BACKEND'] = 'torch'

from time import perf_counter
import moonshine
import torch
from baseHandler import BaseHandler
from rich.console import Console
import logging

logger = logging.getLogger(__name__)
console = Console()


class MoonshineSTTHandler(BaseHandler):
    """
    Handles the Speech To Text generation using a Moonshine model.
    """

    def setup(
        self,
        model_name="moonshine/base",
        torch_dtype="float16",
        gen_kwargs={},
    ):
        self.torch_dtype = getattr(torch, torch_dtype)
        self.gen_kwargs = gen_kwargs

        self.tokenizer = moonshine.load_tokenizer()
        self.model = moonshine.load_model(model_name)

        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        n_steps = 2
        dummy_input = torch.randn(
            (1, 16000),
            dtype=self.torch_dtype,
        )

        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

        for _ in range(n_steps):
            _ = self.model.generate(dummy_input)

        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()

            logger.info(
                f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s"
            )

    def process(self, spoken_prompt):
        logger.debug("infering moonshine...")

        global pipeline_start
        pipeline_start = perf_counter()

        pred_ids = self.model.generate(spoken_prompt[None, :])
        pred_text = self.tokenizer.decode_batch(pred_ids)[0]

        logger.debug("finished whisper inference")
        console.print(f"[yellow]USER: {pred_text}")

        yield (pred_text, "en")
