from melo.api import TTS
import logging
from baseHandler import BaseHandler
import librosa
import numpy as np
from rich.console import Console
import torch

logger = logging.getLogger(__name__)

console = Console()


class MeloTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        device="mps",
        language="EN_NEWEST",
        speaker_to_id="EN-Newest",
        gen_kwargs={},  # Unused
        blocksize=512,
    ):
        self.should_listen = should_listen
        self.device = device
        self.model = TTS(language=language, device=device)
        self.speaker_id = self.model.hps.data.spk2id[speaker_to_id]
        self.blocksize = blocksize
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        _ = self.model.tts_to_file("text", self.speaker_id, quiet=True)

    def process(self, llm_sentence):
        console.print(f"[green]ASSISTANT: {llm_sentence}")
        if self.device == "mps":
            import time

            start = time.time()
            torch.mps.synchronize()  # Waits for all kernels in all streams on the MPS device to complete.
            torch.mps.empty_cache()  # Frees all memory allocated by the MPS device.
            _ = (
                time.time() - start
            )  # Removing this line makes it fail more often. I'm looking into it.

        audio_chunk = self.model.tts_to_file(llm_sentence, self.speaker_id, quiet=True)
        if len(audio_chunk) == 0:
            self.should_listen.set()
            return
        audio_chunk = librosa.resample(audio_chunk, orig_sr=44100, target_sr=16000)
        audio_chunk = (audio_chunk * 32768).astype(np.int16)
        for i in range(0, len(audio_chunk), self.blocksize):
            yield np.pad(
                audio_chunk[i : i + self.blocksize],
                (0, self.blocksize - len(audio_chunk[i : i + self.blocksize])),
            )

        self.should_listen.set()
