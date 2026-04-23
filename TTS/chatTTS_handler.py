from __future__ import annotations

import ChatTTS
import logging
from baseHandler import BaseHandler
from cancel_scope import CancelScope
from pipeline_messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
import librosa
import numpy as np
from rich.console import Console
import torch

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

console = Console()


class ChatTTSHandler(BaseHandler[TTSInput | EndOfResponse]):
    def setup(
        self,
        should_listen,
        device="cuda",
        gen_kwargs={},  # Unused
        stream=True,
        chunk_size=512,
        cancel_scope: CancelScope | None = None,
    ):
        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.device = device
        self.model = ChatTTS.Chat()
        self.model.load(compile=False)  # Doesn't work for me with True
        self.chunk_size = chunk_size
        self.stream = stream
        rnd_spk_emb = self.model.sample_random_speaker()
        self.params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=rnd_spk_emb,
        )
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        _ = self.model.infer("text")

    def process(self, tts_input: TTSInput | EndOfResponse):
        if isinstance(tts_input, EndOfResponse):
            yield AUDIO_RESPONSE_DONE
            return

        runtime_config = tts_input.runtime_config
        text = tts_input.text

        _cancel_gen = self.cancel_scope.generation if self.cancel_scope else None
        console.print(f"[green]ASSISTANT: {text}")
        if self.device == "mps":
            import time

            start = time.time()
            torch.mps.synchronize()  # Waits for all kernels in all streams on the MPS device to complete.
            torch.mps.empty_cache()  # Frees all memory allocated by the MPS device.
            _ = (
                time.time() - start
            )  # Removing this line makes it fail more often. I'm looking into it.

        wavs_gen = self.model.infer(
            text, params_infer_code=self.params_infer_code, stream=self.stream
        )

        if self.stream:
            wavs = [np.array([])]
            for gen in wavs_gen:
                if _cancel_gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(_cancel_gen):
                    logger.info("TTS generation cancelled (interruption)")
                    return
                if gen[0] is None or len(gen[0]) == 0:
                    if not runtime_config:
                        self.should_listen.set()
                    return
                audio_chunk = librosa.resample(gen[0], orig_sr=24000, target_sr=16000)
                audio_chunk = (audio_chunk * 32768).astype(np.int16)[0]
                while len(audio_chunk) > self.chunk_size:
                    yield audio_chunk[: self.chunk_size]  # Return the first chunk_size samples of the audio data
                    audio_chunk = audio_chunk[self.chunk_size :]  # Remove the samples that have already been returned
                yield np.pad(audio_chunk, (0, self.chunk_size - len(audio_chunk)))
        else:
            wavs = wavs_gen
            if len(wavs[0]) == 0:
                if not runtime_config:
                    self.should_listen.set()
                return
            audio_chunk = librosa.resample(wavs[0], orig_sr=24000, target_sr=16000)
            audio_chunk = (audio_chunk * 32768).astype(np.int16)
            for i in range(0, len(audio_chunk), self.chunk_size):
                yield np.pad(
                    audio_chunk[i : i + self.chunk_size],
                    (0, self.chunk_size - len(audio_chunk[i : i + self.chunk_size])),
                )
        if not runtime_config:
            self.should_listen.set()
