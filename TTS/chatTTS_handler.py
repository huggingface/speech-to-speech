import ChatTTS
import logging
from baseHandler import BaseHandler
import librosa
import numpy as np
from rich.console import Console
import torch

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


class ChatTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        device="mps",
        gen_kwargs={},  # Unused
        stream=True,
        chunk_size=512,
    ):
        self.should_listen = should_listen
        self.device = device
        self.model  = ChatTTS.Chat()
        self.model.load(compile=True) # Set to True for better performance
        self.chunk_size = chunk_size
        self.stream = stream
        rnd_spk_emb = self.model.sample_random_speaker()
        self.params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=rnd_spk_emb,
        )
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        _= self.model.infer("text")


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

        wavs_gen = self.model.infer(llm_sentence,params_infer_code=self.params_infer_code, stream=self.stream)

        if self.stream:
            wavs = [np.array([])]
            for gen in wavs_gen:
                print('new chunk gen', len(gen[0]))
                if len(gen[0]) == 0:
                    self.should_listen.set()
                    return
                audio_chunk = librosa.resample(gen[0], orig_sr=24000, target_sr=16000)
                audio_chunk = (audio_chunk * 32768).astype(np.int16)
                print('audio_chunk:', audio_chunk.shape)
                while len(audio_chunk) > self.chunk_size:
                    yield audio_chunk[:self.chunk_size]  # 返回前 chunk_size 字节的数据
                    audio_chunk = audio_chunk[self.chunk_size:]  # 移除已返回的数据
                yield np.pad(audio_chunk, (0,self.chunk_size-len(audio_chunk)))
        else:
            print('check result', wavs_gen)
            wavs = wavs_gen
            if len(wavs[0]) == 0:
                self.should_listen.set()
                return
            audio_chunk = librosa.resample(wavs[0], orig_sr=24000, target_sr=16000)
            audio_chunk = (audio_chunk * 32768).astype(np.int16)
            print('audio_chunk:', audio_chunk.shape)
            for i in range(0, len(audio_chunk), self.chunk_size):
                yield np.pad(
                    audio_chunk[i : i + self.chunk_size],
                    (0, self.chunk_size - len(audio_chunk[i : i + self.chunk_size])),
                )
        self.should_listen.set()


