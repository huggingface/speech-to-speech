import ChatTTS
import logging
from baseHandler import BaseHandler
import librosa
import numpy as np
from rich.console import Console
import torch
from .STV.speech_to_visemes import SpeechToVisemes

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


class ChatTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        device="cuda",
        gen_kwargs={},  # Unused
        stream=True,
        chunk_size=512,
        viseme_flag = True
    ):
        self.should_listen = should_listen
        self.device = device
        self.model = ChatTTS.Chat()
        self.model.load(compile=False)  # Doesn't work for me with True
        self.chunk_size = chunk_size
        self.stream = stream
        rnd_spk_emb = self.model.sample_random_speaker()
        self.params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=rnd_spk_emb,
        )
        self.viseme_flag = viseme_flag
        if self.viseme_flag:
            self.speech_to_visemes = SpeechToVisemes()
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        _ = self.model.infer("text")

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

        wavs_gen = self.model.infer(
            llm_sentence, params_infer_code=self.params_infer_code, stream=self.stream
        )

        if self.stream:
            wavs = [np.array([])]
            for gen in wavs_gen:
                if gen[0] is None or len(gen[0]) == 0:
                    self.should_listen.set()
                    return
                
                # Resample the audio to 16000 Hz
                audio_chunk = librosa.resample(gen[0], orig_sr=24000, target_sr=16000)
                # Ensure the audio is converted to mono (single channel)
                if len(audio_chunk.shape) > 1:
                    audio_chunk = librosa.to_mono(audio_chunk)
                audio_chunk = (audio_chunk * 32768).astype(np.int16)
                
                # Process visemes if viseme_flag is set
                if self.viseme_flag:
                    visemes = self.speech_to_visemes.process(audio_chunk)
                    for viseme in visemes:
                        console.print(f"[blue]ASSISTANT_MOUTH_SHAPE: {viseme['viseme']} -- {viseme['timestamp']}")
                else:
                    visemes = None
                
                # Loop through audio chunks, yielding dict for each chunk
                for i in range(0, len(audio_chunk), self.chunk_size):
                    chunk_data = {
                        "audio": np.pad(
                            audio_chunk[i : i + self.chunk_size],
                            (0, self.chunk_size - len(audio_chunk[i : i + self.chunk_size])),
                        )
                    }
                    # Include text and visemes for the first chunk
                    if i == 0:
                        chunk_data["text"] = llm_sentence  # Assuming llm_sentence is defined elsewhere
                        chunk_data["visemes"] = visemes
                
                    yield chunk_data
        else:
            wavs = wavs_gen
            if len(wavs[0]) == 0:
                self.should_listen.set()
                return
            audio_chunk = librosa.resample(wavs[0], orig_sr=24000, target_sr=16000)
            # Ensure the audio is converted to mono (single channel)
            if len(audio_chunk.shape) > 1:
                audio_chunk = librosa.to_mono(audio_chunk)
            audio_chunk = (audio_chunk * 32768).astype(np.int16)

            if self.viseme_flag:
                visemes = self.speech_to_visemes.process(audio_chunk)
                for viseme in visemes:
                    console.print(f"[blue]ASSISTANT_MOUTH_SHAPE: {viseme['viseme']} -- {viseme['timestamp']}")
            else:
                visemes = None

            for i in range(0, len(audio_chunk), self.chunk_size):
                chunk_data = {
                    "audio": np.pad(
                        audio_chunk[i : i + self.chunk_size],
                        (0, self.chunk_size - len(audio_chunk[i : i + self.chunk_size])),
                    )
                }
                # For the first chunk, include text and visemes
                if i == 0:
                    chunk_data["text"] = llm_sentence
                    chunk_data["visemes"] = visemes            
                yield chunk_data

        self.should_listen.set()
