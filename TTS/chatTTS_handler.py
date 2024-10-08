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
        device="cuda",
        gen_kwargs={},  # Unused
        stream=True,
        chunk_size=512,
    ):
        self.device = device
        self.model = ChatTTS.Chat()
        self.model.load(compile=False)  # Doesn't work for me with True
        self.chunk_size = chunk_size
        self.stream = stream
        rnd_spk_emb = self.model.sample_random_speaker()
        self.params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=rnd_spk_emb,
        )
        self.output_sampling_rate = 16000
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        _ = self.model.infer("text")

    def process(self, llm_sentence):
        if isinstance(llm_sentence, tuple):
            llm_sentence, _ = llm_sentence # Ignore language
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
                    return {
                        "text": llm_sentence,
                        "sentence_end": True
                    }
                
                # Resample the audio to 16000 Hz
                audio_chunk = librosa.resample(gen[0], orig_sr=24000, target_sr=self.output_sampling_rate)
                # Ensure the audio is converted to mono (single channel)
                if len(audio_chunk.shape) > 1:
                    audio_chunk = librosa.to_mono(audio_chunk)
                audio_chunk = (audio_chunk * 32768).astype(np.int16)
                                
                # Loop through audio chunks, yielding dict for each chunk
                for i in range(0, len(audio_chunk), self.chunk_size):
                    chunk_data = {
                        "audio": {
                            "waveform": np.pad(
                                audio_chunk[i : i + self.chunk_size],
                                (0, self.chunk_size - len(audio_chunk[i : i + self.chunk_size])),
                            ),
                            "sampling_rate": self.output_sampling_rate,
                        }
                    }
                    # Include text for the first chunk
                    if i == 0:
                        chunk_data["text"] = llm_sentence  # Assuming llm_sentence is defined elsewhere
                    if i >= len(audio_chunk) - self.chunk_size:
                        # This is the last round
                        chunk_data["sentence_end"] = True
                    yield chunk_data
        else:
            wavs = wavs_gen
            if len(wavs[0]) == 0:
                return {
                    "sentence_end": True
                }
            audio_chunk = librosa.resample(wavs[0], orig_sr=24000, target_sr=self.output_sampling_rate)
            # Ensure the audio is converted to mono (single channel)
            if len(audio_chunk.shape) > 1:
                audio_chunk = librosa.to_mono(audio_chunk)
            audio_chunk = (audio_chunk * 32768).astype(np.int16)

            for i in range(0, len(audio_chunk), self.chunk_size):
                chunk_data = {
                    "audio": {
                        "waveform": np.pad(
                            audio_chunk[i : i + self.chunk_size],
                            (0, self.chunk_size - len(audio_chunk[i : i + self.chunk_size])),
                        ),
                        "sampling_rate": self.output_sampling_rate,
                    }
                }
                # For the first chunk, include text
                if i == 0:
                    chunk_data["text"] = llm_sentence
                if i >= len(audio_chunk) - self.chunk_size:
                    # This is the last round
                    chunk_data["sentence_end"] = True
                yield chunk_data
