import numpy as np
import librosa
import logging
from baseHandler import BaseHandler

logger = logging.getLogger(__name__)

class ResampleHandler(BaseHandler):
    """Resamples incoming audio to a fixed sample rate."""

    def setup(self, input_rate: int = 16000, output_rate: int = 16000):
        self.input_rate = input_rate
        self.output_rate = output_rate

    def process(self, audio_chunk: bytes):
        audio = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        if self.input_rate != self.output_rate:
            audio = librosa.resample(audio, orig_sr=self.input_rate, target_sr=self.output_rate)
        audio = (audio * 32768.0).astype(np.int16)
        yield audio.tobytes()

