import logging
import numpy as np
import scipy.signal

from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.messages import EndOfResponse, AUDIO_RESPONSE_DONE

logger = logging.getLogger(__name__)
console = Console()


class SupertonicTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        supertonic_voice="M1",
        supertonic_lang="na",
        supertonic_speed=1.0,
        **kwargs,
    ):
        self.should_listen = should_listen
        self.supertonic_voice = supertonic_voice
        self.supertonic_lang = supertonic_lang
        self.supertonic_speed = supertonic_speed

        try:
            from supertonic import TTS
        except ImportError:
            logger.error(
                "Supertonic package is not installed. Please install it using `pip install supertonic` or `pip install speech-to-speech[supertonic]`"
            )
            raise

        self.tts = TTS(auto_download=True)
        self.voice_style = self.tts.get_voice_style(voice_name=self.supertonic_voice)
        logger.info(f"Loaded Supertonic TTS with voice '{self.supertonic_voice}'")
        self.warmup()

    def warmup(self):
        logger.info("Warming up Supertonic TTS...")
        _ = self.tts.synthesize(
            text="Warmup",
            lang=self.supertonic_lang,
            voice_style=self.voice_style,
            speed=self.supertonic_speed,
        )

    def process(self, tts_input):
        if isinstance(tts_input, EndOfResponse):
            yield AUDIO_RESPONSE_DONE
            return

        text = tts_input.text

        if len(text.strip()) == 0:
            yield np.zeros(0, dtype=np.int16)
            return

        console.print(f"[green]ASSISTANT: {text}")

        # Supertonic returns (1, num_samples) shaped array at 44.1kHz float32
        wav, duration = self.tts.synthesize(
            text=text,
            lang=self.supertonic_lang,
            voice_style=self.voice_style,
            speed=self.supertonic_speed,
        )
        
        # Squeeze down to 1D
        audio_44k = wav.squeeze()

        # Resample from 44100 to 16000
        audio_16k_float = scipy.signal.resample_poly(audio_44k, 160, 441)

        # Convert to int16 format expected by the audio pipeline
        audio_int16 = (audio_16k_float * 32767).astype(np.int16)

        # Yield in chunks so the streamer can handle it smoothly
        chunk_size = 512
        for i in range(0, len(audio_int16), chunk_size):
            yield audio_int16[i : i + chunk_size]
