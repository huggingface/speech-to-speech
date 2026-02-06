import logging
import numpy as np
from baseHandler import BaseHandler
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

# Language code mapping from Whisper language codes to Kokoro lang codes
WHISPER_LANGUAGE_TO_KOKORO_LANG = {
    "en": "b",  # British English (to match bm_fable voice)
    "ja": "j",  # Japanese
    "zh": "z",  # Chinese
    "fr": "f",  # French
    "es": "e",  # Spanish
    "it": "i",  # Italian
    "pt": "p",  # Portuguese
    "hi": "h",  # Hindi
}


class KokoroTTSHandler(BaseHandler):
    """
    Kokoro TTS handler for CUDA/CPU devices.
    Uses the native kokoro library for inference.
    """

    def setup(
        self,
        should_listen,
        model_name="hexgrad/Kokoro-82M",
        device="cuda",
        voice="bm_fable",
        lang_code="b",
        speed=1.0,
        blocksize=512,
        gen_kwargs=None,  # Unused, but passed by the pipeline
    ):
        self.should_listen = should_listen
        self.model_name = model_name
        self.device = device
        self.voice = voice
        self.lang_code = lang_code
        self.speed = speed
        self.blocksize = blocksize

        # Import kokoro library
        try:
            from kokoro import KPipeline
        except ImportError:
            raise ImportError(
                "kokoro is required for Kokoro TTS. Install with: pip install kokoro>=0.9.2 soundfile\n"
                "Also ensure espeak-ng is installed: apt-get install espeak-ng (Linux) or brew install espeak-ng (macOS)"
            )

        logger.info(f"Loading Kokoro model with lang_code: {lang_code}")
        self.pipeline = KPipeline(lang_code=lang_code)
        logger.info(f"Kokoro pipeline loaded successfully")

        self.warmup()

    def warmup(self):
        """Warm up the model with a dummy inference."""
        logger.info(f"Warming up {self.__class__.__name__}")

        # Run a short dummy inference to warm up the model
        for _ in self.pipeline("Hello", voice=self.voice, speed=self.speed):
            pass

        logger.info(f"{self.__class__.__name__} warmed up")

    def process(self, llm_sentence):
        """
        Process text input and generate audio output.

        Args:
            llm_sentence: Either a string or tuple of (text, language_code)

        Yields:
            Audio chunks as numpy int16 arrays
        """
        from scipy.signal import resample_poly

        language_code = None
        if isinstance(llm_sentence, tuple):
            llm_sentence, language_code = llm_sentence
            # Map Whisper language code to Kokoro language code
            new_lang_code = WHISPER_LANGUAGE_TO_KOKORO_LANG.get(
                language_code, self.lang_code
            )
            # Reinitialize pipeline if language changed
            if new_lang_code != self.lang_code:
                self.lang_code = new_lang_code
                from kokoro import KPipeline

                self.pipeline = KPipeline(lang_code=self.lang_code)

        console.print(f"[green]ASSISTANT: {llm_sentence}")

        # Generate audio using Kokoro
        # The pipeline yields tuples of (graphemes, phonemes, audio)
        for gs, ps, audio in self.pipeline(
            llm_sentence, voice=self.voice, speed=self.speed
        ):
            if audio is None:
                continue

            # Ensure audio is numpy array
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio, dtype=np.float32)
            else:
                audio = audio.astype(np.float32)

            # Kokoro outputs at 24kHz, resample to 16kHz for the pipeline
            # Using scipy's polyphase resampling (fast and high quality)
            # 16000/24000 = 2/3, so up=2, down=3
            audio = resample_poly(audio, up=2, down=3)

            # Convert to int16 format
            audio = (audio * 32768).astype(np.int16)

            # Yield audio in fixed-size chunks
            for i in range(0, len(audio), self.blocksize):
                chunk = audio[i : i + self.blocksize]
                # Pad the last chunk if necessary
                if len(chunk) < self.blocksize:
                    chunk = np.pad(chunk, (0, self.blocksize - len(chunk)))
                yield chunk

        self.should_listen.set()
