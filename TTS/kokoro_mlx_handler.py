import logging
import numpy as np
from baseHandler import BaseHandler
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

# Language code mapping from Whisper language codes to Kokoro lang codes
WHISPER_LANGUAGE_TO_KOKORO_LANG = {
    "en": "b",  # British English
    "ja": "j",  # Japanese
    "zh": "z",  # Chinese
    "fr": "f",  # French
    "es": "e",  # Spanish (placeholder - may need adjustment)
    "it": "i",  # Italian (placeholder - may need adjustment)
    "pt": "p",  # Portuguese (placeholder - may need adjustment)
    "hi": "h",  # Hindi (placeholder - may need adjustment)
}


class KokoroMLXTTSHandler(BaseHandler):
    """
    Kokoro TTS handler using MLX for Apple Silicon devices.
    Uses the mlx-audio library for efficient inference on M-series chips.
    """

    def setup(
        self,
        should_listen,
        model_name="mlx-community/Kokoro-82M-bf16",
        voice="bm_fable",
        lang_code="b",
        speed=1.0,
        blocksize=512,
        gen_kwargs=None,  # Unused, but passed by the pipeline
    ):
        self.should_listen = should_listen
        self.model_name = model_name
        self.voice = voice
        self.lang_code = lang_code
        self.speed = speed
        self.blocksize = blocksize

        # Import mlx-audio TTS utilities
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError:
            raise ImportError(
                "mlx-audio is required for Kokoro MLX TTS. Install with: pip install mlx-audio"
            )

        logger.info(f"Loading Kokoro MLX model: {model_name}")
        self.model = load_model(model_name)
        logger.info(f"Kokoro MLX model loaded successfully")

        # Get or create the pipeline for our language and preload the voice
        # This avoids the voice being reloaded on every generate() call
        self._pipeline = self.model._get_pipeline(lang_code)
        self._voice_tensor = self._pipeline.load_voice(voice)
        logger.info(f"Preloaded voice: {voice}")

        self.warmup()

    def warmup(self):
        """Warm up the model with a dummy inference."""
        logger.info(f"Warming up {self.__class__.__name__}")

        # Run a short dummy inference to warm up the model
        for _ in self.model.generate(
            text="Hello",
            voice=self.voice,
            speed=self.speed,
            lang_code=self.lang_code,
        ):
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
            # If language changed, get new pipeline and reload voice
            if new_lang_code != self.lang_code:
                self.lang_code = new_lang_code
                self._pipeline = self.model._get_pipeline(self.lang_code)
                self._voice_tensor = self._pipeline.load_voice(self.voice)

        console.print(f"[green]ASSISTANT: {llm_sentence}")

        # Generate audio using the preloaded pipeline directly
        # This avoids the voice reload that happens in model.generate()
        for result in self._pipeline(
            text=llm_sentence,
            voice=self.voice,
            speed=self.speed,
        ):
            if result.audio is None:
                continue
            # result.audio is an mx.array with shape (1, samples), convert to numpy and squeeze
            audio = np.array(result.audio, dtype=np.float32).squeeze(0)

            # Trim silence from start and end of audio
            # Kokoro generates ~250ms of silence at the start and variable silence at the end
            threshold = 0.01
            abs_audio = np.abs(audio)
            # Find first and last samples above threshold
            above_threshold = abs_audio > threshold
            if np.any(above_threshold):
                start_idx = np.argmax(above_threshold)
                end_idx = len(audio) - np.argmax(above_threshold[::-1])
                # Add small padding (5ms = 120 samples at 24kHz) to avoid cutting speech
                padding = 120
                start_idx = max(0, start_idx - padding)
                end_idx = min(len(audio), end_idx + padding)
                audio = audio[start_idx:end_idx]

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
