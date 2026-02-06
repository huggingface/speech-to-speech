import logging
import numpy as np
from baseHandler import BaseHandler
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

# Language code mapping from Whisper/langdetect language codes to Kokoro lang codes
WHISPER_LANGUAGE_TO_KOKORO_LANG = {
    "en": "b",  # British English
    "ja": "j",  # Japanese
    "zh": "z",  # Chinese
    "fr": "f",  # French
    "es": "e",  # Spanish
    "it": "i",  # Italian
    "pt": "p",  # Portuguese
    "hi": "h",  # Hindi
    # Additional European languages from Parakeet - map to closest Kokoro language
    "de": "b",  # German -> British English (no German voice available)
    "nl": "b",  # Dutch -> British English
    "pl": "b",  # Polish -> British English
    "ru": "b",  # Russian -> British English
    "uk": "b",  # Ukrainian -> British English
}

# Default voices for each Kokoro language code
# These are native voices that sound natural for each language
# Voice naming: first letter = language, second letter = gender (f=female, m=male)
# Available voices per language:
#   a (American): af_alloy, af_aoede, af_bella, af_heart, af_jessica, af_kore, af_nicole, af_nova, af_river, af_sarah, af_sky
#                 am_adam, am_echo, am_eric, am_fenrir, am_liam, am_michael, am_onyx, am_puck, am_santa
#   b (British):  bf_alice, bf_emma, bf_isabella, bf_lily, bm_daniel, bm_fable, bm_george, bm_lewis
#   e (Spanish):  ef_dora, em_alex, em_santa
#   f (French):   ff_siwis
#   h (Hindi):    hf_alpha, hf_beta, hm_omega, hm_psi
#   i (Italian):  if_sara, im_nicola
#   j (Japanese): jf_alpha, jf_gongitsune, jf_nezumi, jf_tebukuro, jm_kumo
#   p (Portuguese): pf_dora, pm_alex, pm_santa
#   z (Chinese):  zf_xiaobei, zf_xiaoni, zf_xiaoxiao, zf_xiaoyi, zm_yunjian, zm_yunxi, zm_yunxia, zm_yunyang
KOKORO_LANG_DEFAULT_VOICES = {
    "a": "af_heart",      # American English female
    "b": "bm_fable",      # British English male
    "e": "ef_dora",       # Spanish female
    "f": "ff_siwis",      # French female
    "h": "hf_alpha",      # Hindi female
    "i": "if_sara",       # Italian female
    "j": "jf_alpha",      # Japanese female
    "p": "pf_dora",       # Portuguese female
    "z": "zf_xiaobei",    # Chinese female
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

        # Preload voices for common languages to avoid download delays during inference
        self._preload_multilingual_voices()

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

    def _preload_multilingual_voices(self):
        """Preload voices for common languages to avoid download delays during inference."""
        # Only preload a few commonly used language voices to avoid excessive startup time
        # Users speaking other languages will experience a one-time download delay
        preload_langs = ["e", "f", "i", "p"]  # Spanish, French, Italian, Portuguese

        for lang_code in preload_langs:
            voice = KOKORO_LANG_DEFAULT_VOICES.get(lang_code)
            if voice:
                try:
                    logger.info(f"Preloading voice for language '{lang_code}': {voice}")
                    pipeline = self.model._get_pipeline(lang_code)
                    pipeline.load_voice(voice)
                except Exception as e:
                    logger.warning(f"Failed to preload voice {voice}: {e}")

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
            # If language changed, get new pipeline and switch to appropriate voice
            if new_lang_code != self.lang_code:
                # Get the default voice for this language
                new_voice = KOKORO_LANG_DEFAULT_VOICES.get(new_lang_code, self.voice)
                logger.info(f"Language change detected: {self.lang_code} -> {new_lang_code}, voice: {self.voice} -> {new_voice}")
                try:
                    new_pipeline = self.model._get_pipeline(new_lang_code)
                    new_voice_tensor = new_pipeline.load_voice(new_voice)
                    # Only update state after successful loading
                    self.lang_code = new_lang_code
                    self.voice = new_voice
                    self._pipeline = new_pipeline
                    self._voice_tensor = new_voice_tensor
                except Exception as e:
                    logger.warning(f"Failed to switch language/voice: {e}. Keeping current language: {self.lang_code}")
                    # Continue with existing pipeline and voice

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
