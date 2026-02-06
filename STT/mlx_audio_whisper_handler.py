import logging
from time import perf_counter
from baseHandler import BaseHandler
import numpy as np
from rich.console import Console
import mlx.core as mx

logger = logging.getLogger(__name__)

console = Console()

SUPPORTED_LANGUAGES = [
    "en",
    "fr",
    "es",
    "zh",
    "ja",
    "ko",
    "hi",
    "de",
    "pt",
    "pl",
    "it",
    "nl",
]


class MLXAudioWhisperSTTHandler(BaseHandler):
    """
    Handles the Speech To Text generation using MLX Audio's Whisper implementation.
    Optimized for Apple Silicon using the MLX framework.
    """

    def setup(
        self,
        model_name="mlx-community/whisper-large-v3-turbo",
        language=None,
        gen_kwargs={},
    ):
        from mlx_audio.stt.generate import load_model
        from transformers import WhisperProcessor

        self.model_name = model_name
        self.start_language = language
        self.last_language = language
        self.gen_kwargs = gen_kwargs

        # Load the model directly
        logger.info(f"Loading model {model_name}...")
        self.model = load_model(model_name)

        # Check if processor was loaded, if not, load it manually from original model
        if self.model._processor is None:
            logger.info("Processor not found in MLX model, loading from original Whisper model...")
            # Map MLX model names to their original Whisper counterparts
            processor_model_map = {
                "mlx-community/whisper-large-v3-turbo": "openai/whisper-large-v3",
                "mlx-community/whisper-large-v3": "openai/whisper-large-v3",
                "mlx-community/whisper-medium": "openai/whisper-medium",
                "mlx-community/whisper-small": "openai/whisper-small",
                "mlx-community/whisper-base": "openai/whisper-base",
                "mlx-community/whisper-tiny": "openai/whisper-tiny",
            }

            # Get the appropriate processor model name
            processor_model = processor_model_map.get(model_name, "openai/whisper-large-v3")
            logger.info(f"Loading processor from {processor_model}...")

            try:
                self.model._processor = WhisperProcessor.from_pretrained(processor_model)
                logger.info("Processor loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load processor: {e}")
                raise

        logger.info(f"Model {model_name} loaded successfully")

        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        # Warmup with a dummy input
        dummy_audio = np.zeros(16000, dtype=np.float32)

        try:
            # Pre-warm the model by running a transcription
            _ = self.model.generate(
                dummy_audio,
                verbose=False
            )
            logger.info(f"Model warmed up and ready")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def process(self, spoken_prompt):
        logger.debug("inferring mlx-audio whisper...")

        global pipeline_start
        pipeline_start = perf_counter()

        # Convert to numpy array if needed
        if isinstance(spoken_prompt, mx.array):
            audio_input = np.array(spoken_prompt)
        elif not isinstance(spoken_prompt, np.ndarray):
            audio_input = np.array(spoken_prompt, dtype=np.float32)
        else:
            audio_input = spoken_prompt.astype(np.float32)

        # Prepare generation kwargs - only pass valid parameters
        gen_kwargs = {}

        # Add language if specified
        if self.start_language and self.start_language != 'auto':
            gen_kwargs['language'] = self.start_language

        try:
            # Generate transcription directly using model.generate
            result = self.model.generate(
                audio_input,
                verbose=False,
                **gen_kwargs
            )

            # Extract text from result
            pred_text = result.text.strip() if hasattr(result, 'text') else str(result).strip()

            # Try to detect language from result if available
            if hasattr(result, 'language'):
                language_code = result.language
            elif self.start_language and self.start_language != 'auto':
                language_code = self.start_language
            else:
                # Default to last known language or English
                language_code = self.last_language if self.last_language else "en"

            # Validate language code
            if language_code not in SUPPORTED_LANGUAGES:
                logger.warning(f"Detected unsupported language: {language_code}")
                if self.last_language in SUPPORTED_LANGUAGES:
                    language_code = self.last_language
                else:
                    language_code = "en"
            else:
                self.last_language = language_code

        except Exception as e:
            logger.error(f"MLX Audio Whisper inference failed: {e}")
            pred_text = ""
            language_code = self.last_language if self.last_language else "en"

        logger.debug("finished mlx-audio whisper inference")
        console.print(f"[yellow]USER: {pred_text}")
        logger.debug(f"Language Code: {language_code}")

        if self.start_language == "auto":
            language_code += "-auto"

        yield (pred_text, language_code)
