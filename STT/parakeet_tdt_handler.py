"""
Parakeet TDT Speech-to-Text Handler

Supports NVIDIA Parakeet TDT model for high-quality multilingual ASR.
- On Apple Silicon (MPS): Uses mlx-audio with mlx-community/parakeet-tdt-0.6b-v3
- On CUDA: Uses NVIDIA NeMo with nvidia/parakeet-tdt-0.6b-v3

Model supports 25 European languages with automatic language detection.
"""

import logging
from time import perf_counter
from sys import platform
from baseHandler import BaseHandler
import numpy as np
from rich.console import Console
from utils.mlx_lock import MLXLockContext

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

logger = logging.getLogger(__name__)
console = Console()

# Parakeet TDT v3 supports 25 European languages
SUPPORTED_LANGUAGES = [
    "en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "uk",
    "cs", "sk", "hu", "ro", "bg", "hr", "sl", "sr", "da", "no",
    "sv", "fi", "et", "lv", "lt"
]


class ParakeetTDTSTTHandler(BaseHandler):
    """
    Handles Speech-to-Text using NVIDIA Parakeet TDT model.

    On Apple Silicon (MPS): Uses mlx-audio with the MLX-converted model.
    On CUDA: Uses NVIDIA NeMo for optimal performance.

    Parakeet TDT 0.6B v3 is a 600M parameter multilingual ASR model
    supporting 25 European languages with automatic language detection.
    """

    def setup(
        self,
        model_name=None,
        device="auto",
        compute_type="float16",
        language=None,
        gen_kwargs={},
        enable_live_transcription=False,
        live_transcription_update_interval=0.25,
    ):
        """
        Initialize the Parakeet TDT model.

        Args:
            model_name: Model identifier. Defaults are:
                - MPS: "mlx-community/parakeet-tdt-0.6b-v3"
                - CUDA: "nvidia/parakeet-tdt-0.6b-v3"
            device: Device to use ("auto", "cuda", "mps", "cpu")
            compute_type: Compute precision ("float16", "float32")
            language: Target language code (optional, model auto-detects)
            gen_kwargs: Additional generation kwargs
        """
        self.gen_kwargs = gen_kwargs
        self.start_language = language
        self.last_language = language if language else "en"
        self.enable_live_transcription = enable_live_transcription
        self.live_transcription_update_interval = live_transcription_update_interval

        # Determine device
        if device == "auto":
            if platform == "darwin":
                self.device = "mps"
            else:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Set default model based on device
        if model_name is None:
            if self.device == "mps":
                model_name = "mlx-community/parakeet-tdt-0.6b-v3"
            else:
                model_name = "nvidia/parakeet-tdt-0.6b-v3"

        self.model_name = model_name
        self.compute_type = compute_type

        logger.info(f"Loading Parakeet TDT model: {model_name} on {self.device}")

        if self.device == "mps":
            self._setup_mlx(model_name)
        else:
            self._setup_nemo(model_name)

        # Setup streaming handler if live transcription is enabled
        if self.enable_live_transcription:
            if self.backend == "mlx":
                try:
                    from STT.smart_progressive_streaming import SmartProgressiveStreamingHandler
                    self.streaming_handler = SmartProgressiveStreamingHandler(
                        self.model,
                        emission_interval=self.live_transcription_update_interval,
                        max_window_size=15.0,
                        sentence_buffer=2.0,
                    )
                    self.processing_final = False  # Track if we're processing final audio
                    logger.info("Live transcription enabled for Parakeet TDT")
                except ImportError:
                    logger.warning("SmartProgressiveStreamingHandler not available, disabling live transcription")
                    self.enable_live_transcription = False
            else:
                logger.warning("Live transcription only supported with MLX backend, disabling")
                self.enable_live_transcription = False

        self.warmup()

    def _setup_mlx(self, model_name):
        """Setup for Apple Silicon using mlx-audio."""
        try:
            from mlx_audio.stt.generate import load_model

            self.backend = "mlx"
            self.model = load_model(model_name)
            logger.info(f"MLX Audio Parakeet model loaded successfully")
        except ImportError as e:
            raise ImportError(
                "mlx-audio is required for Parakeet TDT on Apple Silicon. "
                "Install with: pip install mlx-audio"
            ) from e

    def _setup_nemo(self, model_name):
        """Setup for CUDA using NVIDIA NeMo."""
        try:
            import nemo.collections.asr as nemo_asr
            import torch

            self.backend = "nemo"

            # Load model from HuggingFace or local path
            if model_name.endswith(".nemo"):
                self.model = nemo_asr.models.ASRModel.restore_from(restore_path=model_name)
            else:
                self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

            # Move to appropriate device
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()

            # Set eval mode
            self.model.eval()

            logger.info(f"NeMo Parakeet model loaded successfully on {self.device}")
        except ImportError as e:
            raise ImportError(
                "NVIDIA NeMo is required for Parakeet TDT on CUDA. "
                "Install with: pip install nemo_toolkit[asr]"
            ) from e

    def warmup(self):
        """Warm up the model with a dummy input."""
        logger.info(f"Warming up {self.__class__.__name__}")

        # Create 1 second of silence at 16kHz
        dummy_audio = np.zeros(16000, dtype=np.float32)

        try:
            if self.backend == "mlx":
                import mlx.core as mx
                # Convert to mx.array and call decode_chunk directly
                audio_mx = mx.array(dummy_audio, dtype=mx.float32)
                _ = self.model.decode_chunk(audio_mx, verbose=False)
            else:
                _ = self.model.transcribe([dummy_audio], batch_size=1, verbose=False)

            logger.info("Model warmed up and ready")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def process(self, spoken_prompt):
        """
        Process audio and generate transcription.

        Args:
            spoken_prompt: Audio data as numpy array (float32, 16kHz)
                         OR tuple of ("progressive"/"final", audio_array) for realtime mode

        Yields:
            Tuple of (transcription_text, language_code)
        """
        logger.debug("Inferring Parakeet TDT...")

        global pipeline_start
        pipeline_start = perf_counter()

        # Check if this is progressive audio from VAD
        is_progressive = False
        if isinstance(spoken_prompt, tuple) and len(spoken_prompt) == 2:
            mode, audio_input = spoken_prompt
            is_progressive = (mode == "progressive")
            is_final = (mode == "final")
        else:
            audio_input = spoken_prompt
            is_final = True

        # Ensure audio is float32 numpy array
        if not isinstance(audio_input, np.ndarray):
            audio_input = np.array(audio_input, dtype=np.float32)
        else:
            audio_input = audio_input.astype(np.float32)

        # Handle progressive updates (live transcription display only)
        if self.enable_live_transcription and self.backend == "mlx" and is_progressive:
            # Ignore progressive updates if we're already processing final audio
            if self.processing_final:
                logger.debug("Skipping stale progressive update (final audio already received)")
                return

            # Try to acquire MLX lock with short timeout - skip if busy (TTS might be using it)
            with MLXLockContext(handler_name="ParakeetSTT-Progressive", timeout=0.01) as acquired:
                if acquired:
                    try:
                        self._show_progressive_transcription(audio_input)
                    except Exception as e:
                        logger.debug(f"Progressive transcription failed: {e}")
                else:
                    logger.debug("Skipping progressive update (MLX busy)")
            return  # Don't yield to queue

        # Handle final transcription (send to LLM)
        try:
            if self.enable_live_transcription and self.backend == "mlx":
                # Mark that we're processing final audio (ignore stale progressive updates)
                self.processing_final = True

                # Acquire MLX lock with longer timeout for final transcription
                with MLXLockContext(handler_name="ParakeetSTT-Final", timeout=5.0) as acquired:
                    if not acquired:
                        logger.error("Failed to acquire MLX lock for final transcription")
                        pred_text = ""
                        language_code = self.last_language
                    else:
                        pred_text, language_code = self._process_mlx_final(audio_input)
            elif self.backend == "mlx":
                with MLXLockContext(handler_name="ParakeetSTT", timeout=5.0):
                    pred_text, language_code = self._process_mlx(audio_input)
            else:
                pred_text, language_code = self._process_nemo(audio_input)

            # Validate and update language
            if language_code and language_code in SUPPORTED_LANGUAGES:
                self.last_language = language_code
            else:
                language_code = self.last_language

        except Exception as e:
            logger.error(f"Parakeet TDT inference failed: {e}")
            pred_text = ""
            language_code = self.last_language

        logger.debug("Finished Parakeet TDT inference")
        console.print(f"[yellow]USER: {pred_text}")
        console.print(f"[dim]Language: {language_code}[/dim]")

        yield (pred_text, language_code)

        # Reset processing_final flag for next utterance
        if self.enable_live_transcription and self.backend == "mlx":
            self.processing_final = False

    def _detect_language_from_text(self, text):
        """
        Detect language from transcribed text using langdetect.

        Args:
            text: Transcribed text string

        Returns:
            Detected language code or None if detection fails
        """
        if not LANGDETECT_AVAILABLE:
            logger.warning("langdetect not available, cannot detect language from text")
            return None

        # Need at least some text for reliable detection
        if not text or len(text.strip()) < 10:
            return None

        try:
            detected = detect(text)
            # Map langdetect codes to our supported languages if needed
            # langdetect uses ISO 639-1 codes which match SUPPORTED_LANGUAGES
            if detected in SUPPORTED_LANGUAGES:
                return detected
            else:
                logger.debug(f"Detected language '{detected}' not in supported languages")
                return None
        except LangDetectException as e:
            logger.debug(f"Language detection failed: {e}")
            return None

    def _show_progressive_transcription(self, audio_input):
        """Show progressive transcription without yielding result."""
        from rich.text import Text

        try:
            # Use streaming handler for progressive transcription
            result = self.streaming_handler.transcribe_incremental(audio_input)

            # Display live transcription with colors (overwrite previous line)
            # Yellow = fixed user text (matches final USER output)
            # Cyan = active/in-progress text (indicates processing)
            text = Text()
            if result.fixed_text:
                text.append("Live: ", style="dim")
                text.append(result.fixed_text, style="yellow")
                if result.active_text:
                    text.append(" ", style="dim")

            if result.active_text:
                if not result.fixed_text:
                    text.append("Live: ", style="dim")
                text.append(result.active_text, style="cyan dim")

            if text:
                console.print(text, end="\r")
        except Exception as e:
            logger.debug(f"Progressive transcription failed: {e}")

    def _process_mlx_streaming(self, audio_input, is_final=False):
        """Process audio using MLX backend with live transcription display."""
        # Use streaming handler for progressive transcription
        result = self.streaming_handler.transcribe_incremental(audio_input)

        if is_final:
            # Clear the live transcription line
            console.print(" " * 100, end="\r")

        # Get final combined text
        if result.fixed_text and result.active_text:
            pred_text = f"{result.fixed_text} {result.active_text}".strip()
        elif result.fixed_text:
            pred_text = result.fixed_text.strip()
        else:
            pred_text = result.active_text.strip()

        # Detect language
        if self.start_language and self.start_language != "auto":
            language_code = self.start_language
        else:
            detected_lang = self._detect_language_from_text(pred_text)
            if detected_lang:
                language_code = detected_lang
            else:
                language_code = self.last_language

        # Reset streaming handler for next audio
        if is_final:
            self.streaming_handler.reset()

        return pred_text, language_code

    def _process_mlx_final(self, audio_input):
        """Process final audio using MLX backend with streaming handler."""
        # If we have fixed sentences from progressive updates, only transcribe the new part
        if hasattr(self.streaming_handler, 'fixed_sentences') and self.streaming_handler.fixed_sentences:
            # Clear the live transcription line
            console.print(" " * 100, end="\r")

            # Get fixed text from previous progressive updates
            fixed_text = " ".join(self.streaming_handler.fixed_sentences).strip()

            # Calculate where fixed part ends in audio
            fixed_end_time = self.streaming_handler.fixed_end_time
            sample_rate = 16000
            fixed_end_sample = int(fixed_end_time * sample_rate)

            # Only transcribe the part after fixed sentences
            if fixed_end_sample < len(audio_input):
                remaining_audio = audio_input[fixed_end_sample:]

                import mlx.core as mx
                audio_mx = mx.array(remaining_audio, dtype=mx.float32)
                result = self.model.decode_chunk(audio_mx, verbose=False)

                if hasattr(result, "text"):
                    new_text = result.text.strip()
                else:
                    new_text = str(result).strip()

                # Combine fixed + new
                pred_text = f"{fixed_text} {new_text}".strip() if new_text else fixed_text
            else:
                # All audio already transcribed in progressive updates
                pred_text = fixed_text

            # Reset streaming handler for next utterance
            self.streaming_handler.reset()
        else:
            # No progressive updates, transcribe everything
            pred_text, language_code = self._process_mlx(audio_input)
            return pred_text, language_code

        # Determine language
        if self.start_language and self.start_language != "auto":
            language_code = self.start_language
        else:
            detected_lang = self._detect_language_from_text(pred_text)
            if detected_lang:
                language_code = detected_lang
            else:
                language_code = self.last_language

        return pred_text, language_code

    def _process_mlx(self, audio_input):
        """Process audio using MLX backend."""
        import mlx.core as mx

        # Convert numpy array to mx.array
        audio_mx = mx.array(audio_input, dtype=mx.float32)

        # Call decode_chunk directly with the audio array
        result = self.model.decode_chunk(audio_mx, verbose=False)

        # Extract text from result
        if hasattr(result, "text"):
            pred_text = result.text.strip()
        else:
            pred_text = str(result).strip()

        # Determine language:
        # 1. Use fixed language if specified by user
        # 2. Try to detect from transcribed text using langdetect
        # 3. Fall back to last known language
        if self.start_language and self.start_language != "auto":
            language_code = self.start_language
        else:
            # Detect language from transcribed text
            detected_lang = self._detect_language_from_text(pred_text)
            if detected_lang:
                language_code = detected_lang
            else:
                language_code = self.last_language

        return pred_text, language_code

    def _process_nemo(self, audio_input):
        """Process audio using NeMo backend."""
        # Transcribe directly from the audio array
        output = self.model.transcribe(
            [audio_input],
            batch_size=1,
            verbose=False,
            **self.gen_kwargs
        )

        # Extract transcription
        if hasattr(output[0], "text"):
            pred_text = output[0].text.strip()
        else:
            pred_text = str(output[0]).strip()

        # Parakeet TDT auto-detects language but doesn't always expose it
        # Default to last known or English
        language_code = self.last_language

        return pred_text, language_code

    def cleanup(self):
        """Clean up model resources."""
        logger.info(f"Cleaning up {self.__class__.__name__}")
        if hasattr(self, "model"):
            del self.model
