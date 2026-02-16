"""
Qwen3 TTS Handler

Supports Qwen3-TTS models for high-quality voice cloning and multilingual speech synthesis.
Models: Qwen3-TTS-12Hz-0.6B-Base, Qwen3-TTS-12Hz-1.7B-Base

Requires:
- qwen-tts library (pip install qwen-tts)
- torch with CUDA support for optimal performance

Optional: For real-time performance on NVIDIA GPUs, install qwen3-tts-cuda-graphs:
  pip install git+https://github.com/andimarafioti/qwen3-tts-cuda-graphs.git
"""

import logging
from time import perf_counter
from pathlib import Path
import numpy as np
from baseHandler import BaseHandler

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_REF_TEXT = "This is a reference audio sample for voice cloning."


class Qwen3TTSHandler(BaseHandler):
    """
    Handles Text-to-Speech using Qwen3-TTS models.
    
    Supports voice cloning via reference audio and multilingual synthesis.
    Optionally uses CUDA graphs for real-time performance on NVIDIA GPUs.
    """

    def setup(
        self,
        should_listen,
        model_name=DEFAULT_MODEL,
        device="cuda",
        dtype="auto",
        attn_implementation="flash_attention_2",
        ref_audio=None,
        ref_text=DEFAULT_REF_TEXT,
        language="English",
        use_cuda_graphs=False,
        blocksize=512,
    ):
        """
        Initialize the Qwen3-TTS model.

        Args:
            should_listen: Threading event to signal when audio is ready
            model_name: Model identifier (HuggingFace Hub or local path)
            device: Device to use ("cuda", "cpu")
            dtype: Data type ("auto", "float16", "bfloat16", "float32")
            attn_implementation: Attention implementation ("flash_attention_2", "eager", "sdpa")
            ref_audio: Path to reference audio file for voice cloning (optional)
            ref_text: Transcription of the reference audio
            language: Target language for synthesis
            use_cuda_graphs: Use CUDA graphs for faster inference (NVIDIA GPUs only)
            blocksize: Audio chunk size for streaming
        """
        self.should_listen = should_listen
        self.model_name = model_name
        self.device = device
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.language = language
        self.attn_implementation = attn_implementation
        self.use_cuda_graphs = use_cuda_graphs
        self.blocksize = blocksize

        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "torch is required for Qwen3 TTS. "
                "Install with: pip install torch"
            ) from e

        # Determine dtype
        if dtype == "auto":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif isinstance(dtype, str):
            self.dtype = getattr(torch, dtype)
        else:
            self.dtype = dtype

        # Load model using appropriate backend
        if self.use_cuda_graphs and device == "cuda" and torch.cuda.is_available():
            self._setup_cuda_graphs()
        else:
            self._setup_standard()

        self.warmup()

    def _setup_standard(self):
        """Setup using standard qwen-tts library."""
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError as e:
            raise ImportError(
                "qwen-tts is required for Qwen3 TTS. "
                "Install with: pip install qwen-tts"
            ) from e

        import torch

        logger.info(f"Loading Qwen3-TTS model: {self.model_name}")
        
        # Try to load from HuggingFace Hub first, fall back to local path
        try:
            self.model = Qwen3TTSModel.from_pretrained(
                self.model_name,
                device_map="cuda:0" if self.device == "cuda" and torch.cuda.is_available() else "cpu",
                torch_dtype=self.dtype,
                attn_implementation=self.attn_implementation,
            )
        except Exception as hub_error:
            # If HF Hub fails, try as local path
            model_path = Path(self.model_name)
            if not model_path.exists():
                raise ValueError(
                    f"Model not found on HuggingFace Hub or as local path: {self.model_name}\n"
                    f"HuggingFace error: {hub_error}"
                )
            
            logger.info(f"Loading from local path: {model_path}")
            self.model = Qwen3TTSModel.from_pretrained(
                str(model_path),
                device_map="cuda:0" if self.device == "cuda" and torch.cuda.is_available() else "cpu",
                torch_dtype=self.dtype,
                attn_implementation=self.attn_implementation,
            )

        self.backend = "standard"
        logger.info("Qwen3-TTS model loaded (standard backend)")

    def _setup_cuda_graphs(self):
        """Setup using CUDA graphs backend for real-time performance."""
        try:
            from qwen3_tts_cuda_graphs import Qwen3TTSCudaGraphs
        except ImportError as e:
            logger.warning(
                "qwen3-tts-cuda-graphs not available, falling back to standard backend. "
                "For real-time performance on NVIDIA GPUs, install: "
                "pip install git+https://github.com/andimarafioti/qwen3-tts-cuda-graphs.git"
            )
            self._setup_standard()
            return

        import torch

        logger.info(f"Loading Qwen3-TTS model with CUDA graphs: {self.model_name}")
        
        try:
            self.model = Qwen3TTSCudaGraphs.from_pretrained(
                self.model_name,
                device=self.device,
                dtype=self.dtype,
                attn_implementation=self.attn_implementation,
            )
            self.backend = "cuda_graphs"
            logger.info("Qwen3-TTS model loaded (CUDA graphs backend)")
        except Exception as e:
            logger.warning(f"CUDA graphs initialization failed: {e}. Falling back to standard backend.")
            self._setup_standard()

    def warmup(self):
        """Warm up the model with a dummy inference."""
        logger.info(f"Warming up {self.__class__.__name__}")
        
        warmup_text = "Hello, this is a warmup."
        
        try:
            # Run dummy inference
            for _ in self.process(warmup_text):
                pass
            logger.info(f"{self.__class__.__name__} warmed up")
        except Exception as e:
            logger.warning(f"Warmup failed (this may happen if no reference audio is provided): {e}")

    def process(self, llm_sentence):
        """
        Process text input and generate audio output.

        Args:
            llm_sentence: Either a string or tuple of (text, language_code)

        Yields:
            Audio chunks as numpy int16 arrays
        """
        import torch

        language_code = None
        if isinstance(llm_sentence, tuple):
            llm_sentence, language_code = llm_sentence
            # Map language code to Qwen3-TTS language names if needed
            # For now, use the configured language
        
        if not llm_sentence:
            llm_sentence = "Hello."

        start = perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        try:
            # Generate audio using voice cloning if reference audio provided
            if self.ref_audio:
                wavs, sr = self.model.generate_voice_clone(
                    text=llm_sentence,
                    language=self.language,
                    ref_audio=self.ref_audio,
                    ref_text=self.ref_text,
                )
            else:
                # Generate with default voice
                wavs, sr = self.model.generate(
                    text=llm_sentence,
                    language=self.language,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            audio = wavs[0]
            
            # Convert to int16 format
            if audio.dtype != np.int16:
                audio = (audio * 32768).astype(np.int16)

            generation_time = perf_counter() - start
            audio_duration = len(audio) / sr
            rtf = audio_duration / generation_time if generation_time > 0 else 0

            logger.info(
                f"Qwen3-TTS generated {audio_duration:.2f}s audio in {generation_time:.2f}s "
                f"(RTF: {rtf:.2f}, {'real-time' if rtf >= 1.0 else 'slower than real-time'})"
            )

            # Resample to 16kHz if needed (speech-to-speech pipeline expects 16kHz)
            if sr != 16000:
                from scipy.signal import resample_poly
                # Calculate resampling ratio
                gcd = np.gcd(16000, sr)
                up = 16000 // gcd
                down = sr // gcd
                audio = resample_poly(audio, up=up, down=down)

            # Yield audio in chunks
            for i in range(0, len(audio), self.blocksize):
                chunk = audio[i : i + self.blocksize]
                # Pad the last chunk if necessary
                if len(chunk) < self.blocksize:
                    chunk = np.pad(chunk, (0, self.blocksize - len(chunk)))
                yield chunk

        except Exception as e:
            logger.error(f"Error during Qwen3-TTS generation: {e}", exc_info=True)
            # Yield silence on error to avoid breaking the pipeline
            yield np.zeros(self.blocksize, dtype=np.int16)

        self.should_listen.set()

    def cleanup(self):
        """Clean up model resources."""
        try:
            import torch
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Qwen3-TTS handler cleaned up")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
