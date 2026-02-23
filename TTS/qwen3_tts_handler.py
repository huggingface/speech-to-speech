"""
Qwen3 TTS Handler

Supports Qwen3-TTS models for high-quality voice cloning and multilingual speech synthesis.
Models: Qwen3-TTS-12Hz-0.6B-Base, Qwen3-TTS-12Hz-1.7B-Base

Requires:
- qwen-tts library (pip install qwen-tts)
- torch with CUDA support for optimal performance

Optional: For real-time performance on NVIDIA GPUs, install faster-qwen3-tts:
  pip install faster-qwen3-tts
"""

import logging
from time import perf_counter
from pathlib import Path
import numpy as np
from baseHandler import BaseHandler
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_MODEL = "Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_REF_TEXT = "This is a reference audio sample for voice cloning."
PIPELINE_SR = 16000


class Qwen3TTSHandler(BaseHandler):
    """
    Handles Text-to-Speech using Qwen3-TTS models.

    Supports voice cloning via reference audio and multilingual synthesis.
    Optionally uses CUDA graphs for real-time performance on NVIDIA GPUs.
    With CUDA graphs, uses streaming generation for low time-to-first-audio.
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
        speaker=None,
        instruct=None,
        use_cuda_graphs=False,
        streaming_chunk_size=8,
        max_new_tokens=200,
        gen_kwargs=None,
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
            streaming_chunk_size: Codec steps per streaming chunk (8 = ~667ms)
            max_new_tokens: Maximum codec tokens to generate (~12 tokens per second of audio)
        """
        self.should_listen = should_listen
        self.model_name = model_name
        self.device = device
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.language = language
        self.speaker = speaker
        self.instruct = instruct
        self.attn_implementation = attn_implementation
        self.use_cuda_graphs = use_cuda_graphs
        self.streaming_chunk_size = streaming_chunk_size
        self.max_new_tokens = max_new_tokens
        self.base_model = None
        self.cuda_graphs_model = None

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
        self.base_model = self.model
        self.cuda_graphs_model = None
        logger.info("Qwen3-TTS model loaded (standard backend)")

    def _setup_cuda_graphs(self):
        """Setup using CUDA graphs backend for real-time performance."""
        try:
            from faster_qwen3_tts import FasterQwen3TTS
        except ImportError:
            logger.warning(
                "faster-qwen3-tts not available, falling back to standard backend. "
                "For real-time performance on NVIDIA GPUs, install: "
                "pip install faster-qwen3-tts"
            )
            self._setup_standard()
            return

        import torch

        logger.info(f"Loading Qwen3-TTS model with CUDA graphs: {self.model_name}")

        try:
            self.model = FasterQwen3TTS.from_pretrained(
                self.model_name,
                device=self.device,
                dtype=self.dtype,
                attn_implementation=self.attn_implementation,
            )
            self.backend = "cuda_graphs"
            self.cuda_graphs_model = self.model
            self.base_model = self.model.model
            logger.info("Qwen3-TTS model loaded (CUDA graphs backend)")
        except Exception as e:
            logger.warning(f"CUDA graphs initialization failed: {e}. Falling back to standard backend.")
            self._setup_standard()

    def warmup(self):
        """Warm up the model with a dummy inference."""
        logger.info(f"Warming up {self.__class__.__name__}")

        warmup_text = "Hello, this is a warmup."

        try:
            for _ in self.process(warmup_text):
                pass
            logger.info(f"{self.__class__.__name__} warmed up")
        except Exception as e:
            logger.warning(f"Warmup failed (this may happen if no reference audio is provided): {e}")

    def _to_int16(self, audio):
        """Convert float audio to int16."""
        if audio.dtype != np.int16:
            return (audio * 32768).astype(np.int16)
        return audio

    def _resample_to_pipeline_sr(self, audio, sr):
        """Resample audio to pipeline sample rate (16kHz) if needed."""
        if sr == PIPELINE_SR:
            return audio
        from scipy.signal import resample_poly
        gcd = np.gcd(PIPELINE_SR, sr)
        return resample_poly(audio, up=PIPELINE_SR // gcd, down=sr // gcd)

    def process(self, llm_sentence):
        """
        Process text input and generate audio output.

        Uses streaming generation with CUDA graphs backend for low latency,
        or batch generation with standard backend.

        Args:
            llm_sentence: Either a string or tuple of (text, language_code)

        Yields:
            Audio chunks as numpy int16 arrays at 16kHz
        """
        if isinstance(llm_sentence, tuple):
            llm_sentence, _language_code = llm_sentence

        if not llm_sentence:
            llm_sentence = "Hello."

        console.print(f"[green]ASSISTANT: {llm_sentence}")

        try:
            if self.backend == "cuda_graphs":
                if self.ref_audio:
                    yield from self._process_streaming(llm_sentence)
                elif self._can_stream_custom_voice():
                    yield from self._process_streaming_custom_voice(llm_sentence)
                else:
                    yield from self._process_batch(llm_sentence)
            else:
                yield from self._process_batch(llm_sentence)
        except Exception as e:
            logger.error(f"Error during Qwen3-TTS generation: {e}", exc_info=True)
            yield np.zeros(self.blocksize, dtype=np.int16)

        self.should_listen.set()

    def _process_streaming(self, text):
        """Stream audio using CUDA graphs streaming generation."""
        start = perf_counter()
        total_samples = 0
        first_chunk = True

        for audio_chunk, sr, timing in self.cuda_graphs_model.generate_voice_clone_streaming(
            text=text,
            language=self.language,
            ref_audio=self.ref_audio,
            ref_text=self.ref_text,
            chunk_size=self.streaming_chunk_size,
            max_new_tokens=self.max_new_tokens,
        ):
            if first_chunk:
                ttfa = perf_counter() - start
                logger.info(f"Qwen3-TTS TTFA: {ttfa:.2f}s (streaming, cuda_graphs)")
                first_chunk = False

            audio_chunk = self._resample_to_pipeline_sr(audio_chunk, sr)
            audio_chunk = self._to_int16(audio_chunk)
            total_samples += len(audio_chunk)
            yield audio_chunk

        generation_time = perf_counter() - start
        audio_duration = total_samples / PIPELINE_SR
        rtf = audio_duration / generation_time if generation_time > 0 else 0
        logger.info(
            f"Qwen3-TTS generated {audio_duration:.2f}s audio in {generation_time:.2f}s "
            f"(RTF: {rtf:.2f}, streaming, cuda_graphs)"
        )

    def _process_streaming_custom_voice(self, text):
        """Stream audio using CUDA graphs custom voice generation."""
        start = perf_counter()
        total_samples = 0
        first_chunk = True

        speaker = self._resolve_speaker()
        if not speaker:
            raise ValueError(
                "CustomVoice generation requires a speaker. "
                "Set qwen3_tts_speaker or use a voice-clone model with ref_audio."
            )

        for audio_chunk, sr, timing in self.cuda_graphs_model.generate_custom_voice_streaming(
            text=text,
            speaker=speaker,
            language=self.language,
            instruct=self.instruct,
            chunk_size=self.streaming_chunk_size,
            max_new_tokens=self.max_new_tokens,
        ):
            if first_chunk:
                ttfa = perf_counter() - start
                logger.info(f"Qwen3-TTS TTFA: {ttfa:.2f}s (streaming, cuda_graphs, custom_voice)")
                first_chunk = False

            audio_chunk = self._resample_to_pipeline_sr(audio_chunk, sr)
            audio_chunk = self._to_int16(audio_chunk)
            total_samples += len(audio_chunk)
            yield audio_chunk

        generation_time = perf_counter() - start
        audio_duration = total_samples / PIPELINE_SR
        rtf = audio_duration / generation_time if generation_time > 0 else 0
        logger.info(
            f"Qwen3-TTS generated {audio_duration:.2f}s audio in {generation_time:.2f}s "
            f"(RTF: {rtf:.2f}, streaming, cuda_graphs, custom_voice)"
        )

    def _resolve_speaker(self):
        if self.speaker:
            return self.speaker
        if not self.base_model:
            return None
        model = getattr(self.base_model, "model", None)
        get_speakers = getattr(model, "get_supported_speakers", None)
        if callable(get_speakers):
            speakers = list(get_speakers() or [])
            if speakers:
                return speakers[0]
        return None

    def _model_type(self):
        if not self.base_model:
            return None
        model = getattr(self.base_model, "model", None)
        return getattr(model, "tts_model_type", None)

    def _can_stream_custom_voice(self):
        if self.backend != "cuda_graphs":
            return False
        if not self.cuda_graphs_model:
            return False
        if not hasattr(self.cuda_graphs_model, "generate_custom_voice_streaming"):
            return False
        return self._model_type() == "custom_voice"

    def _process_batch(self, text):
        """Generate all audio at once, then yield in chunks."""
        start = perf_counter()

        backend_label = self.backend
        model_type = self._model_type()
        if self.ref_audio:
            if self.backend == "cuda_graphs":
                wavs, sr = self.cuda_graphs_model.generate_voice_clone(
                    text=text,
                    language=self.language,
                    ref_audio=self.ref_audio,
                    ref_text=self.ref_text,
                    max_new_tokens=self.max_new_tokens,
                )
            else:
                wavs, sr = self.base_model.generate_voice_clone(
                    text=text,
                    language=self.language,
                    ref_audio=self.ref_audio,
                    ref_text=self.ref_text,
                    max_new_tokens=self.max_new_tokens,
                )
        else:
            if model_type == "custom_voice":
                speaker = self._resolve_speaker()
                if not speaker:
                    raise ValueError(
                        "CustomVoice generation requires a speaker. "
                        "Set qwen3_tts_speaker or use a voice-clone model with ref_audio."
                    )
                if self.backend == "cuda_graphs":
                    wavs, sr = self.cuda_graphs_model.generate_custom_voice(
                        text=text,
                        speaker=speaker,
                        language=self.language,
                        instruct=self.instruct,
                        max_new_tokens=self.max_new_tokens,
                    )
                else:
                    wavs, sr = self.base_model.generate_custom_voice(
                        text=text,
                        speaker=speaker,
                        language=self.language,
                        instruct=self.instruct,
                        max_new_tokens=self.max_new_tokens,
                    )
            elif model_type == "voice_design":
                if not self.instruct:
                    raise ValueError(
                        "VoiceDesign generation requires qwen3_tts_instruct. "
                        "Set it or use a different model type."
                    )
                if self.backend == "cuda_graphs":
                    backend_label = "standard (no ref_audio)"
                wavs, sr = self.base_model.generate_voice_design(
                    text=text,
                    instruct=self.instruct,
                    language=self.language,
                    max_new_tokens=self.max_new_tokens,
                )
            else:
                raise ValueError(
                    "Qwen3-TTS Base model requires ref_audio for voice cloning. "
                    "Provide qwen3_tts_ref_audio or use a CustomVoice/VoiceDesign model."
                )

        audio = self._resample_to_pipeline_sr(wavs[0], sr)
        audio = self._to_int16(audio)

        generation_time = perf_counter() - start
        audio_duration = len(audio) / PIPELINE_SR
        rtf = audio_duration / generation_time if generation_time > 0 else 0
        logger.info(
            f"Qwen3-TTS generated {audio_duration:.2f}s audio in {generation_time:.2f}s "
            f"(RTF: {rtf:.2f}, batch, {backend_label})"
        )

        yield audio

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
