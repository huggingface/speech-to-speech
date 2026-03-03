"""
Qwen3 TTS Handler

Requires faster-qwen3-tts for real-time performance on NVIDIA GPUs:
  pip install faster-qwen3-tts
"""

import logging
from time import perf_counter
import numpy as np
from baseHandler import BaseHandler
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_REF_TEXT = "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes."
PIPELINE_SR = 16000


class Qwen3TTSHandler(BaseHandler):
    """
    Handles Text-to-Speech using Qwen3-TTS via the faster-qwen3-tts backend.

    Supports three generation modes depending on the loaded model:
      - Voice cloning (ref_audio + ref_text)
      - Custom voice (preset speakers)
      - Voice design (instruct prompt)

    All modes use streaming generation for low time-to-first-audio.
    """

    def setup(
        self,
        should_listen,
        model_name=DEFAULT_MODEL,
        device="cuda",
        dtype="auto",
        attn_implementation="eager",
        ref_audio=None,
        ref_text=DEFAULT_REF_TEXT,
        language="English",
        speaker=None,
        instruct=None,
        streaming_chunk_size=8,
        max_new_tokens=360,
        gen_kwargs=None,
    ):
        self.should_listen = should_listen
        self.model_name = model_name
        self.device = device
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.language = language
        self.speaker = speaker
        self.instruct = instruct
        self.streaming_chunk_size = streaming_chunk_size
        self.max_new_tokens = max_new_tokens

        try:
            import torch
        except ImportError as e:
            raise ImportError("torch is required. Install with: pip install torch") from e

        if dtype == "auto":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif isinstance(dtype, str):
            self.dtype = getattr(torch, dtype)
        else:
            self.dtype = dtype

        try:
            from faster_qwen3_tts import FasterQwen3TTS
        except ImportError as e:
            raise ImportError(
                "faster-qwen3-tts is required for Qwen3 TTS. "
                "Install with: pip install faster-qwen3-tts"
            ) from e

        logger.info(f"Loading Qwen3-TTS model: {self.model_name}")
        self.model = FasterQwen3TTS.from_pretrained(
            self.model_name,
            device=self.device,
            dtype=self.dtype,
            attn_implementation=attn_implementation,
        )
        logger.info("Qwen3-TTS model loaded")

        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        try:
            self.model._warmup(prefill_len=100)
        except Exception as e:
            logger.warning(f"CUDA graph capture failed: {e}")
        try:
            for _ in self.process("Hello, this is a warmup."):
                pass
            logger.info(f"{self.__class__.__name__} warmed up")
        except Exception as e:
            logger.warning(f"Warmup generation failed: {e}")

    def _model_type(self):
        inner = getattr(getattr(self.model, "model", None), "model", None)
        return getattr(inner, "tts_model_type", None)

    def _resolve_speaker(self):
        if self.speaker:
            return self.speaker
        inner = getattr(getattr(self.model, "model", None), "model", None)
        get_speakers = getattr(inner, "get_supported_speakers", None)
        if callable(get_speakers):
            speakers = list(get_speakers() or [])
            if speakers:
                return speakers[0]
        return None

    def _to_int16(self, audio):
        if audio.dtype != np.int16:
            return (audio * 32768).astype(np.int16)
        return audio

    def _resample_to_pipeline_sr(self, audio, sr):
        if sr == PIPELINE_SR:
            return audio
        from scipy.signal import resample_poly
        gcd = np.gcd(PIPELINE_SR, sr)
        return resample_poly(audio, up=PIPELINE_SR // gcd, down=sr // gcd)

    def _stream(self, gen, label):
        """Common streaming loop: log TTFA and RTF, yield int16 chunks."""
        start = perf_counter()
        total_samples = 0
        first_chunk = True

        for audio_chunk, sr, _timing in gen:
            if first_chunk:
                logger.info(f"Qwen3-TTS TTFA: {perf_counter() - start:.2f}s ({label})")
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
            f"(RTF: {rtf:.2f}, {label})"
        )

    def process(self, llm_sentence):
        if isinstance(llm_sentence, tuple):
            llm_sentence, _ = llm_sentence
        if not llm_sentence:
            llm_sentence = "Hello."

        console.print(f"[green]ASSISTANT: {llm_sentence}")

        try:
            model_type = self._model_type()
            if self.ref_audio:
                yield from self._process_voice_clone(llm_sentence)
            elif model_type == "custom_voice":
                yield from self._process_custom_voice(llm_sentence)
            elif model_type == "voice_design":
                yield from self._process_voice_design(llm_sentence)
            else:
                raise ValueError(
                    "Qwen3-TTS Base model requires ref_audio for voice cloning. "
                    "Provide qwen3_tts_ref_audio or use a CustomVoice/VoiceDesign model."
                )
        except Exception as e:
            logger.error(f"Error during Qwen3-TTS generation: {e}", exc_info=True)
            yield np.zeros(160, dtype=np.int16)

        self.should_listen.set()

    def _process_voice_clone(self, text):
        yield from self._stream(
            self.model.generate_voice_clone_streaming(
                text=text,
                language=self.language,
                ref_audio=self.ref_audio,
                ref_text=self.ref_text,
                xvec_only=False,
                chunk_size=self.streaming_chunk_size,
                max_new_tokens=self.max_new_tokens,
            ),
            label="voice_clone",
        )

    def _process_custom_voice(self, text):
        speaker = self._resolve_speaker()
        if not speaker:
            raise ValueError(
                "CustomVoice generation requires a speaker. "
                "Set qwen3_tts_speaker or use a voice-clone model with ref_audio."
            )
        yield from self._stream(
            self.model.generate_custom_voice_streaming(
                text=text,
                speaker=speaker,
                language=self.language,
                instruct=self.instruct,
                chunk_size=self.streaming_chunk_size,
                max_new_tokens=self.max_new_tokens,
            ),
            label="custom_voice",
        )

    def _process_voice_design(self, text):
        yield from self._stream(
            self.model.generate_voice_design_streaming(
                text=text,
                instruct=self.instruct,
                language=self.language,
                chunk_size=self.streaming_chunk_size,
                max_new_tokens=self.max_new_tokens,
            ),
            label="voice_design",
        )

    def cleanup(self):
        try:
            import torch
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Qwen3-TTS handler cleaned up")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
