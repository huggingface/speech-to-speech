"""
Qwen3 TTS Handler

- On Apple Silicon: Uses mlx-audio with MLX-converted Qwen3-TTS models.
- On CUDA/CPU: Uses faster-qwen3-tts for low-latency streaming.
"""

import logging
from sys import platform
from time import perf_counter

import numpy as np
from rich.console import Console

from api.openai_realtime.runtime_config import RuntimeConfig
from baseHandler import BaseHandler
from cancel_scope import CancelScope
from pipeline_control import SESSION_END, is_control_message
from pipeline_messages import AUDIO_RESPONSE_DONE, PIPELINE_END, MessageTag
from utils.mlx_lock import MLXLockContext

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_MLX_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
DEFAULT_REF_TEXT = "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes."
MLX_STREAMING_TOKENS_PER_SECOND = 12.5
PIPELINE_SR = 16000


class Qwen3TTSHandler(BaseHandler):
    """
    Handles Text-to-Speech using Qwen3-TTS.

    Backend selection:
      - Apple Silicon (Darwin): mlx-audio
      - Other platforms: faster-qwen3-tts

    Supports three generation modes depending on the loaded model:
      - Voice cloning (ref_audio + ref_text)
      - Custom voice (preset speakers)
      - Voice design (instruct prompt)
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
        xvec_only=False,
        parity_mode=False,
        streaming_chunk_size=8,
        max_new_tokens=360,
        blocksize=512,
        gen_kwargs=None,
        runtime_config: RuntimeConfig | None = None,
        cancel_scope: CancelScope | None = None,
    ):
        self.runtime_config = runtime_config
        self.cancel_scope = cancel_scope
        self.should_listen = should_listen
        self.requested_device = device
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.language = language
        self.speaker = speaker
        self.instruct = instruct
        self.xvec_only = xvec_only
        self.parity_mode = parity_mode
        self.streaming_chunk_size = streaming_chunk_size
        self.max_new_tokens = max_new_tokens
        self.blocksize = blocksize
        self.gen_kwargs = gen_kwargs or {}

        self.backend = "mlx" if platform == "darwin" else "faster_qwen3_tts"

        if self.backend == "mlx":
            self.device = "mps"
            self.model_name = self._resolve_mlx_model_name(model_name)
            logger.info(
                f"Loading Qwen3-TTS model: {self.model_name} via mlx-audio on Apple Silicon"
            )
            self._setup_mlx(self.model_name)
        else:
            self.device = device
            self.model_name = model_name
            logger.info(
                f"Loading Qwen3-TTS model: {self.model_name} via faster-qwen3-tts"
            )
            self._setup_faster(
                model_name=self.model_name,
                dtype=dtype,
                attn_implementation=attn_implementation,
            )

        self.warmup()

    def _setup_faster(self, model_name, dtype, attn_implementation):
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
                "faster-qwen3-tts is required for Qwen3 TTS on non-macOS platforms. "
                "Install with: pip install faster-qwen3-tts"
            ) from e

        self.model = FasterQwen3TTS.from_pretrained(
            model_name,
            device=self.device,
            dtype=self.dtype,
            attn_implementation=attn_implementation,
        )
        logger.info("Qwen3-TTS model loaded")

    def _setup_mlx(self, model_name):
        try:
            from mlx_audio.tts.utils import load_model

            self.model = load_model(model_name)
        except ImportError as e:
            message = str(e)
            if any(
                dep in message
                for dep in (
                    "sentencepiece",
                    "num2words",
                    "misaki",
                    "spacy",
                    "phonemizer",
                    "espeakng_loader",
                )
            ):
                raise ImportError(
                    "Qwen3-TTS on Apple Silicon requires mlx-audio and its TTS dependencies. "
                    f"Missing dependency: {message}. "
                    "Install with: pip install mlx-audio sentencepiece num2words misaki spacy phonemizer-fork espeakng-loader"
                ) from e
            raise ImportError(
                "mlx-audio is required for Qwen3 TTS on Apple Silicon. "
                "Install with: pip install mlx-audio"
            ) from e

        logger.info("MLX Audio Qwen3-TTS model loaded")

    def _resolve_mlx_model_name(self, model_name):
        if not model_name:
            return DEFAULT_MLX_MODEL
        if model_name.startswith("mlx-community/"):
            return model_name
        if model_name.startswith("Qwen/"):
            mapped = model_name.replace("Qwen/", "mlx-community/", 1)
            if not mapped.endswith("-bf16"):
                mapped = f"{mapped}-bf16"
            return mapped
        return model_name

    def _infer_model_type_from_name(self):
        name = (self.model_name or "").lower()
        if "voicedesign" in name:
            return "voice_design"
        if "customvoice" in name:
            return "custom_voice"
        return "base"

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        if self.backend == "faster_qwen3_tts":
            if self.parity_mode:
                logger.info("Qwen3-TTS parity mode enabled: skipping CUDA graph capture warmup")
            else:
                try:
                    self.model._warmup(prefill_len=100)
                except Exception as e:
                    logger.warning(f"CUDA graph capture failed: {e}")

        try:
            for _ in self._warmup_process("Hello, this is a warmup."):
                pass
            logger.info(f"{self.__class__.__name__} warmed up")
        except Exception as e:
            logger.warning(f"Warmup generation failed: {e}")

    def _model_type(self):
        if self.backend == "mlx":
            config = getattr(self.model, "config", None)
            return getattr(config, "tts_model_type", None) or self._infer_model_type_from_name()

        inner = getattr(getattr(self.model, "model", None), "model", None)
        return getattr(inner, "tts_model_type", None) or self._infer_model_type_from_name()

    def _resolve_speaker(self):
        if self.speaker:
            return self.speaker

        for candidate in (
            self.model,
            getattr(getattr(self.model, "model", None), "model", None),
        ):
            get_speakers = getattr(candidate, "get_supported_speakers", None)
            if callable(get_speakers):
                speakers = list(get_speakers() or [])
                if speakers:
                    return speakers[0]

        return None

    def _to_int16(self, audio):
        return np.clip(audio * 32768, -32768, 32767).astype(np.int16)

    def _warmup_process(self, llm_sentence):
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

    def _resample_to_pipeline_sr(self, audio, sr):
        if sr == PIPELINE_SR:
            return audio
        from scipy.signal import resample_poly

        gcd = np.gcd(PIPELINE_SR, sr)
        return resample_poly(audio, up=PIPELINE_SR // gcd, down=sr // gcd)

    def _prepare_audio_chunk(self, item):
        if isinstance(item, tuple):
            audio_chunk, sr, _timing = item
            return np.asarray(audio_chunk, dtype=np.float32), sr

        audio = getattr(item, "audio", None)
        if audio is None:
            return None, None

        audio_chunk = np.asarray(audio, dtype=np.float32).squeeze()
        sr = getattr(item, "sample_rate", None) or PIPELINE_SR
        return audio_chunk, sr

    def _stream(self, gen, label):
        """Common streaming loop: log TTFA and RTF, yield int16 chunks."""
        cancel_gen = self.cancel_scope.generation if self.cancel_scope else None
        start = perf_counter()
        total_samples = 0
        first_chunk = True
        found_speech = False
        leftover = np.array([], dtype=np.int16)

        for item in gen:
            if cancel_gen is not None and self.cancel_scope.is_stale(cancel_gen):
                logger.info("TTS generation cancelled (interruption)")
                return

            audio_chunk, sr = self._prepare_audio_chunk(item)
            if audio_chunk is None or sr is None or audio_chunk.size == 0:
                continue

            if first_chunk:
                logger.info(f"Qwen3-TTS TTFA: {perf_counter() - start:.2f}s ({label})")
                first_chunk = False

            audio_chunk = self._resample_to_pipeline_sr(audio_chunk, sr)
            audio_chunk = self._to_int16(audio_chunk)

            # Trim the initial silent ramp-up, but keep enough preroll to avoid
            # shaving soft initial phonemes at the start of the utterance.
            if not found_speech:
                threshold = int(32768 * 0.01)
                above = np.abs(audio_chunk) > threshold
                if not np.any(above):
                    continue
                start_idx = max(0, int(np.argmax(above)) - int(PIPELINE_SR * 0.040))
                audio_chunk = audio_chunk[start_idx:]
                found_speech = True

            audio_chunk = np.concatenate([leftover, audio_chunk])

            n = (len(audio_chunk) // self.blocksize) * self.blocksize
            for i in range(0, n, self.blocksize):
                yield audio_chunk[i : i + self.blocksize]
                total_samples += self.blocksize
            leftover = audio_chunk[n:]

        if len(leftover) > 0:
            chunk = np.pad(leftover, (0, self.blocksize - len(leftover)))
            yield chunk
            total_samples += len(leftover)

        generation_time = perf_counter() - start
        audio_duration = total_samples / PIPELINE_SR
        rtf = audio_duration / generation_time if generation_time > 0 else 0
        logger.info(
            f"Qwen3-TTS generated {audio_duration:.2f}s audio in {generation_time:.2f}s "
            f"(RTF: {rtf:.2f}, {label})"
        )

    def _coalesce_pending_tts_input(self, current_input):
        """Combine already-queued text chunks before the next TTS synthesis call."""
        if not hasattr(self.queue_in, "mutex") or not hasattr(self.queue_in, "queue"):
            return current_input, False

        def _decode(item):
            if isinstance(item, tuple):
                if item and item[0] == MessageTag.END_OF_RESPONSE:
                    return None, None, True
                if len(item) == 2 and isinstance(item[0], str):
                    return item[0], item[1], False
            elif isinstance(item, str):
                return item, None, False
            return None, None, False

        text, language_code, _ = _decode(current_input)
        if text is None:
            return current_input, False

        parts = [text.strip()] if text and text.strip() else []
        saw_end_of_response = False

        with self.queue_in.mutex:
            while self.queue_in.queue:
                next_item = self.queue_in.queue[0]
                if is_control_message(next_item, SESSION_END.kind):
                    break
                if isinstance(next_item, bytes) and next_item == PIPELINE_END:
                    break

                next_text, next_language_code, is_end = _decode(next_item)
                if is_end:
                    saw_end_of_response = True
                    break
                if next_text is None:
                    break
                if (
                    language_code is not None
                    and next_language_code is not None
                    and next_language_code != language_code
                ):
                    break

                self.queue_in.queue.popleft()
                if next_text.strip():
                    parts.append(next_text.strip())
                if language_code is None:
                    language_code = next_language_code

        combined_text = " ".join(parts).strip()
        if isinstance(current_input, tuple):
            return (combined_text, language_code), saw_end_of_response
        return combined_text, saw_end_of_response

    def process(self, llm_sentence):
        if isinstance(llm_sentence, tuple) and llm_sentence[0] == MessageTag.END_OF_RESPONSE:
            yield AUDIO_RESPONSE_DONE
            return

        llm_sentence, _saw_end_of_response = self._coalesce_pending_tts_input(llm_sentence)

        if isinstance(llm_sentence, tuple):
            llm_sentence, _ = llm_sentence
        if not llm_sentence:
            llm_sentence = "Hello."

        model_type = self._model_type()
        session_voice = None
        if getattr(self, "runtime_config", None):
            session_voice = self.runtime_config.session.audio.output.voice
        if session_voice:
            if model_type == "custom_voice":
                self.speaker = session_voice
                self.ref_audio = None
            else:
                self.ref_audio = session_voice

        console.print(f"[green]ASSISTANT: {llm_sentence}")

        try:
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
        finally:
            if not getattr(self, "runtime_config", None):
                self.should_listen.set()

    def _mlx_streaming_interval(self):
        return max(1, self.streaming_chunk_size) / MLX_STREAMING_TOKENS_PER_SECOND

    def _process_voice_clone(self, text):
        if self.backend == "mlx":
            if self.xvec_only:
                logger.warning("mlx-audio Qwen3-TTS does not support xvec_only; ignoring it")
            if self.parity_mode:
                logger.info("Qwen3-TTS parity mode is CUDA-specific and is ignored on mlx-audio")

            with MLXLockContext(handler_name="Qwen3TTS", timeout=10.0) as acquired:
                if not acquired:
                    raise TimeoutError("Timed out waiting for MLX lock")
                yield from self._stream(
                    self.model.generate(
                        text=text,
                        ref_audio=self.ref_audio,
                        ref_text=self.ref_text,
                        lang_code=self.language,
                        max_tokens=self.max_new_tokens,
                        verbose=False,
                        stream=True,
                        streaming_interval=self._mlx_streaming_interval(),
                        **self.gen_kwargs,
                    ),
                    label="voice_clone_mlx",
                )
            return

        yield from self._stream(
            self.model.generate_voice_clone_streaming(
                text=text,
                language=self.language,
                ref_audio=self.ref_audio,
                ref_text=self.ref_text,
                xvec_only=self.xvec_only,
                chunk_size=self.streaming_chunk_size,
                max_new_tokens=self.max_new_tokens,
                parity_mode=self.parity_mode,
            ),
            label="voice_clone_parity" if self.parity_mode else "voice_clone",
        )

    def _process_custom_voice(self, text):
        speaker = self._resolve_speaker()
        if not speaker:
            raise ValueError(
                "CustomVoice generation requires a speaker. "
                "Set qwen3_tts_speaker or use a voice-clone model with ref_audio."
            )

        if self.backend == "mlx":
            with MLXLockContext(handler_name="Qwen3TTS", timeout=10.0) as acquired:
                if not acquired:
                    raise TimeoutError("Timed out waiting for MLX lock")
                yield from self._stream(
                    self.model.generate_custom_voice(
                        text=text,
                        speaker=speaker,
                        language=self.language,
                        instruct=self.instruct,
                        max_tokens=self.max_new_tokens,
                        verbose=False,
                        stream=True,
                        streaming_interval=self._mlx_streaming_interval(),
                        **self.gen_kwargs,
                    ),
                    label="custom_voice_mlx",
                )
            return

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
        if self.backend == "mlx":
            with MLXLockContext(handler_name="Qwen3TTS", timeout=10.0) as acquired:
                if not acquired:
                    raise TimeoutError("Timed out waiting for MLX lock")
                yield from self._stream(
                    self.model.generate_voice_design(
                        text=text,
                        instruct=self.instruct,
                        language=self.language,
                        max_tokens=self.max_new_tokens,
                        verbose=False,
                        stream=True,
                        streaming_interval=self._mlx_streaming_interval(),
                        **self.gen_kwargs,
                    ),
                    label="voice_design_mlx",
                )
            return

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
            del self.model
            if self.backend == "mlx":
                try:
                    import mlx.core as mx

                    mx.clear_cache()
                except Exception:
                    pass
            else:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            logger.info("Qwen3-TTS handler cleaned up")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
