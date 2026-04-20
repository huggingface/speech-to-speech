"""
Qwen3 TTS Handler

- On Apple Silicon: Uses mlx-audio with MLX-converted Qwen3-TTS models.
- On CUDA/CPU: Uses faster-qwen3-tts for low-latency streaming.
"""

from __future__ import annotations

import logging
from pathlib import Path
from sys import platform
import tempfile
from time import perf_counter

import numpy as np
from rich.console import Console

from baseHandler import BaseHandler
from cancel_scope import CancelScope
from pipeline_control import SESSION_END, is_control_message
from pipeline_messages import AUDIO_RESPONSE_DONE, PIPELINE_END, EndOfResponse, TTSInput
from utils.mlx_lock import MLXLockContext

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_MLX_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-6bit"
DEFAULT_REF_TEXT = "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes."
DEFAULT_FASTER_STREAMING_CHUNK_SIZE = 8
DEFAULT_MLX_STREAMING_CHUNK_SIZE = 4
VALID_MLX_QUANTIZATION_SUFFIXES = ("bf16", "4bit", "6bit", "8bit")
MLX_STREAMING_TOKENS_PER_SECOND = 12.5
PIPELINE_SR = 16000


class Qwen3TTSHandler(BaseHandler[TTSInput | EndOfResponse]):
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
        mlx_quantization=None,
        streaming_chunk_size=None,
        max_new_tokens=360,
        blocksize=512,
        gen_kwargs=None,
        cancel_scope: CancelScope | None = None,
    ):
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
        self.mlx_quantization = self._normalize_mlx_quantization(mlx_quantization)
        self.max_new_tokens = max_new_tokens
        self.blocksize = blocksize
        self.dtype = None
        self.gen_kwargs = gen_kwargs or {}
        self._mlx_ref_audio_cache = {}
        self._mlx_temp_ref_audio_files = set()

        self.backend = "mlx" if platform == "darwin" else "faster_qwen3_tts"
        self.streaming_chunk_size = self._resolve_streaming_chunk_size(
            streaming_chunk_size
        )

        if self.backend == "mlx":
            self.device = "mps"
            self.model_name = self._resolve_mlx_model_name(model_name)
            logger.info(
                f"Loading Qwen3-TTS model: {self.model_name} via mlx-audio on Apple Silicon"
            )
            model_quantization = self._model_name_quantization_suffix(self.model_name)
            if model_quantization and model_quantization != "bf16":
                logger.info(
                    "Using MLX quantized Qwen3-TTS variant: %s",
                    model_quantization,
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

        logger.info(
            "Using Qwen3-TTS streaming chunk size %d (~%.0fms audio per chunk) on %s",
            self.streaming_chunk_size,
            self.streaming_chunk_size / MLX_STREAMING_TOKENS_PER_SECOND * 1000,
            self.backend,
        )

        self._initial_speaker = self.speaker
        self._initial_ref_audio = self.ref_audio

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
                    "misaki",
                    "spacy",
                    "phonemizer",
                    "espeakng_loader",
                )
            ):
                raise ImportError(
                    "Qwen3-TTS on Apple Silicon requires mlx-audio and its TTS dependencies. "
                    f"Missing dependency: {message}. "
                    "Install with: pip install mlx-audio misaki spacy phonemizer-fork espeakng-loader"
                ) from e
            raise ImportError(
                "mlx-audio is required for Qwen3 TTS on Apple Silicon. "
                "Install with: pip install mlx-audio"
            ) from e

        logger.info("MLX Audio Qwen3-TTS model loaded")

    def _normalize_mlx_quantization(self, mlx_quantization):
        if mlx_quantization is None:
            return None

        value = str(mlx_quantization).strip().lower()
        if value in ("", "none", "default"):
            return None
        if value not in VALID_MLX_QUANTIZATION_SUFFIXES:
            raise ValueError(
                "Unsupported qwen3_tts_mlx_quantization value "
                f"{mlx_quantization!r}. Supported values: {', '.join(VALID_MLX_QUANTIZATION_SUFFIXES)}"
            )
        return value

    def _apply_mlx_quantization_suffix(self, model_name):
        if self.mlx_quantization is None:
            return model_name

        desired_suffix = f"-{self.mlx_quantization}"
        for suffix in VALID_MLX_QUANTIZATION_SUFFIXES:
            current_suffix = f"-{suffix}"
            if model_name.endswith(current_suffix):
                return model_name[: -len(current_suffix)] + desired_suffix

        return f"{model_name}{desired_suffix}"

    def _model_name_quantization_suffix(self, model_name):
        if not model_name:
            return None

        for suffix in VALID_MLX_QUANTIZATION_SUFFIXES:
            if model_name.endswith(f"-{suffix}"):
                return suffix

        return None

    def _resolve_mlx_model_name(self, model_name):
        if not model_name:
            return self._apply_mlx_quantization_suffix(DEFAULT_MLX_MODEL)
        if model_name.startswith("mlx-community/"):
            if (
                self.mlx_quantization is None
                and self._model_name_quantization_suffix(model_name) is None
            ):
                return f"{model_name}-6bit"
            return self._apply_mlx_quantization_suffix(model_name)
        if model_name.startswith("Qwen/"):
            mapped = model_name.replace("Qwen/", "mlx-community/", 1)
            if self._model_name_quantization_suffix(mapped) is None:
                if self.mlx_quantization is None:
                    return f"{mapped}-6bit"
                mapped = f"{mapped}-bf16"
            return self._apply_mlx_quantization_suffix(mapped)
        return model_name

    def _resolve_streaming_chunk_size(self, streaming_chunk_size):
        if streaming_chunk_size is not None:
            return max(1, int(streaming_chunk_size))
        if self.backend == "mlx":
            return DEFAULT_MLX_STREAMING_CHUNK_SIZE
        return DEFAULT_FASTER_STREAMING_CHUNK_SIZE

    def _infer_model_type_from_name(self):
        name = (self.model_name or "").lower()
        if "voicedesign" in name:
            return "voice_design"
        if "customvoice" in name:
            return "custom_voice"
        return "base"

    def _resolve_audio_path(self, audio):
        if not isinstance(audio, (str, Path)) or not audio:
            return None

        candidate = Path(audio).expanduser()
        repo_root = Path(__file__).resolve().parents[1]
        search_paths = []

        if candidate.is_absolute():
            search_paths.append(candidate)
        else:
            search_paths.append(Path.cwd() / candidate)
            search_paths.append(repo_root / candidate)

        seen = set()
        for path in search_paths:
            normalized = str(path)
            if normalized in seen:
                continue
            seen.add(normalized)
            if path.exists():
                return path.resolve()

        return None

    def _prepare_mlx_ref_audio(self, ref_audio):
        if self.backend != "mlx" or ref_audio is None:
            return ref_audio

        if not isinstance(ref_audio, (str, Path)):
            return ref_audio

        resolved_path = self._resolve_audio_path(ref_audio)
        if resolved_path is None:
            raise FileNotFoundError(
                "Qwen3-TTS on Apple Silicon requires qwen3_tts_ref_audio to point to "
                f"a readable audio file. Got: {ref_audio!r}"
            )

        cache_key = str(resolved_path)
        cached_path = self._mlx_ref_audio_cache.get(cache_key)
        if cached_path and Path(cached_path).exists():
            return cached_path

        try:
            import soundfile as sf
            from scipy.signal import resample_poly

            waveform, sample_rate = sf.read(
                str(resolved_path), always_2d=False, dtype="float32"
            )
            waveform = np.asarray(waveform, dtype=np.float32)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            target_sample_rate = getattr(self.model, "sample_rate", 24000)
            if sample_rate != target_sample_rate:
                gcd = np.gcd(int(sample_rate), int(target_sample_rate))
                waveform = resample_poly(
                    waveform,
                    up=int(target_sample_rate) // gcd,
                    down=int(sample_rate) // gcd,
                )
                sample_rate = target_sample_rate

            with tempfile.NamedTemporaryFile(
                prefix="qwen3_ref_",
                suffix=".wav",
                delete=False,
            ) as temp_file:
                normalized_path = temp_file.name

            sf.write(
                normalized_path,
                waveform,
                sample_rate,
                format="WAV",
                subtype="PCM_16",
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to normalize Qwen3-TTS reference audio {resolved_path}: {e}"
            ) from e

        self._mlx_ref_audio_cache[cache_key] = normalized_path
        self._mlx_temp_ref_audio_files.add(normalized_path)
        return normalized_path

    def _apply_session_voice_override(self, model_type, runtime_config=None, response=None):
        session_voice = None
        if response and response.audio and response.audio.output:
            session_voice = response.audio.output.voice
        if not session_voice and runtime_config is not None:
            session_voice = runtime_config.session.audio.output.voice
        if not session_voice:
            return

        if model_type == "custom_voice":
            self.speaker = session_voice
            self.ref_audio = None
            return

        if self._resolve_audio_path(session_voice) is not None:
            self.ref_audio = session_voice
            return

        logger.warning(
            "Ignoring Qwen3-TTS session voice override because it is not an audio file path: %r",
            session_voice,
        )

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

    def _coalesce_pending_tts_input(self, current_input: TTSInput):
        """Combine already-queued text chunks before the next TTS synthesis call."""
        if not hasattr(self.queue_in, "mutex") or not hasattr(self.queue_in, "queue"):
            return current_input.text, current_input.language_code, False

        text = current_input.text
        language_code = current_input.language_code

        parts = [text.strip()] if text and text.strip() else []
        saw_end_of_response = False

        with self.queue_in.mutex:
            while self.queue_in.queue:
                next_item = self.queue_in.queue[0]
                if is_control_message(next_item, SESSION_END.kind):
                    break
                if isinstance(next_item, bytes) and next_item == PIPELINE_END:
                    break
                if isinstance(next_item, EndOfResponse):
                    saw_end_of_response = True
                    break
                if not isinstance(next_item, TTSInput):
                    break
                if (
                    language_code is not None
                    and next_item.language_code is not None
                    and next_item.language_code != language_code
                ):
                    break

                self.queue_in.queue.popleft()
                if next_item.text.strip():
                    parts.append(next_item.text.strip())
                if language_code is None:
                    language_code = next_item.language_code

        combined_text = " ".join(parts).strip()
        return combined_text, language_code, saw_end_of_response

    def process(self, tts_input: TTSInput | EndOfResponse):
        if isinstance(tts_input, EndOfResponse):
            self.should_listen.set()
            yield AUDIO_RESPONSE_DONE
            return

        runtime_config = tts_input.runtime_config
        response = tts_input.response

        coalesced_text, _language_code, _saw_end_of_response = self._coalesce_pending_tts_input(tts_input)

        text = coalesced_text or "Hello."

        model_type = self._model_type()
        self._apply_session_voice_override(model_type, runtime_config, response)

        console.print(f"[green]ASSISTANT: {text}")

        try:
            if self.ref_audio:
                yield from self._process_voice_clone(text)
            elif model_type == "custom_voice":
                yield from self._process_custom_voice(text)
            elif model_type == "voice_design":
                yield from self._process_voice_design(text)
            else:
                raise ValueError(
                    "Qwen3-TTS Base model requires ref_audio for voice cloning. "
                    "Provide qwen3_tts_ref_audio or use a CustomVoice/VoiceDesign model."
                )
        except Exception as e:
            if not runtime_config:
                self.should_listen.set()
            logger.error(f"Error during Qwen3-TTS generation: {e}", exc_info=True)

    def _mlx_streaming_interval(self):
        return max(1, self.streaming_chunk_size) / MLX_STREAMING_TOKENS_PER_SECOND

    def _mlx_stream_kwargs(self):
        return {
            "max_tokens": self.max_new_tokens,
            "verbose": False,
            "stream": True,
            "streaming_interval": self._mlx_streaming_interval(),
            **self.gen_kwargs,
        }

    def _stream_mlx_generation(self, generation_fn, label, **generation_kwargs):
        with MLXLockContext(handler_name="Qwen3TTS", timeout=10.0) as acquired:
            if not acquired:
                raise TimeoutError("Timed out waiting for MLX lock")
            yield from self._stream(
                generation_fn(
                    **self._mlx_stream_kwargs(),
                    **generation_kwargs,
                ),
                label=label,
            )

    def _process_voice_clone(self, text):
        if self.backend == "mlx":
            if self.xvec_only:
                logger.warning("mlx-audio Qwen3-TTS does not support xvec_only; ignoring it")
            if self.parity_mode:
                logger.info("Qwen3-TTS parity mode is CUDA-specific and is ignored on mlx-audio")

            yield from self._stream_mlx_generation(
                self.model.generate,
                label="voice_clone_mlx",
                text=text,
                ref_audio=self._prepare_mlx_ref_audio(self.ref_audio),
                ref_text=self.ref_text,
                lang_code=self.language,
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
            yield from self._stream_mlx_generation(
                self.model.generate_custom_voice,
                label="custom_voice_mlx",
                text=text,
                speaker=speaker,
                language=self.language,
                instruct=self.instruct,
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
            yield from self._stream_mlx_generation(
                self.model.generate_voice_design,
                label="voice_design_mlx",
                text=text,
                instruct=self.instruct,
                language=self.language,
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

    def on_session_end(self):
        self.speaker = self._initial_speaker
        self.ref_audio = self._initial_ref_audio
        logger.debug("Qwen3-TTS session state reset")

    def cleanup(self):
        try:
            del self.model
            for path in list(getattr(self, "_mlx_temp_ref_audio_files", set())):
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass
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
