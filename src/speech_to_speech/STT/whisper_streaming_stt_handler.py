"""
Whisper Streaming Speech-to-Text Handler

Progressive-streaming STT built on a transformers Whisper pipeline
(default: openai/whisper-large-v3-turbo). Word-level timestamps let the
SmartProgressiveStreamingHandler freeze completed sentences so already
transcribed audio is never re-transcribed, and the final pass only has to
decode the remaining tail of audio.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from sys import platform
from threading import Lock
from time import perf_counter
from typing import Any, Iterator, Optional, cast

import numpy as np
import torch
from rich.console import Console
from rich.text import Text
from transformers import pipeline

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler
from speech_to_speech.STT.smart_progressive_streaming import (
    PartialTranscription as ProgressiveStreamPartial,
)
from speech_to_speech.STT.smart_progressive_streaming import (
    SmartProgressiveStreamingHandler,
)

logger = logging.getLogger(__name__)
console = Console()

# Priority languages: Swiss national languages (Romansh is not supported by
# Whisper) + English + Turkish. All are covered by NLTK punkt for sentence
# segmentation in the progressive streaming handler.
SUPPORTED_LANGUAGES = [
    "en",
    "de",
    "fr",
    "it",
    "tr",
]


class WhisperStreamingSTTHandler(BaseSTTHandler):
    """
    Handles Speech-to-Text using a transformers Whisper pipeline with
    progressive (live) streaming.

    Progressive audio updates are transcribed with word-level timestamps so
    completed sentences can be frozen; the final transcription stitches the
    frozen sentences with a decode of only the remaining audio tail.
    """

    def setup(
        self,
        model_name: str = "openai/whisper-large-v3-turbo",
        device: str = "auto",
        torch_dtype: str = "float16",
        language: Optional[str] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        live_transcription_update_interval: float = 0.5,
        max_window_size: float = 15.0,
        sentence_buffer: float = 2.0,
    ) -> None:
        """
        Initialize the Whisper streaming model.

        Args:
            model_name: Hugging Face model identifier
            device: Device to use ("auto", "cuda", "mps", "cpu")
            torch_dtype: Compute precision ("float16", "float32")
            language: Target language code (None or "auto" lets Whisper detect)
            gen_kwargs: Extra generate kwargs forwarded to the pipeline
            live_transcription_update_interval: Emission interval for partials
            max_window_size: Maximum re-transcription window before freezing
                completed sentences (seconds)
            sentence_buffer: Trailing seconds of sentences kept active when
                the window slides
        """
        self.gen_kwargs = dict(gen_kwargs or {})
        self.start_language = language
        self._fixed_language = language if language and language != "auto" else None
        self.last_language = self._fixed_language
        self.live_transcription_update_interval = live_transcription_update_interval
        self.compute_lock = Lock()
        self.sample_rate = 16000

        if device == "auto":
            if platform == "darwin" and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.torch_dtype = getattr(torch, torch_dtype)
        if self.device == "cpu":
            self.torch_dtype = torch.float32

        self.model_name = model_name
        logger.info(f"Loading Whisper streaming model: {model_name} on {self.device}")

        self.asr_pipeline = cast(
            Any,
            pipeline(
                "automatic-speech-recognition",
                model=model_name,
                torch_dtype=self.torch_dtype,
                device=self.device,
            ),
        )

        self.streaming_handler = SmartProgressiveStreamingHandler(
            self.asr_pipeline,
            emission_interval=self.live_transcription_update_interval,
            max_window_size=max_window_size,
            sentence_buffer=sentence_buffer,
            language=self._fixed_language,
            gen_kwargs=self.gen_kwargs,
        )
        self.processing_final = False  # Track if we're processing final audio
        self._live_transcription_active = False
        self._live_turn_key: tuple[str | None, int | None] | None = None

        self.warmup()

    def warmup(self) -> None:
        """Warm up the model with a dummy input."""
        logger.info(f"Warming up {self.__class__.__name__}")

        dummy_audio = np.zeros(self.sample_rate, dtype=np.float32)
        try:
            _ = self.asr_pipeline(
                {"raw": dummy_audio, "sampling_rate": self.sample_rate},
                return_timestamps="word",
            )
            logger.info("Model warmed up and ready")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        """
        Process audio and generate transcription.

        Yields:
            :class:`PartialTranscription` or :class:`Transcription`
        """
        process_start_s = perf_counter()
        is_progressive = vad_audio.mode == "progressive"
        audio_input = vad_audio.audio

        # Ensure audio is float32 numpy array
        if not isinstance(audio_input, np.ndarray):
            audio_input = np.array(audio_input, dtype=np.float32)
        else:
            audio_input = audio_input.astype(np.float32)
        audio_duration_s = len(audio_input) / self.sample_rate
        item_age_s = self._item_age_s(vad_audio)

        self._prepare_live_transcription_turn(vad_audio.turn_id, vad_audio.turn_revision)

        # Handle progressive updates: yield tagged partial for TranscriptionNotifier
        if is_progressive:
            # Ignore progressive updates if we're already processing final audio
            if self.processing_final:
                logger.debug("Skipping stale progressive update (final audio already received)")
                return

            # Try to acquire lock with short timeout - skip if busy
            lock_scope_start_s = perf_counter()
            with self._compute_lock_context(handler_name="WhisperSTT-Progressive", timeout=0.01) as acquired:
                if acquired:
                    try:
                        inference_start_s = perf_counter()
                        progressive_text = self._show_progressive_transcription(audio_input)
                        inference_s = perf_counter() - inference_start_s
                        if inference_s >= 0.25:
                            logger.info(
                                "Whisper progressive STT timing turn=%s rev=%s audio=%.3fs age=%.3fs "
                                "lock_scope=%.3fs inference=%.3fs chars=%d",
                                vad_audio.turn_id,
                                vad_audio.turn_revision,
                                audio_duration_s,
                                item_age_s,
                                perf_counter() - lock_scope_start_s,
                                inference_s,
                                len(progressive_text),
                            )
                        if progressive_text:
                            yield PartialTranscription(
                                text=progressive_text,
                                turn_id=vad_audio.turn_id,
                                turn_revision=vad_audio.turn_revision,
                            )
                            return
                    except Exception as e:
                        logger.debug(f"Progressive transcription failed: {e}")
                else:
                    logger.debug("Skipping progressive update (compute busy)")
            return

        # Handle final transcription (send to LLM)
        logger.info(
            "Whisper final STT start turn=%s rev=%s audio=%.3fs age=%.3fs",
            vad_audio.turn_id,
            vad_audio.turn_revision,
            audio_duration_s,
            item_age_s,
        )
        inference_s = 0.0
        lock_scope_s = 0.0
        try:
            # Mark that we're processing final audio (ignore stale progressive updates)
            self.processing_final = True

            # Acquire lock with longer timeout for final transcription
            lock_scope_start_s = perf_counter()
            with self._compute_lock_context(handler_name="WhisperSTT-Final", timeout=5.0) as acquired:
                lock_scope_s = perf_counter() - lock_scope_start_s
                if not acquired:
                    logger.error("Failed to acquire compute lock for final transcription")
                    pred_text = ""
                    language_code = self.last_language
                else:
                    inference_start_s = perf_counter()
                    pred_text, language_code = self._transcribe_final(audio_input)
                    inference_s = perf_counter() - inference_start_s
                    lock_scope_s = perf_counter() - lock_scope_start_s

            # Validate and update language
            if language_code and language_code in SUPPORTED_LANGUAGES:
                self.last_language = language_code
            else:
                language_code = self.last_language

        except Exception as e:
            logger.error(f"Whisper streaming inference failed: {e}")
            pred_text = ""
            language_code = self.last_language

        total_s = perf_counter() - process_start_s
        logger.info(
            "Whisper final STT done turn=%s rev=%s total=%.3fs lock_scope=%.3fs inference=%.3fs chars=%d",
            vad_audio.turn_id,
            vad_audio.turn_revision,
            total_s,
            lock_scope_s,
            inference_s,
            len(pred_text),
        )
        logger.debug("Finished Whisper streaming inference")
        self._clear_live_transcription_line()
        if pred_text.strip():
            console.print(f"[yellow]USER: {pred_text.strip()}")
            if language_code:
                console.print(f"[dim]Language: {language_code}[/dim]")

        # Reset per-utterance live transcription state only after final STT
        # completes. The streaming handler carries fixed sentence timing within
        # an utterance, and stale timing must not leak into the next turn.
        self.processing_final = False
        self._reset_live_transcription_state(clear_turn=True)

        if language_code and self.start_language == "auto":
            language_code += "-auto"

        yield Transcription(
            text=pred_text,
            language_code=language_code,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )

    @property
    def timing_log_level(self) -> int:
        return logging.INFO

    def should_log_timing(self, output: STTOut) -> bool:
        return isinstance(output, Transcription) and self.last_time > self.min_time_to_debug

    def _detect_language_code(self, audio_input: np.ndarray) -> Optional[str]:
        """Detect the spoken language from audio with Whisper's language head.

        A single encoder pass + one decoder step; run once per turn. The
        pipeline's ``return_language`` output is not usable here: transformers
        5.x strips the leading special tokens (incl. the language token) from
        ``generate`` output, so the decoded language always comes back None.
        """
        try:
            model: Any = self.asr_pipeline.model
            feature_extractor: Any = self.asr_pipeline.feature_extractor
            tokenizer: Any = self.asr_pipeline.tokenizer
            features = feature_extractor(
                audio_input,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            ).input_features.to(model.device, dtype=model.dtype)
            language_ids = model.detect_language(input_features=features)
            token = tokenizer.decode(language_ids)
            return str(token)[2:-2]  # remove "<|" and "|>"
        except Exception as e:
            logger.warning(f"Whisper language detection failed: {e}")
            return None

    def _decode(self, audio_input: np.ndarray) -> str:
        """Decode audio to text with the Whisper pipeline.

        Language is only forced when the user configured one; otherwise every
        decode auto-detects so code-switching speech (e.g. an interpreter
        alternating languages every sentence) is transcribed, not translated.
        """
        generate_kwargs = dict(self.gen_kwargs)
        if self._fixed_language:
            generate_kwargs["language"] = self._fixed_language

        result: dict[str, Any] = self.asr_pipeline(
            {"raw": audio_input, "sampling_rate": self.sample_rate},
            # Required for audio > 30s (long-form generation); harmless below
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
        )
        return result.get("text", "").strip()

    def _transcribe_final(self, audio_input: np.ndarray) -> tuple[str, Optional[str]]:
        """Final transcription: stitch frozen sentences with the audio tail.

        The returned language code is report-only (dominant language of the
        turn); it is never fed back into generation.
        """
        language_code = self._fixed_language
        if language_code is None:
            detected = self._detect_language_code(audio_input)
            if detected in SUPPORTED_LANGUAGES:
                language_code = detected
            else:
                if detected is not None:
                    logger.warning("Whisper detected unsupported language: %s", detected)
                language_code = self.last_language

        if not self.streaming_handler.fixed_sentences:
            # No frozen sentences yet, transcribe everything
            return self._decode(audio_input), language_code

        self._clear_live_transcription_line()

        fixed_text = " ".join(self.streaming_handler.fixed_sentences).strip()
        fixed_end_sample = int(self.streaming_handler.fixed_end_time * self.sample_rate)

        if fixed_end_sample > len(audio_input):
            logger.warning(
                "Ignoring stale progressive fixed text: fixed_end_sample=%d exceeds final audio samples=%d",
                fixed_end_sample,
                len(audio_input),
            )
            return self._decode(audio_input), language_code

        if fixed_end_sample < len(audio_input):
            # Only transcribe the part after the frozen sentences
            tail_text = self._decode(audio_input[fixed_end_sample:])
            pred_text = f"{fixed_text} {tail_text}".strip() if tail_text else fixed_text
        else:
            # All audio already transcribed in progressive updates
            pred_text = fixed_text

        return pred_text, language_code

    @contextmanager
    def _compute_lock_context(self, handler_name: str, timeout: float) -> Iterator[bool]:
        lock_start_s = perf_counter()
        acquired = self.compute_lock.acquire(timeout=timeout)
        wait_s = perf_counter() - lock_start_s
        hold_start_s: float | None = None
        if acquired:
            if wait_s >= 0.25:
                logger.info("%s: compute lock acquired after %.2fs", handler_name, wait_s)
            else:
                logger.debug("%s: compute lock acquired after %.3fs", handler_name, wait_s)
            hold_start_s = perf_counter()
        else:
            logger.warning("%s: Failed to acquire compute lock after %.3fs (timeout=%s)", handler_name, wait_s, timeout)
        try:
            yield acquired
        finally:
            if acquired:
                assert hold_start_s is not None
                self.compute_lock.release()
                hold_s = perf_counter() - hold_start_s
                if hold_s >= 0.25:
                    logger.info("%s: compute lock released after holding %.2fs", handler_name, hold_s)
                else:
                    logger.debug("%s: compute lock released after holding %.3fs", handler_name, hold_s)

    def _show_progressive_transcription(self, audio_input: np.ndarray) -> str:
        """Run progressive transcription, print to console, and return the text."""
        result = self.streaming_handler.transcribe_incremental(audio_input)
        rich_text = Text()
        if result.fixed_text:
            rich_text.append("Live: ", style="dim")
            rich_text.append(result.fixed_text, style="yellow")
            if result.active_text:
                rich_text.append(" ", style="dim")

        if result.active_text:
            if not result.fixed_text:
                rich_text.append("Live: ", style="dim")
            rich_text.append(result.active_text, style="cyan dim")

        progressive_text = self._build_progressive_text(result)
        if progressive_text:
            self._print_live_transcription(rich_text, progressive_text)

        return progressive_text

    def _print_live_transcription(self, rich_text: Text, progressive_text: str) -> None:
        is_terminal = bool(getattr(console, "is_terminal", False))
        if is_terminal:
            self._write_live_control("\r\x1b[2K")
            if rich_text:
                console.print(self._truncate_live_transcription(rich_text), end="")
            else:
                fallback = Text("Live: ", style="dim")
                fallback.append(progressive_text, style="cyan dim")
                console.print(self._truncate_live_transcription(fallback), end="")
            self._write_live_control("\r")
            self._live_transcription_active = True
            return

        if rich_text:
            console.print(rich_text)
        else:
            console.print(f"[dim]Live: [/dim]{progressive_text}")

    def _clear_live_transcription_line(self) -> None:
        if not getattr(self, "_live_transcription_active", False):
            return
        self._write_live_control("\r\x1b[2K")
        self._live_transcription_active = False

    def _write_live_control(self, sequence: str) -> None:
        file = getattr(console, "file", None)
        if file is None:
            return
        file.write(sequence)
        file.flush()

    def _truncate_live_transcription(self, text: Text) -> Text:
        text = text.copy()
        width = getattr(console, "width", 80)
        try:
            max_width = max(1, int(width) - 1)
        except (TypeError, ValueError):
            max_width = 79
        text.truncate(max_width, overflow="ellipsis")
        return text

    def _prepare_live_transcription_turn(self, turn_id: str | None, turn_revision: int | None) -> None:
        turn_key = (turn_id, turn_revision)
        if getattr(self, "_live_turn_key", None) == turn_key:
            return
        self._reset_live_transcription_state(clear_turn=False)
        self._live_turn_key = turn_key

    def _reset_live_transcription_state(self, clear_turn: bool) -> None:
        self._clear_live_transcription_line()
        self.streaming_handler.reset()
        if clear_turn:
            self._live_turn_key = None

    def _build_progressive_text(self, result: ProgressiveStreamPartial) -> str:
        parts = []
        if result.fixed_text:
            parts.append(result.fixed_text.strip())
        if result.active_text:
            parts.append(result.active_text.strip())
        return " ".join(part for part in parts if part).strip()

    def cleanup(self) -> None:
        """Clean up model resources."""
        logger.info(f"Cleaning up {self.__class__.__name__}")
        if hasattr(self, "asr_pipeline"):
            del self.asr_pipeline

    def on_session_end(self) -> None:
        super().on_session_end()
        self.last_language = self._fixed_language
        self.processing_final = False
        self._reset_live_transcription_state(clear_turn=True)
        logger.debug("Whisper streaming session state reset")
