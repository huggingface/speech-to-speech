from __future__ import annotations

import logging
from threading import Event
from time import perf_counter
from typing import Any, Iterator, cast

import numpy as np
import soxr

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.console import console
from speech_to_speech.pipeline.events import ResponseFailedEvent
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.pipeline.transcript_logging import log_exception
from speech_to_speech.pipeline.turn_latency import record_response_e2e, record_response_tts_ttfa

logger = logging.getLogger(__name__)
PIPELINE_SAMPLE_RATE = 16000
PHONON_NORMALIZATION_LANGUAGES = frozenset({"en", "fr", "de", "es", "pt", "none"})
OPENAI_REALTIME_VOICES = frozenset(
    {"alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar"}
)


class _TrailingSilenceTrimmer:
    """Keep a short quiet tail; restore every held sample if speech resumes.

    Only samples after observed speech are held, so leading audio is unchanged.
    Every other sample is forwarded by the push that supplies it, even before its
    10 ms window is complete. The pending tail is capped at one second to bound
    memory for abnormal output, so at most one second is removed per utterance.
    """

    def __init__(self, sample_rate: int, keep_ms: int):
        self.frame_size = max(1, sample_rate // 100)
        self.keep_samples = round(sample_rate * keep_ms / 1000)
        self.max_pending = sample_rate
        self.partial = np.empty(0, dtype=np.float32)
        self.forwarded = 0
        self.pending = np.empty(0, dtype=np.float32)
        self.quiet_kept = 0
        self.saw_speech = False

    def _keep_budget(self) -> int:
        return max(0, self.keep_samples - self.quiet_kept)

    def _frame(self, samples: np.ndarray) -> Iterator[np.ndarray]:
        # push() may already have forwarded the start of this window.
        forwarded, self.forwarded = self.forwarded, 0
        quiet = float(np.mean(samples.astype(np.float64) ** 2)) <= 1e-6  # -60 dBFS RMS.
        if not quiet or not self.saw_speech:
            if self.pending.size:
                yield self.pending
                self.pending = np.empty(0, dtype=np.float32)
            self.quiet_kept = 0
            self.saw_speech |= not quiet
            if forwarded < samples.size:
                yield samples[forwarded:]
            return
        keep = min(samples.size, self._keep_budget())
        if keep > forwarded:
            yield samples[forwarded:keep]
        self.quiet_kept += keep
        self.pending = np.concatenate((self.pending, samples[keep:]))
        excess = self.pending.size - self.max_pending
        if excess > 0:
            yield self.pending[:excess]
            self.pending = self.pending[excess:].copy()

    def push(self, samples: np.ndarray) -> Iterator[np.ndarray]:
        self.partial = np.concatenate((self.partial, samples))
        end = self.partial.size - self.partial.size % self.frame_size
        for start in range(0, end, self.frame_size):
            yield from self._frame(self.partial[start : start + self.frame_size])
        self.partial = self.partial[end:].copy()
        # The incomplete window is kept whether or not it turns out quiet when it
        # fits the remaining tail budget (nothing is pending then), so forward it.
        ready = self.partial.size if not self.saw_speech else min(self.partial.size, self._keep_budget())
        if ready > self.forwarded:
            yield self.partial[self.forwarded : ready]
            self.forwarded = ready

    def finish(self) -> Iterator[np.ndarray]:
        if self.partial.size:
            yield from self._frame(self.partial)
        self.partial = np.empty(0, dtype=np.float32)
        self.pending = np.empty(0, dtype=np.float32)


class PhononTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """Stream a supplied Phonon checkpoint through the native CPU runtime."""

    def setup(
        self,
        should_listen: Event,
        model_name: str | None = None,
        revision: str | None = None,
        voice: str | None = None,
        language: str = "en",
        quant: str = "q8",
        blocksize: int = 512,
        num_threads: int | None = None,
        max_trailing_silence_ms: int | None = None,
        gen_kwargs: dict[str, Any] | None = None,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError(
                "Phonon requires a checkpoint: set --phonon_tts_model_name to its directory or Hugging Face repo ID."
            )
        if blocksize <= 0:
            raise ValueError("blocksize must be positive")
        if num_threads is not None and num_threads <= 0:
            raise ValueError("num_threads must be positive")
        if max_trailing_silence_ms is not None and max_trailing_silence_ms < 40:
            raise ValueError("max_trailing_silence_ms must be at least 40, or omitted to disable trimming")
        language = language.strip().lower()
        if language not in PHONON_NORMALIZATION_LANGUAGES:
            raise ValueError(
                f"Unsupported Phonon normalization language {language!r}; choose en, fr, de, es, pt, or none."
            )

        import ptts

        if num_threads is not None:
            # The native pool is process-wide and is sized before the first model load.
            ptts.set_num_threads(num_threads)

        self.should_listen = should_listen
        self.voice = voice
        self.language = language
        self.blocksize = blocksize
        self.max_trailing_silence_ms = max_trailing_silence_ms
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self._failed_responses: set[tuple[int | None, str | None, str | None, int | None]] = set()
        # The published wheels run on CPU; global Torch/MLX device flags do
        # not apply to this Rust runtime. Keep it loaded across utterances.
        self.model = ptts.TTS(config=model_name, revision=revision, lang=language, quant=quant, device="cpu")
        self.voices = self._selectable_voices(ptts)
        self._validate_voice(voice)
        logger.info("Loaded Phonon at %s Hz with %d voices", self.model.sample_rate, len(self.voices))

    @property
    def min_time_to_debug(self) -> float:
        return 0.1

    def _selectable_voices(self, ptts: Any) -> list[str]:
        """Checkpoint voices, then catalog voices in the checkpoint's languages."""
        voices = list(self.model.voices)
        try:
            # Only the catalog index downloads here; ptts fetches a voice on first use.
            catalog = ptts.catalog_voices()
        except Exception as exc:
            logger.warning("Phonon voice catalog unavailable; using checkpoint voices only: %s", exc)
            return voices
        languages = set(self.model.languages)
        voices += [
            entry["name"] for entry in catalog if entry.get("language") in languages and entry["name"] not in voices
        ]
        return voices

    def _validate_voice(self, voice: str | None) -> None:
        if voice is not None and voice not in self.voices:
            raise ValueError(f"Unknown Phonon voice {voice!r}; available voices: {', '.join(self.voices)}")

    def _resolve_voice(self, tts_input: TTSInput) -> str | None:
        voice: Any = self.voice
        response = tts_input.response
        if response and response.audio and response.audio.output and response.audio.output.voice:
            voice = response.audio.output.voice
        elif tts_input.runtime_config is not None:
            audio = tts_input.runtime_config.session.audio
            output = audio.output if audio is not None else None
            if output is not None and output.voice:
                voice = output.voice
        if voice is not None and not isinstance(voice, str):
            raise ValueError("Phonon voice overrides must be checkpoint voice names")
        if voice in OPENAI_REALTIME_VOICES and voice not in self.voices:
            # Realtime clients may only offer OpenAI voices. Keep the configured
            # Phonon voice, or let the checkpoint choose its default.
            voice = self.voice
        self._validate_voice(voice)
        return voice

    def _is_cancelled(self, generation: int | None, tts_input: TTSInput) -> bool:
        return (
            self.stop_event.is_set()
            or (generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation))
            or (
                self.speculative_turns is not None
                and not self.speculative_turns.is_latest(tts_input.turn_id, tts_input.turn_revision)
            )
        )

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        tracker = self.speculative_turns
        identity = (tts_input.cancel_generation, tts_input.response_key, tts_input.turn_id, tts_input.turn_revision)
        if isinstance(tts_input, EndOfResponse):
            self._failed_responses.discard(identity)
            if tracker and not tracker.wait_for_gate(tts_input.turn_id, tts_input.turn_revision):
                if tts_input.response_key is None:
                    return
                tts_input.cleanup_only = True
            yield AUDIO_RESPONSE_DONE
            return

        if tracker and not tracker.wait_for_gate(tts_input.turn_id, tts_input.turn_revision, commit=True):
            return
        generation = tts_input.cancel_generation
        if generation is None and self.cancel_scope is not None:
            generation = self.cancel_scope.generation
        if (
            identity in self._failed_responses
            or not tts_input.text.strip()
            or self._is_cancelled(generation, tts_input)
        ):
            return

        try:
            yield from self._synthesize(tts_input, generation)
        except Exception as exc:
            log_exception(logger, "Phonon speech generation failed", exc)
            if not self._is_cancelled(generation, tts_input):
                self._failed_responses.add(identity)
                self.queue_out.put(
                    cast(
                        TTSOut,
                        ResponseFailedEvent(
                            message="Phonon speech generation failed",
                            turn_id=tts_input.turn_id,
                            turn_revision=tts_input.turn_revision,
                            cancel_generation=generation,
                            response_key=tts_input.response_key,
                        ),
                    )
                )

    def on_session_end(self) -> None:
        self._failed_responses.clear()

    def _synthesize(self, tts_input: TTSInput, generation: int | None) -> Iterator[TTSOut]:
        voice = self._resolve_voice(tts_input)
        console.print(f"[green]ASSISTANT: {tts_input.text}")
        started_at = perf_counter()
        first_audio = True
        first_block = True
        pending = np.empty(0, dtype=np.int16)
        tail_ms = self.max_trailing_silence_ms
        trimmer = _TrailingSilenceTrimmer(PIPELINE_SAMPLE_RATE, tail_ms) if tail_ms is not None else None

        def emit(block: np.ndarray) -> np.ndarray:
            nonlocal first_block
            if first_block:
                first_block = False
                if tts_input.speech_stopped_at_s is not None:
                    record_response_e2e(
                        getattr(self, "turn_latency_store", None),
                        tts_input.response_key,
                        max(0.0, perf_counter() - tts_input.speech_stopped_at_s),
                    )
            return block

        # Closing the context on cancellation, exceptions, or generator.close()
        # stops native generation and joins its worker threads.
        with self.model.stream(tts_input.text, voice=voice) as audio:
            source_rate = audio.sample_rate
            resampler = (
                soxr.ResampleStream(source_rate, PIPELINE_SAMPLE_RATE, 1, dtype="float32", quality="HQ")
                if source_rate != PIPELINE_SAMPLE_RATE
                else None
            )

            def blocks(samples: np.ndarray) -> Iterator[np.ndarray]:
                nonlocal pending
                pcm = np.clip(samples * 32768, -32768, 32767).astype(np.int16)
                pending = np.concatenate((pending, pcm))
                while pending.size >= self.blocksize:
                    if self._is_cancelled(generation, tts_input):
                        return
                    block = pending[: self.blocksize]
                    pending = pending[self.blocksize :]
                    yield emit(block)

            def trimmed_blocks(samples: np.ndarray) -> Iterator[np.ndarray]:
                if trimmer is None:
                    yield from blocks(samples)
                    return
                for retained in trimmer.push(samples):
                    yield from blocks(retained)

            for chunk in audio:
                if self._is_cancelled(generation, tts_input):
                    return
                samples = np.asarray(chunk, dtype=np.float32)
                if first_audio and samples.size:
                    first_audio = False
                    record_response_tts_ttfa(
                        getattr(self, "turn_latency_store", None),
                        tts_input.response_key,
                        perf_counter() - started_at,
                    )
                converted = resampler.resample_chunk(samples) if resampler is not None else samples
                yield from trimmed_blocks(converted)
                if self._is_cancelled(generation, tts_input):
                    return

            if self._is_cancelled(generation, tts_input):
                return
            if first_audio:
                raise RuntimeError("Phonon returned no audio")
            if resampler is not None:
                yield from trimmed_blocks(resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True))
            if trimmer is not None:
                for retained in trimmer.finish():
                    yield from blocks(retained)
            if pending.size and not self._is_cancelled(generation, tts_input):
                yield emit(np.pad(pending, (0, self.blocksize - pending.size)))
