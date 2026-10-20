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

logger = logging.getLogger(__name__)
PIPELINE_SAMPLE_RATE = 16000
PHONON_NORMALIZATION_LANGUAGES = frozenset({"en", "fr", "de", "es", "pt", "none"})


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
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self._failed_responses: set[tuple[int | None, str | None, str | None, int | None]] = set()
        # Each cached runtime owns its language-specific text normalizer. Global
        # Torch/MLX device flags do not apply to this native CPU runtime.
        self._model_options = {"config": model_name, "revision": revision, "quant": quant, "device": "cpu"}
        self.model = ptts.TTS(**self._model_options, lang=language)
        self.supported_languages = set(self.model.languages) & (PHONON_NORMALIZATION_LANGUAGES - {"none"})
        if language != "none" and language not in self.supported_languages:
            raise ValueError(f"Phonon checkpoint does not support language {language!r}.")
        self._language_models = {language: self.model}
        self._validate_voice(voice)
        logger.info("Loaded Phonon at %s Hz; voices: %s", self.model.sample_rate, ", ".join(self.model.voices))

    @property
    def min_time_to_debug(self) -> float:
        return 0.1

    def _validate_voice(self, voice: str | None) -> None:
        if voice is not None and voice not in self.model.voices:
            raise ValueError(f"Unknown Phonon voice {voice!r}; available voices: {', '.join(self.model.voices)}")

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

    def _model_for_input(self, tts_input: TTSInput) -> Any:
        # TTSInput snapshots the selection for the whole response, so changing
        # the session cannot change normalization halfway through a reply.
        language = tts_input.tts_language_code or self.language
        if language != self.language and language not in self.supported_languages:
            if tts_input.selected_language not in (None, "auto"):
                raise ValueError(f"Phonon checkpoint does not support language {language!r}.")
            language = self.language
        if language not in self._language_models:
            import ptts

            logger.info("Loading Phonon text normalizer for %s", language)
            self._language_models[language] = ptts.TTS(**self._model_options, lang=language)
        return self._language_models[language]

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
        model = self._model_for_input(tts_input)
        console.print(f"[green]ASSISTANT: {tts_input.text}")
        started_at = perf_counter()
        first_audio = True
        pending = np.empty(0, dtype=np.int16)

        # Closing the context on cancellation, exceptions, or generator.close()
        # stops native generation and joins its worker threads.
        with model.stream(tts_input.text, voice=voice) as audio:
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
                    yield block

            for chunk in audio:
                if self._is_cancelled(generation, tts_input):
                    return
                samples = np.asarray(chunk, dtype=np.float32)
                if first_audio and samples.size:
                    first_audio = False
                    store = getattr(self, "turn_latency_store", None)
                    latency = store.get_response(tts_input.response_key) if store else None
                    if latency is not None:
                        latency.record_tts_ttfa(perf_counter() - started_at)
                converted = resampler.resample_chunk(samples) if resampler is not None else samples
                yield from blocks(converted)
                if self._is_cancelled(generation, tts_input):
                    return

            if self._is_cancelled(generation, tts_input):
                return
            if first_audio:
                raise RuntimeError("Phonon returned no audio")
            if resampler is not None:
                yield from blocks(resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True))
            if pending.size and not self._is_cancelled(generation, tts_input):
                yield np.pad(pending, (0, self.blocksize - pending.size))
