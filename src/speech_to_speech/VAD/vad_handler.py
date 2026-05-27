from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from queue import Queue
from threading import Event
from typing import Any, TypeAlias

import numpy as np
import torch

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.events import SpeechStartedEvent, SpeechStoppedEvent
from speech_to_speech.pipeline.handler_types import VADIn, VADOut
from speech_to_speech.pipeline.messages import VADAudio
from speech_to_speech.pipeline.queue_types import TextEventItem
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.utils.utils import int2float
from speech_to_speech.VAD.vad_iterator import VADIterator

logger = logging.getLogger(__name__)

VADInput: TypeAlias = bytes | tuple[bytes, RuntimeConfig]

# Optional import for audio enhancement
try:
    from df.enhance import enhance, init_df

    HAS_DF = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_DF = False
    logger.warning(f"DeepFilterNet not available for audio enhancement: {e}")


class VADHandler(BaseHandler[VADIn, VADOut]):
    """
    Handles voice activity detection. When voice activity is detected, audio will be accumulated until the end of speech is detected and then passed
    to the following part.
    """

    def setup(
        self,
        should_listen: Event,
        thresh: float = 0.6,
        sample_rate: int = 16000,
        min_silence_ms: int = 300,
        min_speech_ms: int = 384,
        max_speech_ms: float = float("inf"),
        speech_pad_ms: int = 30,
        audio_enhancement: bool = False,
        enable_realtime_transcription: bool = False,
        realtime_processing_pause: float = 0.5,
        text_output_queue: Queue[TextEventItem] | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        speculative_reopen_ms: int = 1000,
    ) -> None:
        self.should_listen = should_listen
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.enable_realtime_transcription = enable_realtime_transcription
        self.realtime_processing_pause = realtime_processing_pause
        self.text_output_queue = text_output_queue
        self.speculative_turns = speculative_turns
        self.speculative_reopen_ms = speculative_reopen_ms
        self._last_turn_detection: dict | None = None
        self.model, _ = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            trust_repo=True,
            skip_validation=True,
        )
        self.iterator = VADIterator(
            self.model,
            threshold=thresh,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self.audio_enhancement = audio_enhancement
        if audio_enhancement:
            if not HAS_DF:
                logger.error(
                    "Audio enhancement requested but DeepFilterNet is not available. Disabling audio enhancement."
                )
                self.audio_enhancement = False
            else:
                self.enhanced_model, self.df_state, _ = init_df()

        # State for progressive audio release
        self.last_process_time: float = 0.0

        # Cumulative sample counter for audio_start_ms / audio_end_ms
        self._total_samples: int = 0

        # Throttled logging state (summary once per second)
        self._last_log_time = 0.0
        self._log_chunks = 0
        self._log_speech_starts = 0
        self._log_speech_ends = 0
        self._log_progressive_yields = 0
        self._speech_started_emitted = False
        self._turn_counter = 0
        self._current_turn_id: str | None = None
        self._current_turn_revision: int | None = None
        self._speculative_audio_prefix: np.ndarray | None = None
        self._last_final_wall_time: float | None = None
        self._last_final_audio_ms: int | None = None
        self._pending_reopen_candidate: tuple[str, int, int] | None = None

    @property
    def _audio_ms(self) -> int:
        """Cumulative audio received so far, in milliseconds."""
        return int(self._total_samples / self.sample_rate * 1000)

    def _apply_runtime_turn_detection(self, runtime_config: RuntimeConfig | None = None) -> None:
        """Check RuntimeConfig for turn_detection changes and apply them."""
        audio = runtime_config.session.audio if runtime_config else None
        audio_input = audio.input if audio is not None else None
        if not runtime_config or not audio_input or not audio_input.turn_detection:
            return
        td_raw = audio_input.turn_detection

        # Convert Pydantic models (e.g. OpenAI SDK ServerVad) to dict;
        # plain dicts pass through unchanged.
        if hasattr(td_raw, "model_dump"):
            td = td_raw.model_dump(exclude_none=True)
        elif isinstance(td_raw, dict):
            td = td_raw
        else:
            logger.warning(f"Unexpected turn_detection type: {type(td_raw)}")
            return

        # Compare normalized snapshot (identity on td_raw vs stored dict was wrong after first apply).
        if td == self._last_turn_detection:
            return

        self._last_turn_detection = dict(td)

        if "threshold" in td:
            self.iterator.threshold = td["threshold"]
            logger.info(f"VAD threshold updated to {td['threshold']}")
        if "silence_duration_ms" in td:
            self.iterator.min_silence_samples = self.sample_rate * td["silence_duration_ms"] / 1000
            logger.info(f"VAD silence duration updated to {td['silence_duration_ms']}ms")

    def _start_new_turn(self) -> tuple[str, int]:
        self._cancel_pending_reopen()
        self._turn_counter += 1
        self._current_turn_id = f"turn_{self._turn_counter}"
        self._current_turn_revision = 0
        self._speculative_audio_prefix = None
        self._last_final_wall_time = None
        self._last_final_audio_ms = None
        if self.speculative_turns:
            self.speculative_turns.observe(self._current_turn_id, self._current_turn_revision)
        return self._current_turn_id, self._current_turn_revision

    def _speech_buffer_duration_ms(self) -> float:
        if not hasattr(self.iterator, "speech_buffer"):
            return 0.0
        buffer_samples = sum(len(t) for t in self.iterator.speech_buffer())
        return buffer_samples / self.sample_rate * 1000

    def _current_active_speech_duration_ms(self) -> float:
        active_speech_samples = getattr(self.iterator, "active_speech_samples", 0)
        return active_speech_samples / self.sample_rate * 1000

    def _last_utterance_active_speech_duration_ms(self) -> float:
        active_speech_samples = getattr(self.iterator, "last_utterance_active_speech_samples", 0)
        return active_speech_samples / self.sample_rate * 1000

    def _uses_realtime_turn_handling(self) -> bool:
        return self.enable_realtime_transcription or self.speculative_turns is not None

    def _should_reopen_current_turn(self, audio_start_ms: int) -> bool:
        if not self._uses_realtime_turn_handling():
            return False
        if self._current_turn_id is None or self._current_turn_revision is None or self._last_final_audio_ms is None:
            return False
        elapsed_ms = max(0, audio_start_ms - self._last_final_audio_ms)
        if elapsed_ms > self.speculative_reopen_ms:
            return False
        if self.speculative_turns and self.speculative_turns.is_committed(
            self._current_turn_id,
            self._current_turn_revision,
        ):
            return False
        return True

    def _begin_pending_reopen_if_needed(self, audio_start_ms: int) -> None:
        if self._pending_reopen_candidate is not None or not self._should_reopen_current_turn(audio_start_ms):
            return
        if self.speculative_turns is None:
            return
        candidate_revision = self.speculative_turns.begin_reopen_candidate(
            self._current_turn_id,
            self._current_turn_revision,
        )
        if candidate_revision is None or self._current_turn_id is None or self._current_turn_revision is None:
            return
        self._pending_reopen_candidate = (
            self._current_turn_id,
            self._current_turn_revision,
            candidate_revision,
        )
        logger.info(
            "VAD: pending reopen candidate for speculative turn %s revision %d",
            self._current_turn_id,
            candidate_revision,
        )

    def _cancel_pending_reopen(self) -> None:
        if self._pending_reopen_candidate is None:
            return
        turn_id, _base_revision, candidate_revision = self._pending_reopen_candidate
        if self.speculative_turns:
            self.speculative_turns.cancel_reopen_candidate(turn_id, candidate_revision)
        self._pending_reopen_candidate = None

    def _confirm_pending_reopen(self) -> tuple[str, int, bool] | None:
        if self._pending_reopen_candidate is None:
            return None
        turn_id, base_revision, candidate_revision = self._pending_reopen_candidate
        self._pending_reopen_candidate = None
        if self.speculative_turns and not self.speculative_turns.confirm_reopen_candidate(
            turn_id,
            base_revision,
            candidate_revision,
        ):
            return None
        self._current_turn_id = turn_id
        self._current_turn_revision = candidate_revision
        logger.info("VAD: reopened speculative turn %s revision %d", turn_id, candidate_revision)
        return turn_id, candidate_revision, True

    def _reopen_current_turn(self) -> tuple[str, int, bool] | None:
        if self._current_turn_id is None or self._current_turn_revision is None:
            return None

        turn_id = self._current_turn_id
        base_revision = self._current_turn_revision
        if self.speculative_turns is not None:
            candidate_revision = self.speculative_turns.begin_reopen_candidate(turn_id, base_revision)
            if candidate_revision is None or not self.speculative_turns.confirm_reopen_candidate(
                turn_id,
                base_revision,
                candidate_revision,
            ):
                return None
        else:
            candidate_revision = base_revision + 1

        self._current_turn_id = turn_id
        self._current_turn_revision = candidate_revision
        logger.info("VAD: reopened speculative turn %s revision %d", turn_id, candidate_revision)
        return turn_id, candidate_revision, True

    def _ensure_turn_for_speech_start(self, audio_start_ms: int) -> tuple[str, int, bool]:
        if (
            self._speech_started_emitted
            and self._current_turn_id is not None
            and self._current_turn_revision is not None
        ):
            return self._current_turn_id, self._current_turn_revision, False

        confirmed_reopen = self._confirm_pending_reopen()
        if confirmed_reopen is not None:
            return confirmed_reopen

        reopened = False
        if self._should_reopen_current_turn(audio_start_ms):
            reopened_turn = self._reopen_current_turn()
            if reopened_turn is not None:
                return reopened_turn

        self._start_new_turn()

        if self._current_turn_id is None or self._current_turn_revision is None:
            raise RuntimeError("VAD failed to allocate turn metadata")
        if self.speculative_turns:
            self.speculative_turns.observe(self._current_turn_id, self._current_turn_revision)
        return self._current_turn_id, self._current_turn_revision, reopened

    def _current_turn_metadata(self) -> tuple[str | None, int | None]:
        return self._current_turn_id, self._current_turn_revision

    def _combined_turn_audio(self, current_segment: np.ndarray) -> np.ndarray:
        if self._speculative_audio_prefix is None:
            return current_segment
        return np.concatenate((self._speculative_audio_prefix, current_segment))

    def before_emit_output(self, output: VADOut) -> None:
        if isinstance(output, VADAudio):
            self._drop_superseded_vad_audio(output)

    def _drop_superseded_vad_audio(self, latest: VADAudio) -> int:
        if not hasattr(self.queue_out, "mutex") or not hasattr(self.queue_out, "queue"):
            return 0

        dropped = 0
        with self.queue_out.mutex:
            kept: list[Any] = []
            while self.queue_out.queue:
                queued_item = self.queue_out.queue.popleft()
                if isinstance(queued_item, VADAudio) and self._vad_audio_is_superseded(
                    queued_item,
                    latest,
                ):
                    dropped += 1
                else:
                    kept.append(queued_item)
            self.queue_out.queue.extend(kept)
            if dropped:
                self.queue_out.not_full.notify_all()

        if dropped:
            logger.debug(
                "VAD: dropped %d superseded audio chunk(s) before enqueueing turn=%s rev=%s mode=%s",
                dropped,
                latest.turn_id,
                latest.turn_revision,
                latest.mode,
            )
        return dropped

    def _vad_audio_is_superseded(self, queued_item: VADAudio, latest: VADAudio) -> bool:
        if queued_item.turn_id is None or queued_item.turn_revision is None:
            return False
        if self.speculative_turns is not None and not self.speculative_turns.is_latest(
            queued_item.turn_id,
            queued_item.turn_revision,
        ):
            return True
        return (
            queued_item.mode == "progressive"
            and queued_item.turn_id == latest.turn_id
            and queued_item.turn_revision == latest.turn_revision
        )

    def process(self, audio_chunk: VADIn) -> Iterator[VADOut]:
        runtime_config = None
        if isinstance(audio_chunk, tuple):
            audio_chunk, runtime_config = audio_chunk
        self._apply_runtime_turn_detection(runtime_config)

        if not self.should_listen.is_set():
            return

        # Normal listening mode
        self._log_chunks += 1
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        self._total_samples += len(audio_int16)
        audio_float32 = int2float(audio_int16)

        vad_output = self.iterator(torch.from_numpy(audio_float32))

        # Deferred speech_started: only emit once active VAD speech reaches the valid speech threshold.
        is_triggered_now = self.iterator.triggered
        if is_triggered_now and not self._speech_started_emitted:
            segment_samples = sum(len(t) for t in self.iterator.buffer)
            segment_duration_ms = segment_samples / self.sample_rate * 1000
            active_speech_duration_ms = self._current_active_speech_duration_ms()
            if active_speech_duration_ms >= self.min_speech_ms:
                speech_buffer_duration_ms = self._speech_buffer_duration_ms()
                start_ms = max(0, self._audio_ms - int(speech_buffer_duration_ms))
                self._begin_pending_reopen_if_needed(start_ms)
                turn_id, turn_revision, reopened = self._ensure_turn_for_speech_start(start_ms)
                self._speech_started_emitted = True
                self._log_speech_starts += 1
                logger.info(
                    "Speech started (confirmed, active=%.0fms, segment=%.0fms, turn=%s rev=%s)",
                    active_speech_duration_ms,
                    segment_duration_ms,
                    turn_id,
                    turn_revision,
                )
                if self.text_output_queue:
                    self.text_output_queue.put(
                        SpeechStartedEvent(
                            audio_start_ms=start_ms,
                            turn_id=turn_id,
                            turn_revision=turn_revision,
                            reopened=reopened,
                        )
                    )

        # Log a summary once per second instead of every chunk
        now = time.time()
        if now - self._last_log_time >= 1.0:
            state = "SPEAKING" if is_triggered_now else "silent"
            logger.debug(
                f"VAD: {self._log_chunks} chunks/s | {state} | "
                f"starts={self._log_speech_starts} ends={self._log_speech_ends} progressive={self._log_progressive_yields}"
            )
            self._log_chunks = 0
            self._log_speech_starts = 0
            self._log_speech_ends = 0
            self._log_progressive_yields = 0
            self._last_log_time = now

        if self._uses_realtime_turn_handling():
            # Realtime mode keeps turns reopenable; live transcription additionally
            # emits progressive audio chunks while speaking.
            yield from self._process_realtime(vad_output)
        else:
            # Original mode: yield only when speech ends
            yield from self._process_normal(vad_output)

    def _process_realtime(self, vad_output: list[torch.Tensor] | None) -> Iterator[VADOut]:
        """Process with real-time progressive audio release."""
        # Check if we're currently in a speech segment.
        if self.enable_realtime_transcription and hasattr(self.iterator, "buffer") and len(self.iterator.buffer) > 0:
            current_time = time.time()
            duration_ms = self._speech_buffer_duration_ms()
            progressive_pause = self._progressive_processing_pause(duration_ms)

            # Yield accumulated audio periodically while speaking
            if (current_time - self.last_process_time) >= progressive_pause:
                array = torch.cat(self.iterator.speech_buffer()).cpu().numpy()
                duration_ms = len(array) / self.sample_rate * 1000
                active_speech_duration_ms = self._current_active_speech_duration_ms()

                if active_speech_duration_ms >= self.min_speech_ms:
                    self._log_progressive_yields += 1
                    logger.debug(
                        "VAD: yielding progressive audio (segment=%.0fms, active=%.0fms, interval=%.2fs)",
                        duration_ms,
                        active_speech_duration_ms,
                        progressive_pause,
                    )
                    turn_id, turn_revision = self._current_turn_metadata()
                    yield VADAudio(
                        audio=self._combined_turn_audio(array),
                        mode="progressive",
                        turn_id=turn_id,
                        turn_revision=turn_revision,
                    )
                    self.last_process_time = current_time

        # Handle end of speech
        if vad_output is not None:
            if len(vad_output) == 0:
                logger.info("VAD: phantom trigger (empty buffer), closing speech pair")
                if self._speech_started_emitted and self.text_output_queue:
                    turn_id, turn_revision = self._current_turn_metadata()
                    self.text_output_queue.put(
                        SpeechStoppedEvent(
                            audio_end_ms=self._audio_ms,
                            turn_id=turn_id,
                            turn_revision=turn_revision,
                        )
                    )
                if not self._speech_started_emitted:
                    self._cancel_pending_reopen()
                self._speech_started_emitted = False
                return

            array = torch.cat(vad_output).cpu().numpy()
            duration_ms = len(array) / self.sample_rate * 1000
            active_speech_duration_ms = self._last_utterance_active_speech_duration_ms()

            if active_speech_duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms:
                logger.info(
                    "VAD: discarding segment=%.0fms active=%.0fms (active_min=%sms, segment_max=%sms)",
                    duration_ms,
                    active_speech_duration_ms,
                    self.min_speech_ms,
                    self.max_speech_ms,
                )
                if self._speech_started_emitted and self.text_output_queue:
                    turn_id, turn_revision = self._current_turn_metadata()
                    self.text_output_queue.put(
                        SpeechStoppedEvent(
                            audio_end_ms=self._audio_ms,
                            turn_id=turn_id,
                            turn_revision=turn_revision,
                        )
                    )
                if not self._speech_started_emitted:
                    self._cancel_pending_reopen()
                self._speech_started_emitted = False
            else:
                end_ms = self._audio_ms
                if not self._speech_started_emitted:
                    start_ms = max(0, end_ms - int(duration_ms))
                    turn_id, turn_revision, reopened = self._ensure_turn_for_speech_start(start_ms)
                    if self.text_output_queue:
                        self.text_output_queue.put(
                            SpeechStartedEvent(
                                audio_start_ms=start_ms,
                                turn_id=turn_id,
                                turn_revision=turn_revision,
                                reopened=reopened,
                                interrupt_response=False,
                            )
                        )
                else:
                    turn_id, turn_revision = self._current_turn_metadata()
                self._log_speech_ends += 1
                logger.info(
                    "Speech soft-ended (segment=%.0fms, active=%.0fms, turn=%s rev=%s)",
                    duration_ms,
                    active_speech_duration_ms,
                    turn_id,
                    turn_revision,
                )
                if self.audio_enhancement:
                    array = self._apply_audio_enhancement(array)
                output_array = self._combined_turn_audio(array)
                combined_duration_s = len(output_array) / self.sample_rate
                if self.text_output_queue:
                    self.text_output_queue.put(
                        SpeechStoppedEvent(
                            duration_s=combined_duration_s,
                            audio_end_ms=end_ms,
                            turn_id=turn_id,
                            turn_revision=turn_revision,
                        )
                    )
                self._speculative_audio_prefix = output_array
                self._last_final_wall_time = time.time()
                self._last_final_audio_ms = end_ms
                if self.speculative_turns:
                    self.speculative_turns.start_reopen_grace(
                        turn_id,
                        turn_revision,
                        self.speculative_reopen_ms / 1000.0,
                    )
                yield VADAudio(audio=output_array, mode="final", turn_id=turn_id, turn_revision=turn_revision)
                self.last_process_time = 0.0
                self._speech_started_emitted = False

    def _progressive_processing_pause(self, duration_ms: float) -> float:
        base_pause = max(0.0, self.realtime_processing_pause)
        duration_s = duration_ms / 1000.0
        if duration_s < 8.0:
            multiplier = 1.0
        elif duration_s < 15.0:
            multiplier = 2.0
        elif duration_s < 30.0:
            multiplier = 4.0
        else:
            multiplier = 6.0
        return min(base_pause * multiplier, 2.0)

    def _process_normal(self, vad_output: list[torch.Tensor] | None) -> Iterator[VADOut]:
        """Original processing: yield only when speech ends."""
        if vad_output is not None:
            if len(vad_output) == 0:
                logger.info("VAD: phantom trigger (empty buffer), closing speech pair")
                if self._speech_started_emitted and self.text_output_queue:
                    self.text_output_queue.put(SpeechStoppedEvent(audio_end_ms=self._audio_ms))
                self._speech_started_emitted = False
                return

            array = torch.cat(vad_output).cpu().numpy()
            duration_ms = len(array) / self.sample_rate * 1000
            active_speech_duration_ms = self._last_utterance_active_speech_duration_ms()
            if active_speech_duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms:
                logger.info(
                    "VAD: discarding segment=%.0fms active=%.0fms (active_min=%sms, segment_max=%sms)",
                    duration_ms,
                    active_speech_duration_ms,
                    self.min_speech_ms,
                    self.max_speech_ms,
                )
                if self._speech_started_emitted and self.text_output_queue:
                    self.text_output_queue.put(SpeechStoppedEvent(audio_end_ms=self._audio_ms))
                self._speech_started_emitted = False
            else:
                end_ms = self._audio_ms
                if not self._speech_started_emitted and self.text_output_queue:
                    self.text_output_queue.put(
                        SpeechStartedEvent(
                            audio_start_ms=max(0, end_ms - int(duration_ms)),
                            interrupt_response=False,
                        )
                    )
                self._log_speech_ends += 1
                self.should_listen.clear()
                logger.info(f"Speech ended ({duration_ms:.0f}ms), stop listening")
                if self.text_output_queue:
                    self.text_output_queue.put(SpeechStoppedEvent(duration_s=duration_ms / 1000.0, audio_end_ms=end_ms))
                if self.audio_enhancement:
                    array = self._apply_audio_enhancement(array)
                yield VADAudio(audio=array)
                self._speech_started_emitted = False

    def _apply_audio_enhancement(self, array: np.ndarray) -> np.ndarray:
        """Apply audio enhancement if enabled."""
        import torchaudio

        if self.sample_rate != self.df_state.sr():
            audio_float32 = torchaudio.functional.resample(
                torch.from_numpy(array),
                orig_freq=self.sample_rate,
                new_freq=self.df_state.sr(),
            )
            enhanced = enhance(
                self.enhanced_model,
                self.df_state,
                audio_float32.unsqueeze(0),
            )
            enhanced = torchaudio.functional.resample(
                enhanced,
                orig_freq=self.df_state.sr(),
                new_freq=self.sample_rate,
            )
        else:
            enhanced = enhance(self.enhanced_model, self.df_state, torch.from_numpy(array))
        return enhanced.numpy().squeeze()

    def on_session_end(self):
        self.iterator.reset_states()
        self.iterator.buffer = []
        self.last_process_time = 0.0
        self._total_samples = 0
        self._speech_started_emitted = False
        self._turn_counter = 0
        self._current_turn_id = None
        self._current_turn_revision = None
        self._speculative_audio_prefix = None
        self._last_final_wall_time = None
        self._last_final_audio_ms = None
        self._pending_reopen_candidate = None
        if self.speculative_turns:
            self.speculative_turns.reset()
        self.should_listen.set()
        logger.debug("VAD session state reset")

    @property
    def min_time_to_debug(self) -> float:
        return 0.00001
