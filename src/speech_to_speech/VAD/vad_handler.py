from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from queue import Queue
from threading import Event
from typing import Any

import numpy as np
import torch
import torchaudio

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.events import SpeechStartedEvent, SpeechStoppedEvent
from speech_to_speech.pipeline.messages import VADAudio
from speech_to_speech.utils.utils import int2float
from speech_to_speech.VAD.vad_iterator import VADIterator

logger = logging.getLogger(__name__)

# Optional import for audio enhancement
try:
    from df.enhance import enhance, init_df

    HAS_DF = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_DF = False
    logger.warning(f"DeepFilterNet not available for audio enhancement: {e}")


class VADHandler(BaseHandler[bytes | tuple[bytes, RuntimeConfig]]):
    """
    Handles voice activity detection. When voice activity is detected, audio will be accumulated until the end of speech is detected and then passed
    to the following part.
    """

    def setup(
        self,
        should_listen: Event,
        thresh: float = 0.3,
        sample_rate: int = 16000,
        min_silence_ms: int = 1000,
        min_speech_ms: int = 500,
        max_speech_ms: float = float("inf"),
        speech_pad_ms: int = 30,
        audio_enhancement: bool = False,
        enable_realtime_transcription: bool = False,
        realtime_processing_pause: float = 0.25,
        text_output_queue: Queue[Any] | None = None,
    ) -> None:
        self.should_listen = should_listen
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.enable_realtime_transcription = enable_realtime_transcription
        self.realtime_processing_pause = realtime_processing_pause
        self.text_output_queue = text_output_queue
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
        self.last_process_time = 0

        # Cumulative sample counter for audio_start_ms / audio_end_ms
        self._total_samples: int = 0

        # Throttled logging state (summary once per second)
        self._last_log_time = 0.0
        self._log_chunks = 0
        self._log_speech_starts = 0
        self._log_speech_ends = 0
        self._log_progressive_yields = 0
        self._speech_started_emitted = False

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

    def process(self, audio_chunk: bytes | tuple[bytes, RuntimeConfig]) -> Iterator[VADAudio]:
        runtime_config = None
        if isinstance(audio_chunk, tuple):
            audio_chunk, runtime_config = audio_chunk
        self._apply_runtime_turn_detection(runtime_config)
        logger.debug(f"VAD received {len(audio_chunk)} bytes")

        if not self.should_listen.is_set():
            return

        # Normal listening mode
        self._log_chunks += 1
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        self._total_samples += len(audio_int16)
        audio_float32 = int2float(audio_int16)

        vad_output = self.iterator(torch.from_numpy(audio_float32))

        # Deferred speech_started: only emit once buffer >= min_speech_ms
        is_triggered_now = self.iterator.triggered
        if is_triggered_now and not self._speech_started_emitted:
            buffer_samples = sum(len(t) for t in self.iterator.buffer)
            buffer_duration_ms = buffer_samples / self.sample_rate * 1000
            if buffer_duration_ms >= self.min_speech_ms:
                self._speech_started_emitted = True
                self._log_speech_starts += 1
                start_ms = max(0, self._audio_ms - int(buffer_duration_ms))
                logger.info("Speech started (confirmed, %.0fms buffered)", buffer_duration_ms)
                if self.text_output_queue:
                    self.text_output_queue.put(SpeechStartedEvent(audio_start_ms=start_ms))

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

        if self.enable_realtime_transcription:
            # Progressive mode: yield audio chunks while speaking
            yield from self._process_realtime(vad_output)
        else:
            # Original mode: yield only when speech ends
            yield from self._process_normal(vad_output)

    def _process_realtime(self, vad_output):
        """Process with real-time progressive audio release."""
        # Check if we're currently in a speech segment
        if hasattr(self.iterator, "buffer") and len(self.iterator.buffer) > 0:
            current_time = time.time()

            # Yield accumulated audio periodically while speaking
            if (current_time - self.last_process_time) >= self.realtime_processing_pause:
                array = torch.cat(self.iterator.speech_buffer()).cpu().numpy()
                duration_ms = len(array) / self.sample_rate * 1000

                if duration_ms >= self.min_speech_ms:
                    self._log_progressive_yields += 1
                    logger.debug(f"VAD: yielding progressive audio ({duration_ms:.0f}ms)")
                    yield VADAudio(audio=array, mode="progressive")
                    self.last_process_time = current_time

        # Handle end of speech
        if vad_output is not None:
            if len(vad_output) == 0:
                logger.info("VAD: phantom trigger (empty buffer), closing speech pair")
                if self._speech_started_emitted and self.text_output_queue:
                    self.text_output_queue.put(SpeechStoppedEvent(audio_end_ms=self._audio_ms))
                self._speech_started_emitted = False
                return

            array = torch.cat(vad_output).cpu().numpy()
            duration_ms = len(array) / self.sample_rate * 1000

            if duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms:
                logger.info(
                    f"VAD: discarding {duration_ms:.0f}ms segment (bounds: {self.min_speech_ms}-{self.max_speech_ms}ms)"
                )
                if self._speech_started_emitted and self.text_output_queue:
                    self.text_output_queue.put(SpeechStoppedEvent(audio_end_ms=self._audio_ms))
                self._speech_started_emitted = False
            else:
                end_ms = self._audio_ms
                if not self._speech_started_emitted and self.text_output_queue:
                    self.text_output_queue.put(SpeechStartedEvent(audio_start_ms=max(0, end_ms - int(duration_ms))))
                self._log_speech_ends += 1
                self.should_listen.clear()
                logger.info(f"Speech ended ({duration_ms:.0f}ms), stop listening")
                if self.text_output_queue:
                    self.text_output_queue.put(SpeechStoppedEvent(duration_s=duration_ms / 1000.0, audio_end_ms=end_ms))
                if self.audio_enhancement:
                    array = self._apply_audio_enhancement(array)
                yield VADAudio(audio=array, mode="final")
                self.last_process_time = 0
                self._speech_started_emitted = False

    def _process_normal(self, vad_output: list[torch.Tensor] | None) -> Iterator[VADAudio]:
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
            if duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms:
                logger.info(
                    f"VAD: discarding {duration_ms:.0f}ms segment (bounds: {self.min_speech_ms}-{self.max_speech_ms}ms)"
                )
                if self._speech_started_emitted and self.text_output_queue:
                    self.text_output_queue.put(SpeechStoppedEvent(audio_end_ms=self._audio_ms))
                self._speech_started_emitted = False
            else:
                end_ms = self._audio_ms
                if not self._speech_started_emitted and self.text_output_queue:
                    self.text_output_queue.put(SpeechStartedEvent(audio_start_ms=max(0, end_ms - int(duration_ms))))
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
        self.last_process_time = 0
        self._total_samples = 0
        self._speech_started_emitted = False
        self.should_listen.set()
        logger.debug("VAD session state reset")

    @property
    def min_time_to_debug(self) -> float:
        return 0.00001
