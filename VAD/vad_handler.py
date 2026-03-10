import time
import torchaudio
from VAD.vad_iterator import VADIterator
from baseHandler import BaseHandler
import numpy as np
import torch
from rich.console import Console

from utils.utils import int2float
import logging

logger = logging.getLogger(__name__)

# Optional import for audio enhancement
try:
    from df.enhance import enhance, init_df
    HAS_DF = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_DF = False
    logger.warning(f"DeepFilterNet not available for audio enhancement: {e}")

console = Console()


class VADHandler(BaseHandler):
    """
    Handles voice activity detection. When voice activity is detected, audio will be accumulated until the end of speech is detected and then passed
    to the following part.
    """

    def setup(
        self,
        should_listen,
        thresh=0.3,
        sample_rate=16000,
        min_silence_ms=1000,
        min_speech_ms=500,
        max_speech_ms=float("inf"),
        speech_pad_ms=30,
        audio_enhancement=False,
        enable_realtime_transcription=False,
        realtime_processing_pause=0.25,
        text_output_queue=None,
        runtime_config=None,
    ):
        self.should_listen = should_listen
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.enable_realtime_transcription = enable_realtime_transcription
        self.realtime_processing_pause = realtime_processing_pause
        self.text_output_queue = text_output_queue
        self.runtime_config = runtime_config
        self._last_turn_detection = None
        self.model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
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
                logger.error("Audio enhancement requested but DeepFilterNet is not available. Disabling audio enhancement.")
                self.audio_enhancement = False
            else:
                self.enhanced_model, self.df_state, _ = init_df()

        # State for progressive audio release
        self.accumulated_audio = []
        self.last_process_time = 0

        # Throttled logging state (summary once per second)
        self._last_log_time = 0.0
        self._log_chunks = 0
        self._log_speech_starts = 0
        self._log_speech_ends = 0
        self._log_progressive_yields = 0

    def _apply_runtime_turn_detection(self):
        """Check RuntimeConfig for turn_detection changes and apply them."""
        if not self.runtime_config or not self.runtime_config.turn_detection:
            return
        td = self.runtime_config.turn_detection
        if td is self._last_turn_detection:
            return
        self._last_turn_detection = td
        if "threshold" in td:
            self.iterator.threshold = td["threshold"]
            logger.info(f"VAD threshold updated to {td['threshold']}")
        if "silence_duration_ms" in td:
            self.iterator.min_silence_samples = (
                self.sample_rate * td["silence_duration_ms"] / 1000
            )
            logger.info(f"VAD silence duration updated to {td['silence_duration_ms']}ms")

    def process(self, audio_chunk):
        self._apply_runtime_turn_detection()
        logger.debug(f"VAD received {len(audio_chunk)} bytes")

        self._log_chunks += 1
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float32 = int2float(audio_int16)

        # Check speech state BEFORE processing
        was_triggered_before = self.iterator.triggered

        vad_output = self.iterator(torch.from_numpy(audio_float32))

        # Check if speech state changed AFTER processing
        is_triggered_now = self.iterator.triggered
        if is_triggered_now and not was_triggered_before:
            self._log_speech_starts += 1
            logger.info("Speech started")
            if self.text_output_queue:
                self.text_output_queue.put({"type": "speech_started"})

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
                array = torch.cat(self.iterator.buffer).cpu().numpy()
                duration_ms = len(array) / self.sample_rate * 1000

                if duration_ms >= self.min_speech_ms:
                    self._log_progressive_yields += 1
                    logger.debug(f"VAD: yielding progressive audio ({duration_ms:.0f}ms)")
                    # Yield with special flag to indicate this is progressive (not final)
                    yield ("progressive", array)
                    self.last_process_time = current_time

        # Handle end of speech
        if vad_output is not None and len(vad_output) != 0:
            logger.debug("VAD: end of speech detected")
            array = torch.cat(vad_output).cpu().numpy()
            duration_ms = len(array) / self.sample_rate * 1000

            if duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms:
                logger.debug(
                    f"VAD: skipping {duration_ms:.0f}ms segment (out of bounds)"
                )
            else:
                self._log_speech_ends += 1
                self.should_listen.clear()
                logger.info(f"Speech ended ({duration_ms:.0f}ms), stop listening")
                if self.text_output_queue:
                    self.text_output_queue.put({
                        "type": "speech_stopped",
                        "duration_s": duration_ms / 1000.0,
                    })
                if self.audio_enhancement:
                    array = self._apply_audio_enhancement(array)
                yield ("final", array)
                self.last_process_time = 0

    def _process_normal(self, vad_output):
        """Original processing: yield only when speech ends."""
        if vad_output is not None and len(vad_output) != 0:
            logger.debug("VAD: end of speech detected")
            array = torch.cat(vad_output).cpu().numpy()
            duration_ms = len(array) / self.sample_rate * 1000
            if duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms:
                logger.debug(
                    f"VAD: skipping {duration_ms:.0f}ms segment (out of bounds)"
                )
            else:
                self._log_speech_ends += 1
                self.should_listen.clear()
                logger.info(f"Speech ended ({duration_ms:.0f}ms), stop listening")
                if self.text_output_queue:
                    self.text_output_queue.put({
                        "type": "speech_stopped",
                        "duration_s": duration_ms / 1000.0,
                    })
                if self.audio_enhancement:
                    array = self._apply_audio_enhancement(array)
                yield array

    def _apply_audio_enhancement(self, array):
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
            enhanced = enhance(
                self.enhanced_model, self.df_state, torch.from_numpy(array)
            )
        return enhanced.numpy().squeeze()

    @property
    def min_time_to_debug(self):
        return 0.00001
