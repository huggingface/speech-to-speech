import logging
import threading
import time
from queue import Queue

import numpy as np
import sounddevice as sd

from speech_to_speech.pipeline.queue_types import AudioInItem, AudioOutItem

logger = logging.getLogger(__name__)


class LocalAudioStreamer:
    def __init__(
        self,
        input_queue: Queue[AudioInItem],
        output_queue: Queue[AudioOutItem],
        list_play_chunk_size: int = 512,
    ) -> None:
        self.list_play_chunk_size = list_play_chunk_size

        self.stop_event = threading.Event()
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self) -> None:
        # Pre-generate a static dither buffer (±1 LSB, -96 dB) to keep the
        # audio sink active without calling numpy inside the real-time callback.
        dither = np.random.randint(-1, 2, size=(self.list_play_chunk_size, 1), dtype=np.int16)

        def callback(indata: np.ndarray, outdata: np.ndarray, frames: int, time: float, status: str) -> None:
            # During shutdown, just output silence
            if self.stop_event.is_set():
                outdata[:] = 0 * outdata
                return

            if self.output_queue.empty():
                pcm = np.ascontiguousarray(indata, dtype=np.int16)
                self.input_queue.put(pcm.tobytes())
                outdata[:] = dither
            else:
                try:
                    audio_chunk = self.output_queue.get_nowait()
                    # Validate audio chunk is numpy array
                    if isinstance(audio_chunk, np.ndarray):
                        outdata[:] = audio_chunk[:, np.newaxis]
                    else:
                        outdata[:] = 0 * outdata
                except Exception:
                    outdata[:] = 0 * outdata

        logger.debug("Available devices:")
        logger.debug(sd.query_devices())
        with sd.Stream(
            samplerate=16000,
            dtype="int16",
            channels=1,
            callback=callback,
            blocksize=self.list_play_chunk_size,
        ):
            logger.info("Starting local audio stream")
            while not self.stop_event.is_set():
                time.sleep(0.001)
            print("Stopping recording")
