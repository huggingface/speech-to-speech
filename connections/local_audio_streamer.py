import threading
import sounddevice as sd
import numpy as np

import time
import logging

logger = logging.getLogger(__name__)


class LocalAudioStreamer:
    def __init__(
        self,
        input_queue,
        output_queue,
        list_play_chunk_size=512,
    ):
        self.list_play_chunk_size = list_play_chunk_size

        self.stop_event = threading.Event()
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self):
        def callback(indata, outdata, frames, time, status):
            # During shutdown, just output silence
            if self.stop_event.is_set():
                outdata[:] = 0 * outdata
                return

            if self.output_queue.empty():
                self.input_queue.put(indata.copy())
                outdata[:] = 0 * outdata
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
