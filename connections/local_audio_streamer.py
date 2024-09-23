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
            if self.output_queue.empty():
                self.input_queue.put(indata.copy())
                outdata[:] = 0 * outdata
            else:
                data = self.output_queue.get()
                """
                # Check if text data is present and log it
                if data.get('text') is not None:
                    text = data['text']
                    logger.info(f"Text: {text}")
                # Check if viseme data is present and log it
                if data.get('visemes') is not None:
                    visemes = data['visemes']
                    logger.info(f"Visemes: {visemes}")
                """
                outdata[:] = data['audio'][:, np.newaxis]

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
