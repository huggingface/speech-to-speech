import logging
import socket
from queue import Empty, Queue
from threading import Event

import numpy as np
from rich.console import Console

from speech_to_speech.pipeline.control import PipelineControlMessage
from speech_to_speech.pipeline.messages import PIPELINE_END
from speech_to_speech.pipeline.queue_types import AudioOutItem

logger = logging.getLogger(__name__)

console = Console()


class SocketSender:
    """
    Handles sending generated audio packets to the clients.
    """

    def __init__(
        self,
        stop_event: Event,
        queue_in: Queue[AudioOutItem],
        host: str = "0.0.0.0",
        port: int = 12346,
    ) -> None:
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.host = host
        self.port = port

    def run(self) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        logger.info("Sender waiting to be connected...")
        self.conn, _ = self.socket.accept()
        logger.info("sender connected")

        while not self.stop_event.is_set():
            try:
                audio_chunk = self.queue_in.get(timeout=0.5)
            except Empty:
                continue
            if isinstance(audio_chunk, PipelineControlMessage):
                continue
            payload: bytes
            if isinstance(audio_chunk, bytes):
                payload = audio_chunk
            elif isinstance(audio_chunk, np.ndarray):
                payload = audio_chunk.tobytes()
            elif hasattr(audio_chunk, "tobytes"):
                payload = audio_chunk.tobytes()
            else:
                continue
            self.conn.sendall(payload)
            if isinstance(audio_chunk, bytes) and audio_chunk == PIPELINE_END:
                break
        self.conn.close()
        logger.info("Sender closed")
