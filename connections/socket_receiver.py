import socket
from rich.console import Console
import logging

logger = logging.getLogger(__name__)

console = Console()


class SocketReceiver:
    """
    Handles reception of the audio packets from the client.
    """

    def __init__(
        self,
        stop_event,
        queue_out,
        should_listen,
        host="0.0.0.0",
        port=12345,
        chunk_size=1024,
    ):
        self.stop_event = stop_event
        self.queue_out = queue_out
        self.should_listen = should_listen
        self.chunk_size = chunk_size
        self.host = host
        self.port = port

    def receive_full_chunk(self, conn, chunk_size):
        data = b""
        while len(data) < chunk_size:
            packet = conn.recv(chunk_size - len(data))
            if not packet:
                # connection closed
                return None
            data += packet
        return data

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        logger.info("Receiver waiting to be connected...")
        self.conn, _ = self.socket.accept()
        logger.info("receiver connected")

        self.should_listen.set()
        while not self.stop_event.is_set():
            audio_chunk = self.receive_full_chunk(self.conn, self.chunk_size)
            if audio_chunk is None:
                # connection closed
                self.queue_out.put(b"END")
                break
            if self.should_listen.is_set():
                self.queue_out.put(audio_chunk)
        self.conn.close()
        logger.info("Receiver closed")
