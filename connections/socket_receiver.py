import socket
import time
from rich.console import Console
import logging

logger = logging.getLogger(__name__)

console = Console()

# If should_listen stays cleared for longer than this, something in the
# pipeline probably failed (LLM exception, TTS crash, etc.).  Re-enable
# listening so the user isn't permanently locked out.
SHOULD_LISTEN_TIMEOUT_S = 30.0


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
        listen_cleared_at = None
        while not self.stop_event.is_set():
            audio_chunk = self.receive_full_chunk(self.conn, self.chunk_size)
            if audio_chunk is None:
                # connection closed
                self.queue_out.put(b"END")
                break
            if self.should_listen.is_set():
                self.queue_out.put(audio_chunk)
                listen_cleared_at = None
            else:
                # Track how long should_listen has been cleared
                if listen_cleared_at is None:
                    listen_cleared_at = time.monotonic()
                elif time.monotonic() - listen_cleared_at > SHOULD_LISTEN_TIMEOUT_S:
                    logger.warning(
                        "should_listen has been cleared for %.0fs — "
                        "pipeline may be stuck, re-enabling listening",
                        SHOULD_LISTEN_TIMEOUT_S,
                    )
                    self.should_listen.set()
                    listen_cleared_at = None
        self.conn.close()
        logger.info("Receiver closed")
