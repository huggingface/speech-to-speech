import socket
from rich.console import Console
import logging

logger = logging.getLogger(__name__)

console = Console()


class SocketSender:
    """
    Handles sending generated audio packets to the clients.
    """

    def __init__(self, stop_event, queue_in, host="0.0.0.0", port=12346):
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.host = host
        self.socket = None
        self.conn = None
        self.port = port

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        logger.info("Sender waiting to be connected...")

        try:
            self.conn, _ = self.socket.accept()
            logger.info("Sender connected")

            while not self.stop_event.is_set():
                audio_chunk = self.queue_in.get()
                self.conn.sendall(audio_chunk)
                if isinstance(audio_chunk, bytes) and audio_chunk == b"END":
                    break
        except OSError as e:
            # Handle exception due to socket shutdown
            logger.debug(f"SocketSender received exception: {e}. Possibly the sever is in termination process..")

        finally:
            if self.conn is not None:
                self.conn.close()
            logger.info("Sender closed")
 
    def stop(self):
        self.stop_event.set()
        logger.debug("SocketSender: shutdown socket")
        self.socket.shutdown(socket.SHUT_RDWR) # Shutdown the socket to overcome blocking socket calls (e.g. accept())
