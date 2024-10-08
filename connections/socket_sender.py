import socket
from rich.console import Console
import logging
import pickle
import struct

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
        self.port = port

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        logger.info("Sender waiting to be connected...")
        self.conn, _ = self.socket.accept()
        logger.info("sender connected")

        while not self.stop_event.is_set():
            data = self.queue_in.get()
            packet = {}
            if 'audio' in data and data['audio'] is not None:
                audio_chunk = data['audio']
                packet['audio'] = audio_chunk
            if 'text' in data and data['text'] is not None:
                packet['text'] = data['text']
            if 'visemes' in data and data['visemes'] is not None:
                packet['visemes'] = data['visemes']

            # Serialize the packet using pickle
            serialized_packet = pickle.dumps(packet)

            # Compute the length of the serialized packet
            packet_length = len(serialized_packet)

            # Send the packet length as a 4-byte integer using struct
            self.conn.sendall(struct.pack('!I', packet_length))

            # Send the serialized packet
            self.conn.sendall(serialized_packet)

            if 'audio' in data and data['audio'] is not None:
                if isinstance(audio_chunk, bytes) and audio_chunk == b"END":
                    break
            
        self.conn.close()
        logger.info("Sender closed")
