import socket
import threading
from queue import Queue
from dataclasses import dataclass, field
import sounddevice as sd
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class ListenAndPlayArguments:
    send_rate: int = field(default=16000, metadata={"help": "In Hz. Default is 16000."})
    recv_rate: int = field(default=16000, metadata={"help": "In Hz. Default is 16000."})
    list_play_chunk_size: int = field(
        default=1024,
        metadata={"help": "The size of data chunks (in bytes). Default is 1024."},
    )
    host: str = field(
        default="localhost",
        metadata={
            "help": "The hostname or IP address for listening and playing. Default is 'localhost'."
        },
    )
    send_port: int = field(
        default=12345,
        metadata={"help": "The network port for sending data. Default is 12345."},
    )
    recv_port: int = field(
        default=12346,
        metadata={"help": "The network port for receiving data. Default is 12346."},
    )


def listen_and_play(
    send_rate=16000,
    recv_rate=44100,
    list_play_chunk_size=1024,
    host="localhost",
    send_port=12345,
    recv_port=12346,
):
    logging.debug(f"Starting listen_and_play with parameters: send_rate={send_rate}, recv_rate={recv_rate}, list_play_chunk_size={list_play_chunk_size}, host={host}, send_port={send_port}, recv_port={recv_port}")

    send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    send_socket.connect((host, send_port))
    logging.debug(f"Connected to send socket: {host}:{send_port}")

    recv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    recv_socket.connect((host, recv_port))
    logging.debug(f"Connected to receive socket: {host}:{recv_port}")

    print("Recording and streaming...")

    stop_event = threading.Event()
    recv_queue = Queue()
    send_queue = Queue()

    def callback_recv(outdata, frames, time, status):
        if status:
            logging.warning(f"Receive callback status: {status}")
        if not recv_queue.empty():
            data = recv_queue.get()
            outdata[: len(data)] = data
            outdata[len(data) :] = b"\x00" * (len(outdata) - len(data))
            logging.debug(f"Received {len(data)} bytes of audio data")
        else:
            outdata[:] = b"\x00" * len(outdata)
            logging.debug("Receive queue empty, playing silence")

    def callback_send(indata, frames, time, status):
        if status:
            logging.warning(f"Send callback status: {status}")
        if recv_queue.empty():
            data = bytes(indata)
            send_queue.put(data)
            logging.debug(f"Captured {len(data)} bytes of audio data")

    def send(stop_event, send_queue):
        logging.debug("Send thread started")
        while not stop_event.is_set():
            data = send_queue.get()
            send_socket.sendall(data)
            logging.debug(f"Sent {len(data)} bytes of audio data")
        logging.debug("Send thread stopped")

    def recv(stop_event, recv_queue):
        logging.debug("Receive thread started")
        def receive_full_chunk(conn, chunk_size):
            data = b""
            while len(data) < chunk_size:
                packet = conn.recv(chunk_size - len(data))
                if not packet:
                    logging.warning("Connection closed while receiving data")
                    return None
                data += packet
            return data

        while not stop_event.is_set():
            data = receive_full_chunk(recv_socket, list_play_chunk_size * 2)
            if data:
                recv_queue.put(data)
                logging.debug(f"Received {len(data)} bytes of audio data")
            else:
                logging.warning("Failed to receive data")
        logging.debug("Receive thread stopped")

    try:
        send_stream = sd.RawInputStream(
            samplerate=send_rate,
            channels=1,
            dtype="int16",
            blocksize=list_play_chunk_size,
            callback=callback_send,
        )
        logging.debug(f"Created send stream with rate {send_rate} Hz")

        recv_stream = sd.RawOutputStream(
            samplerate=recv_rate,
            channels=1,
            dtype="int16",
            blocksize=list_play_chunk_size,
            callback=callback_recv,
        )
        logging.debug(f"Created receive stream with rate {recv_rate} Hz")

        threading.Thread(target=send_stream.start).start()
        threading.Thread(target=recv_stream.start).start()
        logging.debug("Started audio streams")

        send_thread = threading.Thread(target=send, args=(stop_event, send_queue))
        send_thread.start()
        recv_thread = threading.Thread(target=recv, args=(stop_event, recv_queue))
        recv_thread.start()
        logging.debug("Started send and receive threads")

        input("Press Enter to stop...")

    except KeyboardInterrupt:
        logging.info("\nProgram interrupted by user. Exiting...")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

    finally:
        stop_event.set()
        logging.debug("Stop event set")
        recv_thread.join()
        send_thread.join()
        logging.debug("Threads joined")
        send_socket.close()
        recv_socket.close()
        logging.debug("Sockets closed")
        print("Connection closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Listen and Play Audio")
    parser.add_argument("--send_rate", type=int, default=16000, help="In Hz. Default is 16000.")
    parser.add_argument("--recv_rate", type=int, default=16000, help="In Hz. Default is 16000.")
    parser.add_argument("--list_play_chunk_size", type=int, default=1024, help="The size of data chunks (in bytes). Default is 1024.")
    parser.add_argument("--host", type=str, default="localhost", help="The hostname or IP address for listening and playing. Default is 'localhost'.")
    parser.add_argument("--send_port", type=int, default=12345, help="The network port for sending data. Default is 12345.")
    parser.add_argument("--recv_port", type=int, default=12346, help="The network port for receiving data. Default is 12346.")
    
    args = parser.parse_args()
    logging.debug(f"Parsed arguments: {args}")
    listen_and_play(**vars(args))
