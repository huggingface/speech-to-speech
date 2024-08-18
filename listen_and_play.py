import socket
import threading
from queue import Queue
from dataclasses import dataclass, field
import sounddevice as sd
from transformers import HfArgumentParser

@dataclass
class ListenAndPlayArguments:
    send_rate: int = field(
        default=16000,
        metadata={"help": "In Hz. Default is 16000."}
    )
    recv_rate: int = field(
        default=44100,
        metadata={"help": "In Hz. Default is 44100."}
    )
    chunk_size: int = field(
        default=1024,
        metadata={"help": "The size of data chunks (in bytes). Default is 1024."}
    )
    host: str = field(
        default="localhost",
        metadata={"help": "The hostname or IP address for listening and playing. Default is 'localhost'."}
    )
    send_port: int = field(
        default=12345,
        metadata={"help": "The network port for sending data. Default is 12345."}
    )
    recv_port: int = field(
        default=12346,
        metadata={"help": "The network port for receiving data. Default is 12346."}
    )

class AudioStreamer:
    def __init__(self, args: ListenAndPlayArguments):
        self.args = args
        self.stop_event = threading.Event()
        self.send_queue = Queue()
        self.recv_queue = Queue()

        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.send_socket.connect((args.host, args.send_port))

        self.recv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.recv_socket.connect((args.host, args.recv_port))

    def send_audio(self):
        while not self.stop_event.is_set():
            data = self.send_queue.get()
            self.send_socket.sendall(data)

    def receive_audio(self):
        while not self.stop_event.is_set():
            data = self._receive_full_chunk(self.args.chunk_size * 2)
            if data:
                self.recv_queue.put(data)

    def _receive_full_chunk(self, chunk_size):
        data = b''
        while len(data) < chunk_size:
            packet = self.recv_socket.recv(chunk_size - len(data))
            if not packet:
                return None
            data += packet
        return data

    def callback_send(self, indata, frames, time, status):
        self.send_queue.put(bytes(indata))

    def callback_recv(self, outdata, frames, time, status):
        if not self.recv_queue.empty():
            data = self.recv_queue.get()
            outdata[:len(data)] = data
            outdata[len(data):] = b'\x00' * (len(outdata) - len(data))
        else:
            outdata[:] = b'\x00' * len(outdata)

    def start(self):
        send_stream = sd.RawInputStream(
            samplerate=self.args.send_rate, channels=1, dtype='int16',
            blocksize=self.args.chunk_size, callback=self.callback_send
        )
        recv_stream = sd.RawOutputStream(
            samplerate=self.args.recv_rate, channels=1, dtype='int16',
            blocksize=self.args.chunk_size, callback=self.callback_recv
        )

        send_stream.start()
        recv_stream.start()

        send_thread = threading.Thread(target=self.send_audio)
        recv_thread = threading.Thread(target=self.receive_audio)
        send_thread.start()
        recv_thread.start()

        input("Press Enter to stop...")

        self.stop_event.set()
        send_thread.join()
        recv_thread.join()
        send_stream.stop()
        recv_stream.stop()

        self.cleanup()

    def cleanup(self):
        self.send_socket.close()
        self.recv_socket.close()
        print("Connection closed.")

def main():
    parser = HfArgumentParser((ListenAndPlayArguments,))
    args = parser.parse_args_into_dataclasses()[0]
    streamer = AudioStreamer(args)
    streamer.start()

if __name__ == "__main__":
    main()
