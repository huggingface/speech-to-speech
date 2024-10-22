import threading
from queue import Queue
import sounddevice as sd
import numpy as np
import time
from dataclasses import dataclass, field
import websocket
import ssl


@dataclass
class AudioStreamingClientArguments:
    sample_rate: int = field(
        default=16000, metadata={"help": "Audio sample rate in Hz. Default is 16000."}
    )
    chunk_size: int = field(
        default=512,
        metadata={"help": "The size of audio chunks in samples. Default is 512."},
    )
    api_url: str = field(
        default="https://yxfmjcvuzgi123sw.us-east-1.aws.endpoints.huggingface.cloud",
        metadata={"help": "The URL of the API endpoint."},
    )
    auth_token: str = field(
        default="your_auth_token",
        metadata={"help": "Authentication token for the API."},
    )


class AudioStreamingClient:
    def __init__(self, args: AudioStreamingClientArguments):
        self.args = args
        self.stop_event = threading.Event()
        self.send_queue = Queue()
        self.recv_queue = Queue()
        self.session_id = None
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.args.auth_token}",
            "Content-Type": "application/json",
        }
        self.session_state = (
            "idle"  # Possible states: idle, sending, processing, waiting
        )
        self.ws_ready = threading.Event()

    def start(self):
        print("Starting audio streaming...")

        ws_url = self.args.api_url.replace("http", "ws") + "/ws"

        self.ws = websocket.WebSocketApp(
            ws_url,
            header=[f"{key}: {value}" for key, value in self.headers.items()],
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )

        self.ws_thread = threading.Thread(
            target=self.ws.run_forever, kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}}
        )
        self.ws_thread.start()

        # Wait for the WebSocket to be ready
        self.ws_ready.wait()
        self.start_audio_streaming()

    def start_audio_streaming(self):
        self.send_thread = threading.Thread(target=self.send_audio)
        self.play_thread = threading.Thread(target=self.play_audio)

        with sd.InputStream(
            samplerate=self.args.sample_rate,
            channels=1,
            dtype="int16",
            callback=self.audio_input_callback,
            blocksize=self.args.chunk_size,
        ):
            self.send_thread.start()
            self.play_thread.start()
            input("Press Enter to stop streaming... \n")
            self.on_shutdown()

    def on_open(self, ws):
        print("WebSocket connection opened.")
        self.ws_ready.set()  # Signal that the WebSocket is ready

    def on_message(self, ws, message):
        # message is bytes
        if message == b"DONE":
            print("listen")
            self.session_state = "listen"
        else:
            if self.session_state != "processing":
                print("processing")
                self.session_state = "processing"
            audio_np = np.frombuffer(message, dtype=np.int16)
            self.recv_queue.put(audio_np)

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket connection closed.")

    def on_shutdown(self):
        self.stop_event.set()
        self.send_thread.join()
        self.play_thread.join()
        self.ws.close()
        self.ws_thread.join()
        print("Service shutdown.")

    def send_audio(self):
        while not self.stop_event.is_set():
            if not self.send_queue.empty():
                chunk = self.send_queue.get()
                if self.session_state != "processing":
                    self.ws.send(chunk.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)
                else:
                    self.ws.send([], opcode=websocket.ABNF.OPCODE_BINARY)  # handshake
            time.sleep(0.01)

    def audio_input_callback(self, indata, frames, time, status):
        self.send_queue.put(indata.copy())

    def audio_out_callback(self, outdata, frames, time, status):
        if not self.recv_queue.empty():
            chunk = self.recv_queue.get()

            # Ensure chunk is int16 and clip to valid range
            chunk_int16 = np.clip(chunk, -32768, 32767).astype(np.int16)

            if len(chunk_int16) < len(outdata):
                outdata[: len(chunk_int16), 0] = chunk_int16
                outdata[len(chunk_int16) :] = 0
            else:
                outdata[:, 0] = chunk_int16[: len(outdata)]
        else:
            outdata[:] = 0

    def play_audio(self):
        with sd.OutputStream(
            samplerate=self.args.sample_rate,
            channels=1,
            dtype="int16",
            callback=self.audio_out_callback,
            blocksize=self.args.chunk_size,
        ):
            while not self.stop_event.is_set():
                time.sleep(0.1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio Streaming Client")
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Audio sample rate in Hz. Default is 16000.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024,
        help="The size of audio chunks in samples. Default is 1024.",
    )
    parser.add_argument(
        "--api_url", type=str, required=True, help="The URL of the API endpoint."
    )
    parser.add_argument(
        "--auth_token",
        type=str,
        required=True,
        help="Authentication token for the API.",
    )

    args = parser.parse_args()
    client_args = AudioStreamingClientArguments(**vars(args))
    client = AudioStreamingClient(client_args)
    client.start()
