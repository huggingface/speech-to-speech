import threading
from queue import Queue
import sounddevice as sd
import numpy as np
import requests
import base64
import time
from dataclasses import dataclass, field

@dataclass
class AudioStreamingClientArguments:
    sample_rate: int = field(default=16000, metadata={"help": "Audio sample rate in Hz. Default is 16000."})
    chunk_size: int = field(default=512, metadata={"help": "The size of audio chunks in samples. Default is 1024."})
    api_url: str = field(default="https://yxfmjcvuzgi123sw.us-east-1.aws.endpoints.huggingface.cloud", metadata={"help": "The URL of the API endpoint."})
    auth_token: str = field(default="your_auth_token", metadata={"help": "Authentication token for the API."})

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
            "Content-Type": "application/json"
        }
        self.session_state = "idle"  # Possible states: idle, sending, processing, waiting

    def start(self):
        print("Starting audio streaming...")
        
        send_thread = threading.Thread(target=self.send_audio)
        play_thread = threading.Thread(target=self.play_audio)

        with sd.InputStream(samplerate=self.args.sample_rate, channels=1, dtype='int16', callback=self.audio_callback, blocksize=self.args.chunk_size):
            send_thread.start()
            play_thread.start()

            try:
                input("Press Enter to stop streaming...")
            except KeyboardInterrupt:
                print("\nStreaming interrupted by user.")
            finally:
                self.stop_event.set()
                send_thread.join()
                play_thread.join()
                print("Audio streaming stopped.")

    def audio_callback(self, indata, frames, time, status):
        self.send_queue.put(indata.copy())

    def send_audio(self):
        buffer = b''
        while not self.stop_event.is_set():
            if self.session_state != "processing" and not self.send_queue.empty():
                chunk = self.send_queue.get().tobytes()
                buffer += chunk
                if len(buffer) >= self.args.chunk_size * 2:  # * 2 because of int16
                    self.send_request(buffer)
                    buffer = b''
            else:
                self.send_request()
                time.sleep(0.1)

    def send_request(self, audio_data=None):
        payload = {"input_type": "speech",
                   "inputs": ""}

        if audio_data is not None:
            print("Sending audio data")
            payload["inputs"] = base64.b64encode(audio_data).decode('utf-8')

        if self.session_id:
            payload["session_id"] = self.session_id
            payload["request_type"] = "continue"
        else:
            payload["request_type"] = "start"

        try:
            response = requests.post(self.args.api_url, headers=self.headers, json=payload)
            response_data = response.json()

            if "session_id" in response_data:
                self.session_id = response_data["session_id"]

            if "status" in response_data and response_data["status"] == "processing":
                print("Processing audio data")
                self.session_state = "processing"

            if "output" in response_data and response_data["output"]:
                print("Received audio data")
                self.session_state = "processing"  # Set state to processing when we start receiving audio
                audio_bytes = base64.b64decode(response_data["output"])
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
                # Split the audio into smaller chunks for playback
                for i in range(0, len(audio_np), self.args.chunk_size):
                    chunk = audio_np[i:i+self.args.chunk_size]
                    self.recv_queue.put(chunk)

            if "status" in response_data and response_data["status"] == "completed":
                print("Completed audio processing")
                self.session_state = None
                self.session_id = None
                while not self.recv_queue.empty():
                    time.sleep(0.01)  # wait for the queue to empty
                while not self.send_queue.empty():
                    _ = self.send_queue.get()  # Clear the queue

        except Exception as e:
            print(f"Error sending request: {e}")
            self.session_state = "idle"  # Reset state to idle in case of error

    def play_audio(self):
        def audio_callback(outdata, frames, time, status):
            if not self.recv_queue.empty():
                chunk = self.recv_queue.get()
                
                # Ensure chunk is int16 and clip to valid range
                chunk_int16 = np.clip(chunk, -32768, 32767).astype(np.int16)
                
                if len(chunk_int16) < len(outdata):
                    outdata[:len(chunk_int16), 0] = chunk_int16
                    outdata[len(chunk_int16):] = 0
                else:
                    outdata[:, 0] = chunk_int16[:len(outdata)]
            else:
                outdata[:] = 0

        with sd.OutputStream(samplerate=self.args.sample_rate, channels=1, dtype='int16', callback=audio_callback, blocksize=self.args.chunk_size):
            while not self.stop_event.is_set():
                time.sleep(0.01)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio Streaming Client")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate in Hz. Default is 16000.")
    parser.add_argument("--chunk_size", type=int, default=1024, help="The size of audio chunks in samples. Default is 1024.")
    parser.add_argument("--api_url", type=str, required=True, help="The URL of the API endpoint.")
    parser.add_argument("--auth_token", type=str, required=True, help="Authentication token for the API.")

    args = parser.parse_args()
    client_args = AudioStreamingClientArguments(**vars(args))
    client = AudioStreamingClient(client_args)
    client.start()