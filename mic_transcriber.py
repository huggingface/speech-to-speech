import os
os.environ['KERAS_BACKEND'] = 'torch'

import queue
import threading
import time
import datetime
import argparse
import torch
import numpy as np
import sounddevice as sd
from pathlib import Path
import platform
from utils.utils import int2float
from lightning_whisper_mlx import LightningWhisperMLX

from VAD.vad_iterator import VADIterator

class SimpleSpeechTranscriber:
    def __init__(self, args):
        # Initialize parameters
        self.sample_rate = args.sample_rate
        if self.sample_rate == 16000:
            self.chunk_size = 512
        elif self.sample_rate == 8000:
            self.chunk_size = 256
        else:
            raise ValueError(f"Unsupported sample rate: {self.sample_rate}")
        self.device = args.device
        self.stop_event = threading.Event()
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        
        # Set up the output directory and file
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = Path(args.output_dir) / f"transcription_{timestamp}.txt"
        
        # Initialize VAD model
        self.model, _ = torch.hub.load(
            "snakers4/silero-vad", 
            "silero_vad",
            force_reload=False
        )
        
        self.vad_iterator = VADIterator(
            model=self.model,
            threshold=args.vad_threshold,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=args.min_silence_ms,
            speech_pad_ms=args.speech_pad_ms
        )
        
        self.model = LightningWhisperMLX(
            model=args.model_name,
            batch_size=6,
            quant=None
        )
        self.warmup_lightning_whisper()
    
    def audio_callback(self, indata, frames, time, status):
        """Callback function for the audio stream"""
        if status:
            print(f"Audio callback status: {status}")       
        self.audio_queue.put(indata.copy())
    
    def process_audio(self):
        """Process audio chunks and detect speech with VAD"""
        while not self.stop_event.is_set():
            try:
                audio_chunk = self.audio_queue.get(timeout=0.5)
                
                audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
                audio_float32 = int2float(audio_int16)
                # Process with VAD
                vad_output = self.vad_iterator(torch.from_numpy(audio_float32))
                
                if vad_output is not None and len(vad_output) > 0:
                    # Speech detected and ended, concatenate the audio chunks
                    speech_audio = torch.cat(vad_output).cpu().numpy()
                    
                    # Create a timestamp for when this speech was detected
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Put in transcription queue
                    self.transcription_queue.put((speech_audio, timestamp))
                    print(f"Speech detected! ({len(speech_audio)/self.sample_rate:.2f}s)")
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio processing: {e}")
    
    def transcribe_speech(self):
        """Transcribe detected speech segments"""
        while not self.stop_event.is_set():
            try:
                speech_audio, timestamp = self.transcription_queue.get(timeout=0.5)
                
                # For Lightning Whisper MLX
                transcription_dict = self.model.transcribe(speech_audio)
                transcription = transcription_dict["text"].strip()
                
                # Clear MPS cache after inference
                if self.device == "mps":
                    torch.mps.empty_cache()
                
                # Save to file with timestamp
                with open(self.output_file, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] {transcription}\n")
                
                print(f"Transcription: {transcription}")
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in transcription: {e}")
    
    def warmup_lightning_whisper(self):
        """Warm up the Lightning Whisper MLX model"""
        print("Warming up Lightning Whisper MLX model...")
        
        # Warmup step
        n_steps = 1
        dummy_input = np.array([0] * 512)
        
        for _ in range(n_steps):
            _ = self.model.transcribe(dummy_input)["text"].strip()
        
        print("Lightning Whisper MLX model warmed up!")
    
    def run(self):
        """Start the transcription pipeline"""
        print(f"Starting audio capture (sample rate: {self.sample_rate}Hz)")
        print(f"Transcriptions will be saved to: {self.output_file}")
        
        # Start the processing threads
        audio_thread = threading.Thread(target=self.process_audio)
        transcription_thread = threading.Thread(target=self.transcribe_speech)
        
        audio_thread.start()
        transcription_thread.start()
        
        try:
            # Start the audio stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                dtype="int16",
                blocksize=self.chunk_size
            ):
                print("Listening to microphone... Press Ctrl+C to stop.")
                while not self.stop_event.is_set():
                    time.sleep(0.001)
        
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            # Clean up
            self.stop_event.set()
            audio_thread.join()
            transcription_thread.join()
            print("Stopped.")

def main():
    parser = argparse.ArgumentParser(description="Simple speech transcription pipeline")
    
    # Audio parameters
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")
    
    # VAD parameters
    parser.add_argument("--vad-threshold", type=float, default=0.3, help="VAD threshold")
    parser.add_argument("--min-silence-ms", type=int, default=500, help="Minimum silence duration in ms")
    parser.add_argument("--min-speech-ms", type=int, default=500, help="Minimum speech duration in ms")
    parser.add_argument("--speech-pad-ms", type=int, default=500, help="Speech padding in ms")
    
    parser.add_argument("--model-name", type=str, default="large-v3", # large v3 has the best performance for french
                        help="Model name for lightning-whisper-mlx")  # it's a bit slow but this doesn't need to be super fast
    
    # Output parameters
    parser.add_argument("--output-dir", type=str, default="transcripts", help="Output directory for transcriptions")
    
    # Device parameters
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu", 
                        help="Device to use for models (cpu, mps)")
    
    args = parser.parse_args()
    
    # Check if running on Mac and give info about MPS
    if platform == "darwin" and torch.backends.mps.is_available():
        print("Running on Mac with Metal Performance Shaders (MPS) support")
        print("Using device:", args.device)
    
    transcriber = SimpleSpeechTranscriber(args)
    transcriber.run()

if __name__ == "__main__":
    main() 