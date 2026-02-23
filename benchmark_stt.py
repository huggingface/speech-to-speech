"""
STT Benchmarking Script

This script benchmarks different Speech-to-Text (STT) handlers to compare their performance.
Measures: inference time, warmup time, memory usage, and transcription quality.

Usage:
    python benchmark_stt.py --audio_file path/to/audio.wav --iterations 10
    python benchmark_stt.py --audio_file path/to/audio.wav --handlers whisper mlx-audio-whisper
"""

import argparse
import logging
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import json
from queue import Queue
from threading import Event
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BenchmarkResult:
    """Stores benchmark results for a single STT handler."""

    def __init__(self, handler_name: str):
        self.handler_name = handler_name
        self.warmup_time = 0.0
        self.inference_times = []
        self.time_to_first_token = []
        self.transcriptions = []
        self.errors = []

    def add_inference(self, time_taken: float, transcription: str, ttft: float = None):
        self.inference_times.append(time_taken)
        self.transcriptions.append(transcription)
        if ttft is not None:
            self.time_to_first_token.append(ttft)

    def add_error(self, error: str):
        self.errors.append(error)

    def get_stats(self) -> Dict[str, Any]:
        """Calculate statistics from benchmark results."""
        if not self.inference_times:
            return {
                "handler": self.handler_name,
                "status": "failed",
                "errors": self.errors,
            }

        stats = {
            "handler": self.handler_name,
            "warmup_time": self.warmup_time,
            "avg_inference_time": np.mean(self.inference_times),
            "min_inference_time": np.min(self.inference_times),
            "max_inference_time": np.max(self.inference_times),
            "std_inference_time": np.std(self.inference_times),
            "total_iterations": len(self.inference_times),
            "errors": self.errors,
            "sample_transcription": self.transcriptions[0] if self.transcriptions else None,
        }

        # Add time to first token stats if available
        if self.time_to_first_token:
            stats["avg_time_to_first_token"] = np.mean(self.time_to_first_token)
            stats["min_time_to_first_token"] = np.min(self.time_to_first_token)
            stats["max_time_to_first_token"] = np.max(self.time_to_first_token)
            stats["std_time_to_first_token"] = np.std(self.time_to_first_token)

        return stats


def load_audio(audio_path: str) -> np.ndarray:
    """Load audio file and return as numpy array."""
    logger.info(f"Loading audio from: {audio_path}")
    audio, sample_rate = sf.read(audio_path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample to 16kHz if needed (most whisper models expect 16kHz)
    if sample_rate != 16000:
        logger.warning(f"Audio sample rate is {sample_rate}Hz, resampling to 16000Hz")
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        except ImportError:
            logger.error("librosa not installed. Please install it with: pip install librosa")
            logger.error("Attempting scipy resampling as fallback...")
            from scipy import signal
            # Calculate resampling ratio
            num_samples = int(len(audio) * 16000 / sample_rate)
            audio = signal.resample(audio, num_samples)

    return audio.astype(np.float32)


def benchmark_handler(
    handler_name: str,
    audio: np.ndarray,
    iterations: int,
    handler_kwargs: Dict[str, Any] = None
) -> BenchmarkResult:
    """Benchmark a single STT handler."""
    logger.info(f"Benchmarking {handler_name}...")
    result = BenchmarkResult(handler_name)

    try:
        # Create queues and events for handler
        stop_event = Event()
        queue_in = Queue()
        queue_out = Queue()

        # Import and instantiate handler
        handler = None
        if handler_name == "whisper":
            from STT.whisper_stt_handler import WhisperSTTHandler
            setup_kwargs = handler_kwargs or {
                "model_name": "distil-whisper/distil-large-v3",
                "device": "cuda",
                "torch_dtype": "float16",
            }
            handler = WhisperSTTHandler(
                stop_event,
                queue_in=queue_in,
                queue_out=queue_out,
                setup_kwargs=setup_kwargs
            )

        elif handler_name == "whisper-mlx":
            from STT.lightning_whisper_mlx_handler import LightningWhisperSTTHandler
            setup_kwargs = handler_kwargs or {
                "model_name": "large-v3",
                "device": "mps",
            }
            handler = LightningWhisperSTTHandler(
                stop_event,
                queue_in=queue_in,
                queue_out=queue_out,
                setup_kwargs=setup_kwargs
            )

        elif handler_name == "mlx-audio-whisper":
            from STT.mlx_audio_whisper_handler import MLXAudioWhisperSTTHandler
            setup_kwargs = handler_kwargs or {
                "model_name": "mlx-community/whisper-large-v3-turbo",
            }
            handler = MLXAudioWhisperSTTHandler(
                stop_event,
                queue_in=queue_in,
                queue_out=queue_out,
                setup_kwargs=setup_kwargs
            )

        elif handler_name == "faster-whisper":
            from STT.faster_whisper_handler import FasterWhisperSTTHandler
            setup_kwargs = handler_kwargs or {
                "model_name": "large-v3",
                "device": "auto",
                "compute_type": "float16",
            }
            handler = FasterWhisperSTTHandler(
                stop_event,
                queue_in=queue_in,
                queue_out=queue_out,
                setup_kwargs=setup_kwargs
            )

        elif handler_name == "moonshine":
            from STT.moonshine_handler import MoonshineSTTHandler
            handler = MoonshineSTTHandler(
                stop_event,
                queue_in=queue_in,
                queue_out=queue_out,
            )

        elif handler_name == "parakeet-tdt":
            from STT.parakeet_tdt_handler import ParakeetTDTSTTHandler
            setup_kwargs = handler_kwargs or {
                "device": "mps",
                "enable_live_transcription": False,
            }
            handler = ParakeetTDTSTTHandler(
                stop_event,
                queue_in=queue_in,
                queue_out=queue_out,
                setup_kwargs=setup_kwargs
            )

        elif handler_name == "parakeet-tdt-progressive":
            from STT.parakeet_tdt_handler import ParakeetTDTSTTHandler
            setup_kwargs = handler_kwargs or {
                "device": "mps",
                "enable_live_transcription": True,
                "live_transcription_update_interval": 0.25,
            }
            handler = ParakeetTDTSTTHandler(
                stop_event,
                queue_in=queue_in,
                queue_out=queue_out,
                setup_kwargs=setup_kwargs
            )
        else:
            raise ValueError(f"Unknown handler: {handler_name}")

        # Warmup is done in handler setup
        logger.info(f"Handler {handler_name} initialized and warmed up")

        # Additional warmup on the actual audio (excluded from timings)
        for _ in handler.process(audio):
            pass

        # Run benchmark iterations
        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations} for {handler_name}")

            start_time = time.perf_counter()

            # Process audio
            transcription = None
            time_to_first_token = None
            first_output = True

            for output in handler.process(audio):
                # Measure time to first token
                if first_output:
                    time_to_first_token = time.perf_counter() - start_time
                    first_output = False

                if isinstance(output, tuple):
                    transcription = output[0]  # (text, language)
                else:
                    transcription = output

            end_time = time.perf_counter()

            time_taken = end_time - start_time
            result.add_inference(time_taken, transcription, time_to_first_token)

            ttft_str = f", TTFT: {time_to_first_token:.4f}s" if time_to_first_token else ""
            logger.info(f"  Time: {time_taken:.4f}s{ttft_str}, Text: {transcription[:50]}...")

        # Cleanup
        handler.cleanup()
        stop_event.set()

    except Exception as e:
        logger.error(f"Error benchmarking {handler_name}: {e}", exc_info=True)
        result.add_error(str(e))

    return result


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results in a formatted table."""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)

    for result in results:
        stats = result.get_stats()
        print(f"\nHandler: {stats['handler']}")
        print("-" * 80)

        if stats.get("status") == "failed":
            print(f"  Status: FAILED")
            print(f"  Errors: {stats['errors']}")
            continue

        print(f"  Warmup Time:          {stats['warmup_time']:.4f}s")
        print(f"  Avg Inference Time:   {stats['avg_inference_time']:.4f}s")
        print(f"  Min Inference Time:   {stats['min_inference_time']:.4f}s")
        print(f"  Max Inference Time:   {stats['max_inference_time']:.4f}s")
        print(f"  Std Deviation:        {stats['std_inference_time']:.4f}s")

        # Print time to first token stats if available
        if 'avg_time_to_first_token' in stats:
            print(f"\n  Time to First Token:")
            print(f"    Avg TTFT:           {stats['avg_time_to_first_token']:.4f}s")
            print(f"    Min TTFT:           {stats['min_time_to_first_token']:.4f}s")
            print(f"    Max TTFT:           {stats['max_time_to_first_token']:.4f}s")
            print(f"    Std TTFT:           {stats['std_time_to_first_token']:.4f}s")

        print(f"\n  Total Iterations:     {stats['total_iterations']}")
        print(f"  Sample Transcription: {stats['sample_transcription']}")

        if stats['errors']:
            print(f"  Errors: {stats['errors']}")

    # Comparison table
    print("\n" + "="*80)
    print("COMPARISON (Average Inference Time)")
    print("="*80)

    successful_results = [r for r in results if r.inference_times]
    if successful_results:
        sorted_results = sorted(successful_results, key=lambda x: np.mean(x.inference_times))

        fastest = sorted_results[0]
        fastest_time = np.mean(fastest.inference_times)

        for result in sorted_results:
            avg_time = np.mean(result.inference_times)
            speedup = avg_time / fastest_time
            print(f"  {result.handler_name:25s}: {avg_time:.4f}s  ({speedup:.2f}x slower than fastest)")


def save_results(results: List[BenchmarkResult], output_file: str):
    """Save benchmark results to JSON file."""
    data = {
        "results": [r.get_stats() for r in results],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark STT handlers")
    parser.add_argument(
        "--audio_file",
        type=str,
        required=True,
        help="Path to audio file for benchmarking"
    )
    parser.add_argument(
        "--handlers",
        nargs="+",
        default=["whisper", "whisper-mlx", "mlx-audio-whisper", "faster-whisper", "parakeet-tdt", "parakeet-tdt-progressive"],
        help="List of handlers to benchmark (default: all)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations per handler (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stt_benchmark_results.json",
        help="Output JSON file for results (default: stt_benchmark_results.json)"
    )

    args = parser.parse_args()

    # Validate audio file exists
    if not Path(args.audio_file).exists():
        logger.error(f"Audio file not found: {args.audio_file}")
        return

    # Load audio
    audio = load_audio(args.audio_file)
    logger.info(f"Audio loaded: {len(audio)} samples, {len(audio)/16000:.2f}s duration")

    # Run benchmarks
    results = []
    for handler_name in args.handlers:
        result = benchmark_handler(handler_name, audio, args.iterations)
        results.append(result)

    # Print and save results
    print_results(results)
    save_results(results, args.output)

    logger.info("Benchmarking complete!")


if __name__ == "__main__":
    main()
