"""
TTS Benchmarking Script

Benchmarks Text-to-Speech (TTS) handlers to compare performance.
Measures: warmup time, inference time, time-to-first-chunk, audio duration, and RTF.

Usage:
    python benchmark_tts.py --text "Hello world" --iterations 3
    python benchmark_tts.py --handlers kokoro parler
"""

import argparse
import logging
import time
import numpy as np
from typing import List, Dict, Any
import json
from queue import Queue
from threading import Event
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16000


class BenchmarkResult:
    """Stores benchmark results for a single TTS handler."""

    def __init__(self, handler_name: str):
        self.handler_name = handler_name
        self.warmup_time = 0.0
        self.inference_times = []
        self.time_to_first_chunk = []
        self.audio_durations = []
        self.errors = []

    def add_inference(self, time_taken: float, audio_duration: float, ttfc: float = None):
        self.inference_times.append(time_taken)
        self.audio_durations.append(audio_duration)
        if ttfc is not None:
            self.time_to_first_chunk.append(ttfc)

    def add_error(self, error: str):
        self.errors.append(error)

    def get_stats(self) -> Dict[str, Any]:
        if not self.inference_times:
            return {
                "handler": self.handler_name,
                "status": "failed",
                "errors": self.errors,
            }

        avg_time = float(np.mean(self.inference_times))
        avg_audio = float(np.mean(self.audio_durations))
        avg_rtf = avg_audio / avg_time if avg_time > 0 else 0.0

        stats = {
            "handler": self.handler_name,
            "warmup_time": self.warmup_time,
            "avg_inference_time": avg_time,
            "min_inference_time": float(np.min(self.inference_times)),
            "max_inference_time": float(np.max(self.inference_times)),
            "std_inference_time": float(np.std(self.inference_times)),
            "avg_audio_duration": avg_audio,
            "min_audio_duration": float(np.min(self.audio_durations)),
            "max_audio_duration": float(np.max(self.audio_durations)),
            "std_audio_duration": float(np.std(self.audio_durations)),
            "avg_rtf": avg_rtf,
            "total_iterations": len(self.inference_times),
            "errors": self.errors,
        }

        if self.time_to_first_chunk:
            stats["avg_time_to_first_chunk"] = float(np.mean(self.time_to_first_chunk))
            stats["min_time_to_first_chunk"] = float(np.min(self.time_to_first_chunk))
            stats["max_time_to_first_chunk"] = float(np.max(self.time_to_first_chunk))
            stats["std_time_to_first_chunk"] = float(np.std(self.time_to_first_chunk))

        return stats


def benchmark_handler(handler_name: str, text: str, iterations: int, handler_kwargs: Dict[str, Any] = None) -> BenchmarkResult:
    logger.info(f"Benchmarking {handler_name}...")
    result = BenchmarkResult(handler_name)

    try:
        stop_event = Event()
        should_listen = Event()
        queue_in = Queue()
        queue_out = Queue()

        handler = None
        setup_kwargs = handler_kwargs or {}

        start_setup = time.perf_counter()

        if handler_name == "kokoro":
            from TTS.kokoro_handler import KokoroTTSHandler
            setup_kwargs = {"device": "auto", **setup_kwargs}
            handler = KokoroTTSHandler(
                stop_event,
                queue_in=queue_in,
                queue_out=queue_out,
                setup_args=(should_listen,),
                setup_kwargs=setup_kwargs,
            )
        elif handler_name == "parler":
            from TTS.parler_handler import ParlerTTSHandler
            setup_kwargs = {"device": "cuda", "torch_dtype": "float16", **setup_kwargs}
            handler = ParlerTTSHandler(
                stop_event,
                queue_in=queue_in,
                queue_out=queue_out,
                setup_args=(should_listen,),
                setup_kwargs=setup_kwargs,
            )
        elif handler_name == "pocket_tts":
            from TTS.pocket_tts_handler import PocketTTSHandler
            setup_kwargs = {"device": "cpu", **setup_kwargs}
            handler = PocketTTSHandler(
                stop_event,
                queue_in=queue_in,
                queue_out=queue_out,
                setup_args=(should_listen,),
                setup_kwargs=setup_kwargs,
            )
        elif handler_name == "melo":
            from TTS.melo_handler import MeloTTSHandler
            setup_kwargs = {"device": "cuda", "language": "en", "speaker_to_id": "en", **setup_kwargs}
            handler = MeloTTSHandler(
                stop_event,
                queue_in=queue_in,
                queue_out=queue_out,
                setup_args=(should_listen,),
                setup_kwargs=setup_kwargs,
            )
        elif handler_name == "qwen3-tts":
            from TTS.qwen3_tts_handler import Qwen3TTSHandler
            setup_kwargs = {
                "device": "cuda",
                "model_name": "Qwen3-TTS-12Hz-0.6B-Base",
                **setup_kwargs,
            }
            handler = Qwen3TTSHandler(
                stop_event,
                queue_in=queue_in,
                queue_out=queue_out,
                setup_args=(should_listen,),
                setup_kwargs=setup_kwargs,
            )
        else:
            raise ValueError(f"Unknown handler: {handler_name}")

        result.warmup_time = time.perf_counter() - start_setup
        logger.info(f"Handler {handler_name} initialized and warmed up in {result.warmup_time:.3f}s")

        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations} for {handler_name}")
            start_time = time.perf_counter()
            time_to_first_chunk = None
            first_output = True
            total_samples = 0

            for chunk in handler.process(text):
                if first_output:
                    time_to_first_chunk = time.perf_counter() - start_time
                    first_output = False
                if chunk is None:
                    continue
                try:
                    total_samples += len(chunk)
                except Exception:
                    pass

            end_time = time.perf_counter()
            time_taken = end_time - start_time
            audio_duration = total_samples / DEFAULT_SAMPLE_RATE if total_samples > 0 else 0.0

            result.add_inference(time_taken, audio_duration, time_to_first_chunk)
            ttfc_str = f", TTFC: {time_to_first_chunk:.4f}s" if time_to_first_chunk else ""
            logger.info(
                f"  Time: {time_taken:.4f}s{ttfc_str}, Audio: {audio_duration:.2f}s, RTF: {audio_duration / time_taken if time_taken > 0 else 0:.2f}"
            )

        handler.cleanup()
        stop_event.set()

    except Exception as e:
        logger.error(f"Error benchmarking {handler_name}: {e}", exc_info=True)
        result.add_error(str(e))

    return result


def print_results(results: List[BenchmarkResult]):
    print("\n" + "=" * 80)
    print("TTS BENCHMARK RESULTS")
    print("=" * 80)

    for result in results:
        stats = result.get_stats()
        print(f"\nHandler: {stats['handler']}")
        print("-" * 80)

        if stats.get("status") == "failed":
            print("  Status: FAILED")
            print(f"  Errors: {stats['errors']}")
            continue

        print(f"  Warmup Time:          {stats['warmup_time']:.4f}s")
        print(f"  Avg Inference Time:   {stats['avg_inference_time']:.4f}s")
        print(f"  Min Inference Time:   {stats['min_inference_time']:.4f}s")
        print(f"  Max Inference Time:   {stats['max_inference_time']:.4f}s")
        print(f"  Std Deviation:        {stats['std_inference_time']:.4f}s")

        print(f"  Avg Audio Duration:   {stats['avg_audio_duration']:.2f}s")
        print(f"  Min Audio Duration:   {stats['min_audio_duration']:.2f}s")
        print(f"  Max Audio Duration:   {stats['max_audio_duration']:.2f}s")
        print(f"  Std Audio Duration:   {stats['std_audio_duration']:.4f}s")
        print(f"  Avg RTF:              {stats['avg_rtf']:.2f}")

        if "avg_time_to_first_chunk" in stats:
            print("\n  Time to First Chunk:")
            print(f"    Avg TTFC:           {stats['avg_time_to_first_chunk']:.4f}s")
            print(f"    Min TTFC:           {stats['min_time_to_first_chunk']:.4f}s")
            print(f"    Max TTFC:           {stats['max_time_to_first_chunk']:.4f}s")
            print(f"    Std TTFC:           {stats['std_time_to_first_chunk']:.4f}s")

        print(f"\n  Total Iterations:     {stats['total_iterations']}")

        if stats["errors"]:
            print(f"  Errors: {stats['errors']}")

    print("\n" + "=" * 80)
    print("COMPARISON (Average Inference Time)")
    print("=" * 80)

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
    data = {
        "results": [r.get_stats() for r in results],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark TTS handlers")
    parser.add_argument(
        "--text",
        type=str,
        default="Hello from the speech to speech benchmark. This is a latency test.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--handlers",
        nargs="+",
        default=["kokoro", "parler"],
        help="List of handlers to benchmark (default: kokoro parler)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations per handler (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tts_benchmark_results.json",
        help="Output JSON file for results (default: tts_benchmark_results.json)",
    )

    args = parser.parse_args()

    if not args.handlers:
        logger.error("No handlers provided")
        return

    results = []
    for handler_name in args.handlers:
        result = benchmark_handler(handler_name, args.text, args.iterations)
        results.append(result)

    print_results(results)
    save_results(results, args.output)

    logger.info("TTS benchmarking complete!")


if __name__ == "__main__":
    main()
