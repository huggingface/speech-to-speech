"""Microphone/file demo: python -m speech_to_speech.diarization.demo --help."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from collections import deque
from dataclasses import asdict
from queue import Empty, Full, Queue
from threading import Event
from time import monotonic, sleep

import numpy as np

from .streaming import SpeakerSegment, StreamingDiarizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show streaming speaker activity from a microphone or recording.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", help="Local audio file; converted to mono at the model's sample rate.")
    source.add_argument("--microphone", action="store_true")
    parser.add_argument("--model", required=True, help="Transformers diarization checkpoint ID or local directory.")
    parser.add_argument("--revision", help="Optional checkpoint revision; used for both model and processor.")
    parser.add_argument("--device", default="cpu", help="For example cpu, cuda, or mps.")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument(
        "--streaming-mode", choices=("low_latency", "very_low_latency", "ultra_low_latency"), default="low_latency"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--input-device", type=int, help="Sounddevice input device index.")
    parser.add_argument("--realtime", action="store_true", help="Pace file playback at real time.")
    parser.add_argument(
        "--json", action="store_true", help="Print JSON Lines updates instead of a live terminal display."
    )
    parser.add_argument(
        "--transcribe", metavar="ASR_MODEL", help="File only: also align timestamped Whisper words after diarization."
    )
    return parser


def microphone_blocks(sample_rate: int, device: int | None, stop: Event):
    import sounddevice as sd

    queue: Queue[np.ndarray] = Queue(maxsize=50)
    overflow = Event()

    def callback(indata, frames, time_info, status):
        if status:
            overflow.set()
            return
        try:
            queue.put_nowait(indata[:, 0].copy())
        except Full:
            overflow.set()

    # Inference runs in the consumer, never in the audio callback. A dropped
    # block would corrupt both timestamps and cache continuity, so fail clearly.
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=sample_rate // 10,
        device=device,
        callback=callback,
    ):
        while not stop.is_set():
            if overflow.is_set():
                raise RuntimeError(
                    "Microphone audio was dropped; use a faster device/model mode and restart the session."
                )
            try:
                yield queue.get(timeout=0.1)
            except Empty:
                continue


def file_blocks(audio: np.ndarray, sample_rate: int, realtime: bool):
    started = monotonic()
    block_size = max(1, sample_rate // 10)
    for start in range(0, len(audio), block_size):
        block = audio[start : start + block_size]
        if realtime:
            sleep(max(0, started + (start + len(block)) / sample_rate - monotonic()))
        yield block


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.transcribe and not args.audio:
        parser.error("--transcribe requires --audio")
    if args.realtime and not args.audio:
        parser.error("--realtime requires --audio")
    diarizer = StreamingDiarizer.from_pretrained(
        args.model,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        streaming_mode=args.streaming_mode,
        threshold=args.threshold,
    )
    stop = Event()
    print(f"Input buffer latency: {diarizer.processor.streaming_latency_ms} ms (plus compute).", file=sys.stderr)
    if args.audio:
        import librosa

        audio, _ = librosa.load(args.audio, sr=diarizer.sample_rate, mono=True)
        blocks = file_blocks(audio, diarizer.sample_rate, args.realtime)
    else:
        print("Listening; press Ctrl-C to finish.", file=sys.stderr)
        blocks = microphone_blocks(diarizer.sample_rate, args.input_device, stop)

    history: deque[SpeakerSegment] = deque(maxlen=12)
    # Full history is needed only for optional file transcription.
    transcript_segments: list[SpeakerSegment] = []
    live = None
    if not args.json:
        from rich.live import Live

        live = Live(refresh_per_second=8)
        live.start()

    def report(segments: list[SpeakerSegment], *, final: bool = False):
        history.extend(segments)
        if args.transcribe:
            transcript_segments.extend(segments)
        if args.json:
            print(
                json.dumps(
                    {
                        "type": "diarization",
                        "processed_seconds": diarizer.processed_seconds,
                        "active_speakers": diarizer.active_speakers,
                        "segments": [asdict(segment) for segment in segments],
                        "final": final,
                    }
                ),
                flush=True,
            )
        else:
            from rich.table import Table

            active = ", ".join(f"speaker_{speaker}" for speaker in diarizer.active_speakers) or "silence"
            table = Table(title=f"{diarizer.processed_seconds:.2f}s — {active}")
            table.add_column("Speaker")
            table.add_column("Start")
            table.add_column("End")
            for segment in history:
                table.add_row(f"speaker_{segment.speaker}", f"{segment.start:.2f}s", f"{segment.end:.2f}s")
            assert live is not None
            live.update(table)

    # Defer Ctrl-C until the current forward completes so its cache and frame
    # cursor cannot be left half-updated before flushing the final audio.
    previous_sigint = signal.signal(signal.SIGINT, lambda signum, frame: stop.set())
    try:
        for block in blocks:
            before = diarizer.processed_seconds
            segments = diarizer.push(block, sample_rate=diarizer.sample_rate)
            if diarizer.processed_seconds != before:
                report(segments)
            if stop.is_set():
                break
        report(diarizer.finish(), final=True)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        blocks.close()
        if live is not None:
            live.stop()

    if args.transcribe and stop.is_set():
        print("Recording interrupted; skipping full-file transcription.", file=sys.stderr)
        return
    if args.transcribe:
        from transformers import pipeline

        from .alignment import align_words

        asr = pipeline("automatic-speech-recognition", model=args.transcribe, device=args.device)
        result = asr(
            {"array": audio, "sampling_rate": diarizer.sample_rate}, return_timestamps="word", chunk_length_s=30
        )
        if not isinstance(result, dict):
            raise RuntimeError("Expected a single timestamped ASR result")
        for word in align_words(result["chunks"], transcript_segments):
            if args.json:
                print(json.dumps({"type": "word", **asdict(word)}), flush=True)
            else:
                labels = "+".join(f"speaker_{speaker}" for speaker in word.speakers) or "unknown"
                print(f"{word.start:7.2f}–{word.end:7.2f} [{labels}] {word.text}")


if __name__ == "__main__":
    main()
