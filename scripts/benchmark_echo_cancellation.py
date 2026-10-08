"""Measure CPU seconds per audio second; optional pacing includes idle wakeups.

Run from a source checkout with PYTHONPATH=src. Imports, data generation,
and processor construction happen before timing. CPU % means one full core.
"""

import argparse
import importlib.metadata
import json
import platform
import time

import numpy as np

from speech_to_speech.api.openai_realtime.echo_canceller import EchoCanceller


def run(backend, scenario, duration, paced):
    rate, block = 16000, 1024
    samples = int(duration * rate) // block * block
    rng = np.random.default_rng(0)
    played = (rng.standard_normal(samples) * 3000).astype(np.int16)
    echo = np.zeros(samples, dtype=np.int16)
    echo[320:] = (played[:-320] * 0.3).astype(np.int16)
    if scenario == "idle":
        played[:] = 0
        echo[:] = 0
    processor = None
    if backend == "pywebrtc":
        processor = EchoCanceller(rate, rate)
    rendered = [played[i : i + block].tobytes() for i in range(0, samples, block)]
    captured = [echo[i : i + block].tobytes() for i in range(0, samples, block)]
    cleaned = bytearray()
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    for index, (far, near) in enumerate(zip(rendered, captured)):
        if processor:
            processor.render(far, 0.01)
            near = processor.capture(near, 0.01)
        cleaned.extend(near)
        if paced:
            time.sleep(max(0, wall_start + (index + 1) * block / rate - time.perf_counter()))
    cpu = time.process_time() - cpu_start
    wall = time.perf_counter() - wall_start
    reduction = None
    if scenario == "echo":
        output = np.frombuffer(bytes(cleaned), dtype=np.int16).astype(float)
        tail = min(rate, len(output))
        reduction = float(
            10
            * np.log10(
                np.mean(echo[len(output) - tail : len(output)].astype(float) ** 2)
                / (np.mean(output[-tail:] ** 2) + 1e-9)
            )
        )
    return dict(
        backend=backend,
        scenario=scenario,
        audio_seconds=samples / rate,
        cpu_seconds=cpu,
        wall_seconds=wall,
        core_percent=100 * cpu / (samples / rate),
        reduction_db=reduction,
        output_samples=len(cleaned) // 2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--paced", action="store_true")
    args = parser.parse_args()
    if args.seconds < 1 or args.repeats < 1:
        parser.error("seconds and repeats must be at least 1")
    packages = ["numpy", "pywebrtc-audio"]
    backends = ["off", "pywebrtc"]
    print(
        json.dumps(
            dict(
                machine=platform.node(),
                platform=platform.platform(),
                versions={p: importlib.metadata.version(p) for p in packages},
                paced=args.paced,
            )
        ),
        flush=True,
    )
    for repeat in range(args.repeats):
        for scenario in ("idle", "echo"):
            for backend in backends:
                print(json.dumps(dict(repeat=repeat, **run(backend, scenario, args.seconds, args.paced))), flush=True)
