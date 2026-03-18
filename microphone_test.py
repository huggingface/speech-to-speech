from dataclasses import dataclass, field
from queue import Empty, Queue
import time

import numpy as np
import sounddevice as sd
from transformers import HfArgumentParser


@dataclass
class MicrophoneTestArguments:
    sample_rate: int = field(
        default=16000,
        metadata={"help": "Microphone sample rate in Hz. Default is 16000."},
    )
    blocksize: int = field(
        default=1024,
        metadata={"help": "Audio block size. Default is 1024."},
    )
    input_device: int | None = field(
        default=None,
        metadata={"help": "Optional sounddevice input device index."},
    )
    monitor: bool = field(
        default=False,
        metadata={"help": "If true, route microphone audio to the default speakers."},
    )


def _format_meter(level: float, width: int = 40) -> str:
    clamped = max(0.0, min(level, 1.0))
    filled = int(clamped * width)
    return "#" * filled + "-" * (width - filled)


def microphone_test(
    sample_rate: int = 16000,
    blocksize: int = 1024,
    input_device: int | None = None,
    monitor: bool = False,
):
    level_queue: Queue[float] = Queue()

    default_input, default_output = sd.default.device
    selected_input = default_input if input_device is None else input_device
    input_info = sd.query_devices(selected_input, "input")

    print("Microphone test started")
    print(f"Input device: {input_info['name']}")
    if monitor:
        output_info = sd.query_devices(default_output, "output")
        print(f"Output device: {output_info['name']}")
        print("Monitoring is enabled. Use headphones to avoid feedback.")
    print("Speak into the microphone. Press Ctrl+C to stop.")

    def input_callback(indata, frames, time_info, status):
        if status:
            print(status)
        audio = indata.astype(np.float32) / 32768.0
        level = float(np.sqrt(np.mean(np.square(audio))))
        level_queue.put(level)

    def duplex_callback(indata, outdata, frames, time_info, status):
        if status:
            print(status)
        audio = indata.astype(np.float32) / 32768.0
        level = float(np.sqrt(np.mean(np.square(audio))))
        level_queue.put(level)
        outdata[:] = indata

    stream_kwargs = {
        "samplerate": sample_rate,
        "blocksize": blocksize,
        "dtype": "int16",
        "channels": 1,
        "device": (selected_input, default_output) if monitor else selected_input,
    }

    stream_cls = sd.Stream if monitor else sd.InputStream
    callback = duplex_callback if monitor else input_callback

    with stream_cls(callback=callback, **stream_kwargs):
        last_level = 0.0
        while True:
            try:
                last_level = level_queue.get(timeout=0.2)
            except Empty:
                pass

            meter = _format_meter(last_level * 8.0)
            print(f"\rLevel [{meter}] {last_level:.3f}", end="", flush=True)
            time.sleep(0.05)


if __name__ == "__main__":
    parser = HfArgumentParser((MicrophoneTestArguments,))
    (microphone_test_kwargs,) = parser.parse_args_into_dataclasses()
    try:
        microphone_test(**vars(microphone_test_kwargs))
    except KeyboardInterrupt:
        print("\nMicrophone test stopped")