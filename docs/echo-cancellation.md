# Echo cancellation

`--echo-cancellation` makes the packaged `talk` and `local` clients remove
their own speaker playback from the microphone, so you can interrupt the
assistant by voice without headphones. It stays off by default; headphones or
a headset don't need it.

## How it works

The client uses `pywebrtc-audio`, which wraps WebRTC's echo canceller (AEC3).
Noise suppression and gain control stay off. The binding applies its
high-pass filter whenever echo cancellation is on.

The client uses playback and capture timestamps to align speaker references
with microphone samples. It processes complete 10 ms frames together in one
native call; partial frames carry over to the next callback.

On Windows, cancellation uses the default WASAPI input and output devices.
If you set `--input-device` or `--output-device`, select WASAPI devices; MME
does not provide capture timestamps. Shared-mode conversion keeps audio at
16 kHz even when the Windows mixer uses another rate. The `soxr` dependency
needs the [Microsoft Visual C++ x64 runtime](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

With cancellation enabled, a separate audio process owns the device streams.
Model and tool code in the parent cannot hold its Python execution lock.
Networking and tool execution stay in the parent. Python scripts that call the
client directly must protect their entry point with `if __name__ == "__main__":`
so the audio process can start safely.

`--echo-cancellation` cannot be combined with
`--local_audio_block_mic_during_playback`, which pauses the microphone during
playback and so disables voice interruptions.

## Limits

- Echo cancellation in the packaged client requires `--send-rate` and
  `--recv-rate` to both be 16 kHz, their default.
- The client does not correct clock drift between separate input and output
  devices.
- While the assistant speaks, the canceller can turn the user's voice down
  along with the echo. Devices, volume, room acoustics and load affect how
  well cancellation works.

## Benchmark

To measure the canceller's processing cost on your machine from a source
checkout:

```bash
uv sync --python 3.11
PYTHONPATH=src .venv/bin/python scripts/benchmark_echo_cancellation.py --seconds 10 --repeats 3 --paced
```

The script prints CPU time and echo reduction as JSON lines, with and without
cancellation. It excludes audio drivers, process communication, networking
and model inference.
`--paced` feeds audio at real-time speed; leave it out to run as fast as
possible. The script uses random noise as playback, so it measures CPU cost,
not how well the canceller keeps speech.
