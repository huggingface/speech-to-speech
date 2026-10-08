# Echo cancellation

`--echo-cancellation` makes the packaged `talk` and `local` clients remove
their own speaker playback from the microphone. See the
[README](../README.md) for setup. It stays off by default.

## How it works

The `aec` extra installs `pywebrtc-audio`, which wraps WebRTC's echo
canceller (AEC3). Noise suppression and gain control stay off. The binding
applies its high-pass filter whenever echo cancellation is on.

The speaker callback queues each block it plays. The microphone callback
pairs that audio with its own in 10 ms frames and calls the canceller. The
queue holds at most 500 ms of playback; when it runs short, the canceller
treats the missing playback as silence.

## Limits

- `--send-rate` and `--recv-rate` must match, at 8, 16, 32 or 48 kHz. Both
  default to 16 kHz.
- The client does not correct clock drift between separate input and output
  devices.
- While the assistant speaks, the canceller can turn the user's voice down
  along with the echo. In three trials on a MacBook Air, recorded speech
  played over a reply was transcribed twice; the third transcript came back
  empty.

## CPU cost

On a MacBook Air M2 with its built-in mic and speakers, the idle `talk`
client used 4.6% of one core without echo cancellation and 7.4% with it,
averaged over three twelve-second windows. The server ran on a separate
machine; its idle CPU did not change.

While the assistant speaks, the two modes do different work, because without
cancellation the client hears its own echo and cuts replies short. Compare
idle numbers to judge the added cost.

## Benchmark

To measure the canceller's processing cost on your machine from a source
checkout:

```bash
uv sync --python 3.11 --extra aec
PYTHONPATH=src .venv/bin/python scripts/benchmark_echo_cancellation.py --seconds 10 --repeats 3 --paced
```

The script prints CPU time and echo reduction as JSON lines, with and without
cancellation. It excludes audio drivers, networking and model inference.
`--paced` feeds audio at real-time speed; leave it out to run as fast as
possible. The script uses random noise as playback, so it measures CPU cost,
not how well the canceller keeps speech.
