# Echo cancellation: binding choice and CPU cost

The client uses LiveKit's WebRTC audio processing module. Its separate render
and capture calls fit the two sounddevice callbacks and accept different input
and output rates. The original choice followed
[LiveKit's console client](https://github.com/livekit/agents/blob/main/livekit-agents/livekit/agents/cli/_legacy.py)
and passed a live MacBook speaker and voice-interruption check. We had not
compared `pywebrtc-audio` before review.

## Smaller binding comparison

We tested `livekit` 1.1.20 against `pywebrtc-audio` 0.2.0. Both have macOS,
Linux and Windows wheels. For CPython 3.11 on Apple Silicon, the compressed
[pywebrtc-audio wheel](https://pypi.org/project/pywebrtc-audio/0.2.0/) is 358 KiB;
the [LiveKit wheel](https://pypi.org/project/livekit/1.1.20/) is 8.9 MiB.
These are download sizes, excluding dependencies.

`pywebrtc-audio` is a viable smaller alternative. On the same saved 16 kHz
MacBook speaker recording, both removed all six false VAD barge-ins at a
20 ms delay hint. After the first two seconds of playback, LiveKit removed
43.2–43.7 dB of echo; pywebrtc removed 38.7–39.4 dB across delay hints of
0, 20, 60, 100 and 150 ms. Noise suppression and gain control were off.

The synthetic delayed-noise check gives a different result: LiveKit removes
about 47 dB, while pywebrtc removes 11–15 dB with the comparison adapter.
Running the exact three-second signal from the existing audio-client test gives
47.2 dB for LiveKit and 17.3 dB for pywebrtc, below its 30 dB requirement.
The adapter carries incomplete 10 ms frames between 1024-sample callbacks;
it does not pass partially filled frames to either binding. This result
applies to that signal and adapter, not every real speech stream.

Keep LiveKit for this PR. It already passes the live hardware check and the
existing synthetic regression. The
[pywebrtc API](https://github.com/strands-labs/pywebrtc-audio/blob/main/src/pywebrtc_audio/_webrtc_audio.pyi)
pairs equal-length microphone and playback arrays at one sample rate.
Switching would need playback buffering, resampling for different device
rates, and checks for callbacks that arrive independently. Our comparison
serializes playback and capture at the same rate; it does not validate those
cases. A smaller binding remains worth a separate client comparison.

## CPU measurements

CPU percent below means percent of **one core**, computed as process CPU time
per second of audio. Imports, processor construction and signal generation
happen before timing. The workload uses mono 16 kHz audio, 1024-sample
callbacks, silence for idle, and a delayed quieter copy of synthetic playback
for echo. Both bindings receive a 20 ms delay hint.

MacBook Air M2, three paced runs of 9.984 seconds per case:

| Binding | Idle CPU | Echo CPU |
| --- | --- | --- |
| off | 0.13% (0.10–0.19%) | 0.11% (0.11–0.12%) |
| livekit | 2.82% (2.55–3.02%) | 2.78% (2.59–2.95%) |
| pywebrtc | 2.02% (1.85–2.18%) | 2.41% (2.32–2.46%) |

Each cell shows the mean and the range across three runs.

The Linux workstation ran three 29.952-second audio workloads per case without
pacing: LiveKit used 0.27–0.37% of one core, and pywebrtc used 0.34–0.49%.
Those numbers express processing cost per audio second, not the CPU percent
of the accelerated process while it runs.

This isolates audio processing and the callback buffers. It excludes the
sounddevice driver, WebSocket client, and server models. It does not measure
idle CPU for the whole pipeline or justify enabling cancellation by default
on all hardware. Echo cancellation remains opt-in.

See the [README commands](../README.md) to rerun the benchmark. The optional
pywebrtc comparison uses a same-rate paired adapter; the packaged client
continues to use LiveKit.
