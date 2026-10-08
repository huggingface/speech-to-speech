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

LiveKit remains the binding in this PR and passes the existing synthetic
regression. We also tested the experimental pywebrtc adapter on the real
microphone and speaker path described below. The
[pywebrtc API](https://github.com/strands-labs/pywebrtc-audio/blob/main/src/pywebrtc_audio/_webrtc_audio.pyi)
pairs equal-length microphone and playback arrays at one sample rate.
Our experimental live adapter queues playback for the microphone callback,
uses a lock for the shared queue, caps it at 500 ms, and fills missing
reference samples with silence. It processes whole 10 ms frames and calls
the native processor only from the microphone thread. It needs equal rates
and does not correct device clock drift. Switching the production binding
still needs mixed-rate handling and broader live hardware checks.

## Actual client and server measurements

We ran the real packaged talk client on a MacBook Air M2, using its built-in
microphone and speakers at 50% volume over Tailscale to an existing Linux GPU
server. The server used Parakeet TDT, Qwen3-4B-Instruct-2507, and Qwen3-TTS GGML
on GPU 0. Models stayed loaded; no other client shared the server pipeline.

Each binding ran three trials with rotated order. Each trial measured twelve
seconds of idle listening, assistant playback, and recorded speech during
playback. A fixed typed bicycle prompt started each active phase. The speech
phase played a fixed weather follow-up through a separate audio player, so
it was absent from the client's echo reference. This tests a repeatable
acoustic speech stimulus rather than a person speaking live.

CPU below is process CPU time divided by wall time, as a percent of **one
core**. Client and server run on different machines. Initialization and waiting
for the first real speaker block were outside the measured windows. The client
used normal display settings, plus the same light mic/output telemetry in all
modes. Raw microphone peaks and actual speaker frames confirmed that the devices
worked; an earlier SSH launch received all-zero mic input and was rejected.

| Binding | Phase | Client CPU, mean (range) | Server CPU, mean (range) | Speech starts | Played seconds |
| --- | --- | --- | --- | --- | --- |
| off | idle | 4.64% (4.45–4.80) | 3.82% (3.57–3.98) | 0, 0, 0 | 0.00, 0.00, 0.00 |
| off | playback | 5.64% (5.18–6.36) | 26.36% (12.81–34.76) | 1, 2, 2 | 0.64, 0.96, 1.09 |
| off | speech during playback | 5.54% (5.27–5.89) | 43.04% (39.05–46.86) | 4, 4, 3 | 2.88, 2.30, 1.54 |
| livekit | idle | 7.57% (7.32–7.82) | 3.94% (3.83–4.07) | 0, 0, 0 | 0.00, 0.00, 0.00 |
| livekit | playback | 12.05% (11.67–12.26) | 131.29% (130.43–132.25) | 0, 0, 0 | 11.65, 11.65, 11.78 |
| livekit | speech during playback | 10.81% (9.34–12.58) | 95.80% (49.47–123.14) | 1, 1, 1 | 7.55, 8.13, 3.20 |
| pywebrtc | idle | 7.41% (6.92–7.84) | 3.87% (3.82–3.90) | 0, 0, 0 | 0.00, 0.00, 0.00 |
| pywebrtc | playback | 11.58% (10.88–12.39) | 131.98% (131.27–132.34) | 0, 0, 0 | 11.71, 11.65, 11.71 |
| pywebrtc | speech during playback | 8.62% (7.83–10.01) | 71.73% (44.12–126.15) | 1, 1, 1 | 8.32, 2.75, 8.00 |

Each row reports three runs. Idle client CPU rose from 4.64% to 7.57% with
LiveKit: **2.93 percentage points of one core**. Server idle means were
3.82% off and 3.94% on, within the observed ranges. The pywebrtc adapter used
7.41% idle client CPU; its range overlaps LiveKit's, so this small sample does
not establish a meaningful CPU advantage for either binding.

During playback, off produced 1, 2 and 2 false speech starts; both cancellers
produced zero in all three runs. During the recorded interruption, both
produced exactly one speech start in each run. Each binding transcribed the
weather follow-up in two of three trials; the other trial ended with an empty
transcription. That does not establish reliable recognition in every overlap.
Off produced 4, 4 and 3 speech starts, mixing the stimulus with echo.
The speaker callbacks confirmed actual playback at each interruption start.

These measurements include the real client callbacks, WebSocket handling and
server pipeline. They exclude shared macOS audio-driver CPU and the separate
player used as the speech stimulus. Active windows do not have equal inference
or playback work: the off client hears echo and cancels replies. Compare idle
windows to estimate the added steady cost; do not subtract active CPU as if
only the binding changed. This short check does not measure battery impact.
Echo cancellation remains opt-in pending broader hardware checks.

## Isolated processing measurements

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
