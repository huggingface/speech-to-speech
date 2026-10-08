# Echo cancellation: binding choice and CPU cost

The client uses `pywebrtc-audio` 0.2.x through the optional `aec` extra.
Both input and output default to mono 16 kHz; cancellation requires matching
`--send-rate` and `--recv-rate`. Without cancellation, those rates can differ.
Noise suppression and gain control remain off. The binding automatically
applies a high-pass filter with echo cancellation, as described in its
[API documentation](https://pypi.org/project/pywebrtc-audio/0.2.0/).

## Why pywebrtc-audio

The original PR used LiveKit following its console client. We compared the
smaller binding after review, tested a paired adapter through the actual
microphone/speaker callbacks, and switched after the user's live speech test
worked well. For CPython 3.11 on Apple Silicon, the compressed pywebrtc wheel
is 358 KiB versus 8.9 MiB for LiveKit, excluding dependencies.

The speaker callback queues reference samples; the microphone callback pairs
them with capture frames. Incomplete 10 ms frames carry over between callbacks.
A lock protects the shared queue, only the mic thread calls the native
processor, and the queue caps references at 500 ms if capture stalls. Missing
references become silence. Clock drift and mixed-rate resampling remain outside
this adapter; unequal rates fail with a clear error when cancellation is on.

The original delayed-noise regression removed about 47 dB with LiveKit and
17 dB with pywebrtc. Its 30 dB cutoff over-weighted that signal relative to
speech behavior. We replaced the noise input with a speech fixture derived
from the existing package reference recording. The speech regression retains
the 30 dB requirement (about 53 dB measured) and also checks that an overlapping
near voice survives with useful amplitude and correlation, allowing for the
binding's processing delay. This avoids selecting a binding solely by how
much it suppresses noise or accepting a processor that silences all input.

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


These real-client measurements preceded the production switch: the pywebrtc
rows used the same paired processing and buffering now in the client, with
extra telemetry for the measurement. LiveKit is a historical comparison and
is no longer a project dependency. The overlap transcript misses remain in
the table; the later human test is a separate qualitative check.

## Rerunning the processing benchmark

See the [README commands](../README.md). The script compares cancellation off
and the production pywebrtc binding. It measures the canceller and buffers,
excluding audio drivers, networking and server models. CPU percent is process
CPU time per audio second; `--paced` processes at real-time speed. The synthetic
noise signal in this CPU workload is not the speech-quality acceptance test.
