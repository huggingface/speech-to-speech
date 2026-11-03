# Native ZeroGPU handler validation

Tested 2026-11-03 on private Space `andito/speech-to-speech-zerogpu-quality`,
revision `b22337d2797a1facfe4f89966509e30a74aaec3d`.

Promoted to `andito/speech-to-speech-zerogpu-realtime`, revision
`2816dff2da09a201983cd5de92ca03d97fa00363`. The main Space then passed an
automatic spoken-turn smoke test: 8.00s preparation, 2.42s server E2E,
.57s STT, .88s LLM, and .27s TTS first audio.

The launcher selected the library's registered `zerogpu` STT, LLM, and TTS
handlers. No experiment-local model adapters were imported. GPU diagnostics
reported NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb, with STT,
LLM, TTS, and the TTS codec all on `cuda:0`.

Models: nvidia/parakeet-tdt-0.6b-v3; Qwen/Qwen3.5-4B (BF16, thinking off);
Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice (Torch backend, Aiden).

## Automatic turn probe

Preparation before sending speech took **9.06 seconds**. The same 1.64-second
16 kHz mono PCM16 utterance was sent three times at microphone speed. The probe
sent neither manual audio commits nor response creation requests.

| Turn | Server E2E | STT | LLM | TTS first audio | Smart Turn hold |
|---|---:|---:|---:|---:|---:|
| 1 | 2.51s | .67s | .91s | .33s | 0s |
| 2 | 2.34s | .32s | .42s | .20s | 1.25s |
| 3 | 1.15s | .33s | .45s | .21s | .02s |

All responses completed and streamed audio. Stages overlap and do not sum to
E2E. Preparation is separate; it has not been eliminated. Browser playback is
excluded. These are smoke measurements, not a statistical latency benchmark.

## Interruption and history

A request introducing the user as Alex and asking for a long story was followed
by "Stop. What is my name?" while the first response was active. The first
response was cancelled, and the next completed with "Your name is Alex."
Its server E2E was 1.22 seconds. No manual cancellation message was used.

## Allocation renewal

A same-connection probe with a 20-second pause between turns also completed
both replies. After idle allocation expiry, the second turn took **5.96 seconds**
E2E (STT stage including allocation wait: 4.57 seconds). Renewal works, but its
cold delay is still visible; this is a remaining usability issue.

## Local regression checks

- 159 Python tests passed across `test_zerogpu*.py`, backend registry, Realtime
  pipeline builder, and demo server tests.
- All 67 `demo/tests/*.test.mjs` tests passed, including readiness opt-in,
  ordinary connections without readiness events, and existing SDK transports.
- Ruff checks passed for the new modules, registry, tests, and example.

No human microphone/browser listening test was performed. Remaining deployment
and lifecycle limitations are documented in README.md; this is an experimental
backend proposal, not a claim of production readiness.

## PR branch verification against current main

The isolated PR branch was based on main at `2e2676cd`. After applying only the
ZeroGPU changes, 206 targeted Python tests and all 70 demo JavaScript tests
passed. Repository-wide Ruff checks/format checks and Mypy also passed. The live
Space measurements above refer to the deployment revisions recorded above,
rather than claiming a new live measurement for the updated main baseline.
