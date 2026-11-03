---
title: Speech to Speech on ZeroGPU
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
python_version: '3.12'
---

# Native ZeroGPU handlers

Experimental deployment of the normal speech-to-speech voice demo with Parakeet
STT, Qwen3.5-4B, and Qwen3-TTS CustomVoice all executing on ZeroGPU. The native
pipeline retains VAD, Smart Turn, history, sentence streaming, and interruptions.
The browser waits at **Preparing models** before sending microphone audio.

The library registers `zerogpu` in each of the STT, LLM, and TTS backend registries.
The handlers delegate inference to one shared worker; they do not implement a
second conversation pipeline. Local/API defaults are unchanged. This initial
integration loads all three models together, even if only one handler is selected.

## Deploy

1. Create a Gradio Space with ZeroGPU hardware.
2. Install the normal demo's frontend dependencies (`npm ci` in `demo`).
3. From the repository root, run:

   ```sh
   python examples/zerogpu/stage.py /tmp/speech-to-speech-zerogpu
   ```

4. Upload that explicit snapshot to the Space. Never upload your entire working
   directory: the staging script includes only application files, library source,
   and demo assets. The Space installs the pinned `requirements.txt`.
5. Open the Space and connect once. After preparation, turn-taking is automatic.

`app.py` imports `spaces` before model libraries, explicitly initializes the
shared runtime, and selects the three registered handlers. `host.py` uses the
normal Realtime pipeline builder and WebSocket router, serves the normal demo,
and advertises `modelReadiness: true`. The optional library extra is
`speech-to-speech[zerogpu]`; use the example's pinned requirements for the tested
Space environment. Selecting the backend in an ordinary CLI process is not a
replacement for this Space startup sequence.

The launcher accepts `STT_MODEL`, `LLM_MODEL`, `TTS_MODEL`, and `TTS_SPEAKER`.
Defaults are `nvidia/parakeet-tdt-0.6b-v3`, `Qwen/Qwen3.5-4B`,
`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, and `Aiden`. Only these defaults have been
validated end to end. The implementations support their corresponding model
families, not arbitrary architectures. TTS uses the library's Torch backend;
GGML on ZeroGPU has not been established by this integration.

## Lifecycle and limits

The runtime loads weights before ZeroGPU startup without performing inference.
Inside a GPU allocation, it captures TTS CUDA graphs exclusively, then warms
STT, LLM, and TTS before announcing readiness. GPU assertions reject CPU fallback.
A bounded worker shares the allocation between model stages and nearby turns.
Token and audio output stream through private filesystem IPC in the Space;
model execution has separate lanes so LLM generation and TTS can overlap.

This is intentionally experimental, with the following limits:

- English conversation, one admitted demo session, text-only LLM, CustomVoice
  TTS, and no background history compaction.
- GPU leases have a 90-second limit and release after 15 seconds idle. Later
  allocations may introduce another cold delay. Readiness UI currently covers
  initial connection only; it does not hide or eliminate allocation costs.
- TTS cancellation takes effect when its next audio chunk is received. Waiting
  for a fresh allocation can therefore delay cancellation cleanup.
- Startup uses the private `spaces.zero.startup()` hook because this example
  mounts Gradio alongside the normal FastAPI/WebSocket demo. Spaces is pinned;
  upgrading requires a new deployment check.
- Worker command files are retained to avoid replay/late-worker races; long-lived
  deployments need a safe retention policy before this is production-ready.

## Verification

Unit tests in `tests/test_zerogpu*.py` cover initialization, lease ownership,
errors, cancellation, native history/voice semantics, and readiness gating.
`demo/tests/model-readiness.test.mjs` also checks that ordinary demos do not wait
for custom readiness events.

The probes send paced microphone-format PCM and rely on native VAD; they never
send manual commits or `response.create`:

```sh
python examples/zerogpu/benchmark.py wss://YOUR-SPACE.hf.space/v1/realtime \
  /path/to/mono-16khz-pcm16.wav --wait-models --turns 3
python examples/zerogpu/probe.py wss://YOUR-SPACE.hf.space/v1/realtime \
  /path/to/story-request.wav /path/to/interruption.wav --wait-models --interrupt-on response
```

Private Spaces use the locally configured Hugging Face token. Preparation time
is reported separately from speech-end-to-first-audio latency. Server timings
exclude browser playback; audio quality still requires listening in the browser.
