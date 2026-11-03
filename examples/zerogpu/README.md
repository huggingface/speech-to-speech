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
STT, Qwen3.5-4B, and a selected native TTS backend all executing on ZeroGPU. The native
pipeline retains VAD, Smart Turn, history, sentence streaming, and interruptions.
The browser waits at **Preparing models** before sending microphone audio.

The library registers `zerogpu` in each of the STT, LLM, and TTS backend registries.
The handlers delegate inference to one shared worker; they do not implement a
second conversation pipeline. The TTS proxy inherits only BaseHandler and has
no imports or inheritance from a concrete TTS model. Local/API defaults are unchanged. This initial
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

## Model selection

Execution and TTS model selection are separate. The default stays Qwen3
CustomVoice; the same parent handler can use Kokoro:

```python
import spaces  # before CUDA/model libraries
from speech_to_speech.zerogpu import runtime

runtime.initialize(tts_backend="kokoro")
# Or: runtime.initialize(tts_backend="qwen3", tts_voice="Aiden")
```

For a package installation, Kokoro also requires the optional dependency
`speech-to-speech[zerogpu,kokoro]` and `espeak-ng`; the example includes both.

Set the Space variable `TTS_BACKEND=kokoro` (or `qwen3`) and optionally
`TTS_MODEL` and `TTS_VOICE`. Run staging with the same `TTS_BACKEND` value to
label the example's model credits correctly. Defaults are:

| Backend | Model | Voice |
|---|---|---|
| `qwen3` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `Aiden` |
| `kokoro` | `hexgrad/Kokoro-82M` | `bm_fable` |

The provider owns model loading, allocation-time preparation, supported
language/voice metadata, and native PCM16 synthesis. Qwen reuses the native
CustomVoice synthesis path and voice/language helpers. Kokoro reuses its native
handler process and shares one CUDA model across language pipelines. The parent
owns session voice state, speculative gates, cancellation, timing, and failure
reporting. No conversation history or live session objects cross into the worker.

A new TTS backend needs a loading adapter implementing this small provider
contract and an entry in the adapter catalog. It does not need a new parent
handler or changes to the lease broker. Existing backend names identify the
models; unsupported combinations fail explicitly. This is not automatic support
for every registered TTS backend: some initialize CUDA outside PyTorch or require
additional model-specific setup. GGML remains unverified.

STT is still Parakeet (`STT_MODEL`, default `nvidia/parakeet-tdt-0.6b-v3`), and
LLM execution uses Transformers causal models (`LLM_MODEL`, default
`Qwen/Qwen3.5-4B`). A different STT architecture is not supported merely by
changing its model ID. This iteration establishes TTS backend flexibility;
it does not claim a universal executor for all model types.

## Lifecycle and limits

The runtime loads weights before ZeroGPU startup without performing inference.
Inside a GPU allocation, it performs backend-specific preparation (including
exclusive CUDA graph capture for Qwen), then warms
STT, LLM, and TTS before announcing readiness. GPU assertions reject CPU fallback.
A bounded worker shares the allocation between model stages and nearby turns.
Token and audio output stream through private filesystem IPC in the Space;
model execution has separate lanes so LLM generation and TTS can overlap.

This is intentionally experimental, with the following limits:

- English conversation, one admitted demo session, text-only LLM, and no
  background history compaction. Qwen supports CustomVoice; Kokoro supports
  native voice IDs and aliases. Voice cloning is not exposed by this proxy.
- GPU leases have a 90-second limit and release after 15 seconds idle. Later
  allocations may introduce another cold delay. Readiness UI currently covers
  initial connection only; it does not hide or eliminate allocation costs.
- The parent stops forwarding immediately on cancellation, including while
  waiting for first audio. Native GPU work and producer cleanup stop at the
  next yield boundary; allocation waits can delay that cleanup.
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

Live validation sends paced 16 kHz PCM and relies on native VAD and Smart Turn,
without manual commits or response creation. Results and deployment revisions
are recorded in the PR description. Preparation time is separate from
speech-end-to-first-audio latency; browser playback is excluded.
