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

Run the normal hands-free demo with GPU STT, LLM, and TTS. The existing pipeline
retains VAD, Smart Turn, history, sentence streaming, and interruptions. The
browser waits at **Preparing models** before sending microphone audio.

## Deploy

1. Create a Gradio Space with ZeroGPU hardware.
2. Install frontend dependencies with `npm ci` in `demo`.
3. From the repository root, run:

   ```sh
   python examples/zerogpu/stage.py /tmp/speech-to-speech-zerogpu
   ```

4. Upload that snapshot to the Space. It includes application files, library
   source, and demo assets; the Space installs the pinned requirements.
5. Open the Space and connect. After preparation, turn-taking is automatic.

`app.py` imports `spaces` before CUDA libraries, initializes the runtime, selects
registered `zerogpu` handlers, and serves the existing Realtime API and demo.
Selecting handlers alone in an ordinary CLI process does not perform this startup
sequence. Local/API defaults are unchanged.

## Models

The same TTS handler supports Qwen CustomVoice and Kokoro through native synthesis
adapters. Set Space variables `TTS_BACKEND`, `TTS_MODEL`, and `TTS_VOICE` as needed:

| Backend | Default model | Default voice |
|---|---|---|
| `qwen3` (default) | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `Aiden` |
| `kokoro` | `hexgrad/Kokoro-82M` | `bm_fable` |

Use the same `TTS_BACKEND` when staging so model credits match. For a package
installation, use `speech-to-speech[zerogpu]`; Kokoro also requires the `kokoro`
extra and `espeak-ng`. The example requirements include both providers.

Programmatic initialization, before constructing handlers:

```python
import spaces  # before CUDA/model libraries
from speech_to_speech.zerogpu import runtime

runtime.initialize(tts_backend="kokoro")
# Or: runtime.initialize(tts_backend="qwen3", tts_voice="Aiden")
```

STT uses Parakeet (`STT_MODEL`, default `nvidia/parakeet-tdt-0.6b-v3`). LLM uses
Transformers causal models (`LLM_MODEL`, default `Qwen/Qwen3.5-4B`). Changing a
model ID does not add support for a different architecture.

## How it works and limits

Handlers keep conversation state in the host. A shared worker loads model weights
before ZeroGPU startup, then prepares CUDA graphs and warms all three models
inside an allocation. Private file IPC streams tokens and audio; separate model
lanes allow LLM and TTS execution to overlap. GPU checks reject CPU fallback.

TTS providers adapt loading, allocation-time preparation, voice/language metadata,
and native PCM16 synthesis. Adding a provider does not require another parent
handler or changes to the lease broker. Other TTS backends, voice cloning, and
GGML are not supported by this integration.

- Experimental: English conversation, one admitted session, text-only LLM, and no
  background history compaction. All three models load together.
- Leases last at most 90 seconds and release after 15 seconds idle. Later
  allocations can incur another cold delay; readiness UI covers first connection.
- Cancellation stops host forwarding immediately. Worker cleanup waits for a
  synthesis yield or allocation completion.
- Mounted Gradio startup uses the private `spaces.zero.startup()` hook. Keep the
  pinned Spaces version; upgrades require deployment testing.
- Command files are retained to avoid replay/late-worker races. Long-lived
  deployments need a safe retention policy.

Lifecycle, provider, cancellation, and readiness tests are in
`tests/test_zerogpu*.py` and `demo/tests/model-readiness.test.mjs`. Live GPU
validation results and deployment revisions are recorded in the PR description.
