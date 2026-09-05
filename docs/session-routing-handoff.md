# Initial routing from a trusted admission proxy

`session_routing_enabled` is an opt-in extension for deployments whose allocator
has already selected compatible STT, LLM and TTS routes. Enable it only on a
private listener behind that proxy. The public proxy must authenticate the
allocation and construct the header from its signed claims; never forward a
client's copy. This server does not authenticate the header itself.

The internal WebSocket handshake supplies `X-Speech-Session-Routing`:

```json
{
  "id": "allocator-session-id",
  "pipeline": "qwen-gemma-qwen",
  "routes": {
    "stt": {"model": "qwen-asr", "provider": "hf", "protocol": "transcriptions"},
    "llm": {"model": "gemma", "provider": "hf", "protocol": "chat_completions"},
    "tts": {"model": "qwen-tts", "provider": "hf", "protocol": "speech", "voice": "aiden"}
  }
}
```

The configuration requires remote OpenAI-compatible STT and TTS handlers and a
consistent Chat Completions or Responses LLM adapter across the CPU pool.
In this mode, building CPU pipeline units initializes the remote clients without
running inference against their bootstrap models. Gateway pools own readiness
and model warmup; a failed optional model must not prevent the session service
from starting. Without this opt-in, the adapters retain their startup warmups.
Malformed, disabled, oversized, or mismatched handoffs fail before claiming a
pipeline. URLs, credentials, catalogs, capacity policies and workers remain
deployment-owned. All routes on a CPU worker must accept that worker's configured
audio format, sample rate and request options. For example, OpenAI speech uses
`stream_format=audio`, not vLLM's `stream` or `language` extensions; configure
compatible options when advertising both. The handoff does not translate dialects.

The immutable route travels through the existing per-session RuntimeConfig.
Each inference request uses its selected model plus `X-Speech-Provider` and
`X-Speech-Session-Id`; background LLM compaction retains the same selection.
The gateway consumes these headers. Requests still carry full conversation
context. This supplies an affinity identifier, not a backend cache or migration
guarantee.

`session.created` reports the admitted LLM model and initial TTS voice. Ordinary
session updates, tools, cancellation and context behavior remain unchanged.
A different `session.update.model` is rejected before applying any other fields;
repeating the admitted model is allowed. Mid-session switching remains separate
work in [#547](https://github.com/huggingface/speech-to-speech/issues/547).
Standard clients require no new WebSocket messages. Unconfigured sessions retain
their existing behavior. This extension currently covers the managed WebSocket
handoff, not WebRTC admission.
