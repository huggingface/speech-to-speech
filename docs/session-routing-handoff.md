# Session routing from a trusted admission proxy

`session_routing_enabled` is an opt-in extension for deployments whose allocator
selects compatible remote STT, LLM and TTS routes. Enable it only on a
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

The route snapshot travels through the existing per-session RuntimeConfig.
Each inference request uses its selected model plus `X-Speech-Provider` and
`X-Speech-Session-Id`; background LLM compaction retains the same selection.
The gateway consumes these headers. Requests still carry full conversation
context. This supplies an affinity identifier, not a backend cache or migration
guarantee.

`session.created` reports the admitted LLM model and initial TTS voice. Without
the additional `updates_enabled` handshake capability, admitted models remain
fixed and ordinary session settings retain their existing behavior.

## Optional model selection through session.update

A trusted proxy may supply `"updates_enabled": true` in the handshake. In this
mode each route may be `null`, including all three for an initially empty session.
The deployment proxy supports the following public extension:

```json
{
  "type": "session.update",
  "event_id": "select-models",
  "session": {
    "type": "realtime",
    "models": {
      "stt": {"model": "qwen-asr", "provider": "hf"},
      "llm": "gemma-route",
      "tts": null
    }
  }
}
```

Omitted stages keep their selections. A `null` value disables a stage; it does
not unload process-wide weights. `session.model` remains an LLM shorthand.
`session.created` and `session.updated` include the effective `models` map and
the effective LLM `model`. Disabling TTS produces text-only responses; adding it
again enables audio unless the update explicitly requests text. With STT disabled,
text input remains available, and a declared audio-input LLM can receive VAD
audio directly through the existing Chat Completions audio path. The Responses
adapter's audio path uses Chat Completions internally, so it is not advertised
as a native Responses audio route. With the LLM disabled, STT can transcribe but
`response.create` fails clearly. Disabled stages never use bootstrap defaults.

The allocator resolves/authorizes choices and reserves added pool capacity. The
proxy strips client-supplied `_session_routing` fields and attaches this private
envelope to the standard update before forwarding to the private listener:

```json
{
  "_session_routing": {
    "update_id": "unique-update-id",
    "routing": {
      "id": "allocator-session-id",
      "pipeline": "deployment-accounting-key",
      "updates_enabled": true,
      "routes": {
        "stt": null,
        "tts": null,
        "llm": {
          "model": "audio-llm-route",
          "provider": "hf",
          "protocol": "chat_completions",
          "capabilities": {
            "context_window": 32768,
            "tools": true,
            "images": false,
            "audio_input": true,
            "continuation": "full_context"
          }
        }
      }
    }
  }
}
```

The envelope accompanies `type`, `event_id` and `session`; it is not a public
client API. TTS routes also declare a default `voice` and supported `voices`.
The corresponding `session.updated` or `error` carries the private
`_session_routing` update ID. The proxy settles the reservation and strips that
field before forwarding the event. It holds the old/new pool union until this
acknowledgement and blocks subsequent client events while settlement is pending.
On an uncertain/lost handoff, the proxy must close and release the allocation.

Updates require an idle turn boundary. The dispatcher rejects active/pending
responses and unresolved tools, then passes a barrier through the handler chain
to check that cancellation cleanup and earlier audio have finished. Progressive
STT, hidden prefetch and compaction must also finish. A rejected combined update
changes no session fields. A successful update replaces the configuration as a
whole while retaining the session ID, conversation ID and Chat. Old provider
compaction callbacks are detached; each future request receives the selected
model/provider plus the original allocator affinity ID.

The initial compatibility boundary is deliberately limited: the same LLM adapter
protocol, full-context continuation, an equal or larger declared context window,
and capabilities for configured tools and retained images/audio/tool history.
The context-window floor survives LLM removal and resets with the session;
disabling and re-adding the stage cannot bypass it while retaining Chat.
Unsupported voice choices fail during a routed update. No tokenizer conversion,
backend-local continuation migration, in-flight generation migration or arbitrary
local model loading is implemented.

This extends [#547](https://github.com/huggingface/speech-to-speech/issues/547).
OpenAI's hosted Realtime API does not allow model changes through `session.update`
([official reference](https://developers.openai.com/api/reference/resources/realtime/client-events#session.update)).
The optional `models` field may require a client's raw-event/extra-fields support;
the LLM-only `model` spelling uses the existing event shape. Clients that do not
enable/request the extension keep existing supported WebSocket/WebRTC flows.
Managed routing currently covers WebSocket only; ordinary WebRTC behavior is
unchanged.
