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
  "updates_enabled": true,
  "routes": {
    "stt": {"model": "qwen-asr", "provider": "hf", "protocol": "transcriptions"},
    "llm": {"model": "gemma", "provider": "hf", "protocol": "chat_completions"},
    "tts": {"model": "qwen-tts", "provider": "hf", "protocol": "speech", "voice": "aiden", "model_family": "qwen3-tts"}
  }
}
```

The configuration requires remote OpenAI-compatible STT and TTS handlers and a
consistent Chat Completions or Responses LLM adapter across the CPU pool.
In this mode, building CPU pipeline units initializes the remote clients without
running inference against their bootstrap models. Gateway pools own readiness
and model warmup; an unavailable bootstrap model must not prevent the session service
from starting. Without this opt-in, the adapters retain their startup warmups.
Malformed, oversized, or mismatched handoffs fail before claiming a
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

## Changing models through session.update

A trusted proxy may supply `"updates_enabled": true` in the handshake. All
three stages remain selected throughout the session. The deployment proxy
supports the following public extension:

```json
{
  "type": "session.update",
  "event_id": "select-models",
  "session": {
    "type": "realtime",
    "models": {
      "stt": {"model": "asr-alternative", "provider": "hf"},
      "llm": "gemma-alternative",
      "tts": {"model": "speech-alternative", "provider": "hf"}
    }
  }
}
```

Omitted stages keep their selections. A `null` stage is rejected.
`session.model` remains an LLM shorthand.
`session.created` and `session.updated` include the effective `models` map and
the effective LLM `model`. The selected routes replace the bootstrap model
defaults for subsequent STT, LLM and TTS requests.

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
        "stt": {"model": "asr-alternative", "provider": "hf", "protocol": "transcriptions"},
        "tts": {"model": "speech-alternative", "provider": "hf", "protocol": "speech", "voice": "alloy"},
        "llm": {
          "model": "gemma-alternative",
          "provider": "hf",
          "protocol": "chat_completions",
          "capabilities": {
            "context_window": 32768,
            "tools": true,
            "images": false,
            "audio_input": false,
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

For Qwen3-TTS aliases, the proxy must include `"model_family": "qwen3-tts"`
in each initial or replacement TTS route. This preserves the adapter's existing
language-name conversion and unsupported-language Auto handling even when the
alias does not contain `qwen3-tts`. Canonical model names containing that marker
continue to work without the field. The family is private adapter metadata;
requests and public acknowledgements still use the selected model alias.

The corresponding `session.updated` or `error` carries the private
`_session_routing` update ID. The proxy settles the reservation and strips that
field before forwarding the event. It holds the old/new pool union until this
acknowledgement. The acknowledgement can arrive after other events, so the proxy
must keep forwarding client events while settlement is pending. A later routed
update supersedes a pending one, which settles with an error. On an
uncertain/lost handoff, the proxy must close and release the allocation.

A switch takes effect at an idle turn boundary. When one arrives during a turn
(an active or pending response, unresolved tool call, speech in progress,
cancellation cleanup, progressive STT, hidden prefetch or compaction), it waits
for that work to finish while other client events are processed normally, then
applies and sends `session.updated`. The server checks idleness by passing a
barrier through the handler chain. A switch that cannot apply within 60 seconds
settles with an error. A pending switch applies its other session fields when
it takes effect, after any plain updates sent while it was pending. A rejected
combined update changes no session fields. A successful update replaces the
configuration as a whole while retaining the session ID, conversation ID and
Chat. Old provider compaction callbacks are detached; each future request
receives the selected model/provider plus the original allocator affinity ID.

The initial compatibility boundary is deliberately limited: the same LLM adapter
protocol, full-context continuation, and tool capability when the session
configures tools. The declared context window is informational: if retained
history exceeds the new model's window, that provider reports it on the next
turn, as it would for a fixed model. Retained history never
blocks a switch. On an LLM change, provider-local reasoning items are dropped,
and retained images or audio are dropped when the destination does not declare
them (audio becomes the usual history placeholder). Messages and tool calls and
results carry over. A TTS change keeps the current voice when the new route
supports it and otherwise uses the route's default voice; an explicitly
requested unsupported voice fails the update. No tokenizer conversion,
backend-local continuation migration, in-flight generation migration or arbitrary
local model loading is implemented.

This extends [#547](https://github.com/huggingface/speech-to-speech/issues/547).
OpenAI's hosted Realtime API does not allow model changes through `session.update`
([official reference](https://developers.openai.com/api/reference/resources/realtime/client-events#session.update)).
The hosted API also permits a voice change only before the session has produced
audio. This opt-in routing extension can change the voice after earlier audio
when the new TTS route does not support the current one, so clients must not assume hosted Realtime
voice-change behavior on these connections.
The optional `models` field may require a client's raw-event/extra-fields support;
the LLM-only `model` spelling uses the existing event shape. Clients that do not
enable/request the extension keep existing supported WebSocket/WebRTC flows.
Managed routing currently covers WebSocket only; ordinary WebRTC behavior is
unchanged.
