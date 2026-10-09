# YAML configuration

Use a YAML file to define reusable VAD, STT, LLM, and TTS blocks, connect them
into named pipelines, and start selected pipelines from the CLI or Python.

## Start with a Mac configuration

The [Mac example](../example_configs/mac.yaml) runs speech recognition, the LLM,
and speech synthesis locally on Apple Silicon, with no API key. Use the file
from a repository checkout, or save the following as `example_configs/mac.yaml`:

```yaml
schema_version: 1

blocks:
  vad: {kind: vad, backend: silero}
  stt: {kind: stt, backend: parakeet-tdt}
  llm:
    kind: llm
    backend: mlx-lm
    settings:
      model_name: mlx-community/Qwen3-4B-Instruct-2507-4bit
  tts: {kind: tts, backend: qwen3}

pipelines:
  mac:
    options:
      mac_optimal_settings: true
    stages: {vad: vad, stt: stt, llm: llm, tts: tts}
```

YAML selects each backend explicitly. `mac_optimal_settings` supplies preset
defaults for those backends; explicit settings override them.

## Build your configuration

1. Define a block for each stage. `kind` is `vad`, `stt`, `llm`, or `tts`;
   `backend` selects an existing supported backend. Put its arguments in
   `settings`, using field names such as `openai_stt_base_url` or `model_name`.
   See the [STT](../src/speech_to_speech/STT/README.md),
   [LLM](../src/speech_to_speech/LLM/README.md), and
   [TTS](../src/speech_to_speech/TTS/README.md) component guides.
2. Connect blocks through a pipeline's `stages`. All four stages are required,
   and each reference must name a block of the matching kind. Several blocks
   can use the same backend with different settings; several pipelines can
   reuse a block with independent settings and runtime state.
3. Set `num_pipelines` to the number of isolated instances of that definition.
   It defaults to `1` and must be a positive integer.

Block IDs and pipeline names are case-sensitive, start with a letter, and
contain up to 64 letters, digits, underscores, or hyphens. Their namespaces
are separate. The file must have `schema_version: 1` and nonempty `blocks`
and `pipelines` mappings.

| Where | What belongs there |
|---|---|
| Block `settings` | Backend argument fields, including existing prefixes and inherited fields; VAD excludes the `vad` selector |
| Pipeline `options` | Pipeline-wide options such as `device`, `mac_optimal_settings`, diarization, and `enable_visemes` / `stv_model_name` / `stv_device` |
| `runtime` | Process logging: `log_level` and `log_transcripts` |
| `runtime.server` | Listener `host` and `port` |
| `runtime.client` | Packaged client settings such as `url`, `voice`, `input_device`, `output_device`, `playback_buffer_ms`, and `tool_module` |

Settings use the existing [argument definitions](../src/speech_to_speech/arguments_classes/)
and defaults. Unknown fields fail. Backend selectors belong in `backend`, not `settings`.

To bypass STT, use `backend: none` with empty settings. The LLM backend and
remote model must support audio input; diarization requires an STT backend.
The existing LLM proxy capability checks also apply.

## Run a pipeline

```bash
speech-to-speech local -f example_configs/mac.yaml
speech-to-speech serve -f example_configs/mac.yaml --name mac
speech-to-speech talk -f example_configs/mac.yaml
```

`-f` and `--file` are equivalent. `serve` activates all definitions when names
are omitted; repeat `--name` to select several distinct definitions. Each
definition's `num_pipelines` controls its instance count. `local` requires one
selected definition; specify its name when the file contains several pipelines.

`local` starts the server and one microphone/speaker client over loopback. Its
client URL follows the server port; an explicit non-loopback host or conflicting
client URL fails.

`talk` reads `runtime.client` and process logging settings. It does not start
pipeline handlers or require server/block environment values. An optional single
`--name` selects a named route declared in the file. It changes the URL, not the
client's model. With this example, omitted names connect to the local server's
unqualified route.
File mode accepts file selection, pipeline names, and help; put other
settings in YAML rather than mixing in legacy flags. Duplicate file options or
names and unknown names fail. Existing CLI flags and flat JSON input still work
outside file mode.

## Serve several named pipelines

The [concurrent example](../example_configs/multi.yaml) defines `primary` with
two instances and `secondary` with one. It uses OpenAI-compatible STT and TTS
servers plus separate LLM connection settings. Start those providers first and
set the example's LLM URL, model, and key environment values. Shared blocks
produce independent handler instances and session state. Reusing a block does
not enable model-weight sharing.

```bash
speech-to-speech serve -f example_configs/multi.yaml
speech-to-speech serve -f example_configs/multi.yaml --name primary --name secondary
speech-to-speech talk -f example_configs/multi.yaml --name primary
```

All selected definitions share `runtime.server.host` and `runtime.server.port`.
Names are case-sensitive selectors, not authentication credentials.
Existing access-control behavior still applies.

| Operation | Named endpoint |
|---|---|
| WebSocket | `/v1/pipelines/{name}/realtime` |
| WebRTC offer | `/v1/pipelines/{name}/realtime/calls` |
| WebRTC hangup | `/v1/pipelines/{name}/realtime/calls/{call_id}` |
| Responses proxy | `/v1/pipelines/{name}/responses` |
| Chat-completions proxy | `/v1/pipelines/{name}/chat/completions` |

The packaged client also accepts a named endpoint through legacy `talk --url`.
File-mode `talk --name` preserves the URL's scheme, authority, and base-path prefix.
An already named URL must agree with the selected name.

Existing unqualified creation and proxy endpoints remain aliases when exactly
one definition is active. With several, they reject ambiguity rather than select
by declaration order or model name. Unqualified WebRTC hangup remains valid by
call ID; named hangup must match the call's original definition. Named offers
return named `Location` headers.

A session keeps its definition and instance until release. Exhausting one
definition never uses another's capacity. Admission is immediate, with no queue.

| Condition | Error type | HTTP status |
|---|---|---|
| Unknown or inactive definition | `unknown_pipeline` | 404 |
| Ambiguous unqualified request | `pipeline_selection_required` | 400 |
| No healthy started instance | `pipeline_unavailable` | 503 |
| Instances occupied, draining, or quarantined | `session_limit_reached` | 503 |

WebSocket admission errors use the existing error event and close code 1008.
Proxy calls use the selected definition's LLM settings without consuming speech
capacity. Disabled proxies and unsupported operations retain their 501 response.

`/v1/pool` retains aggregate `size`, `in_use`, and unit occupancy states. Configured
mode adds `available`, named `pipelines` summaries, and unit identity and health.
Failed health can overlap active or draining occupancy; availability requires a
healthy ready unit without a reserved session or stop request.
`/v1/usage` retains aggregate counters and adds named summaries. Single-definition
aliases share counters, and named summaries do not add another copy to aggregate usage.

Construction and initial startup are all-or-nothing. After readiness, a failed
worker disables its instance while healthy instances continue serving. Failed
instances cannot recover through disconnect or a late drain marker. Healthy
quarantined instances recover only after `SESSION_END` proves their chain drained.
Caller cancellation or listener failure stops the whole runtime.

## Values, paths, and validation

Use `{env: NAME}` for scalar settings or `num_pipelines`. References resolve
from the environment at resolution time; `.env` files and string interpolation
are not supported. Strings remain text, booleans accept exactly `true` or
`false`, integers accept signed decimals, and floats must be finite decimal
numbers. Collections must be written as YAML. Environment-shaped mappings
inside open provider dictionaries remain literal data.

Quoted scalars remain strings: `"2"` cannot supply an integer field. Explicit
`false`, zero, empty strings, and empty collections keep their meanings.
Collections replace defaults as complete values. `null` is allowed for nullable
fields; VAD `max_speech_ms: null` also means unbounded duration. Missing explicit
environment values fail even when a backend has a credential fallback.

These filesystem fields expand `~` and resolve relative to the YAML file's directory:

- `vad_firered_model_dir`, `smart_turn_model_path`
- `omnivoice_ref_audio`, `omnivoice_voice_clone_prompt`, `omnivoice_ref_voices_dir`
- `qwen3_tts_gguf_talker_path`, `qwen3_tts_gguf_codec_path`, `qwen3_tts_ref_cache_dir`
- `qwen3_tts_ref_audio`, `qwen3_tts_ref_spk`, `qwen3_tts_ref_rvq`

Model IDs, URLs, tool module names, and ambiguous path-or-model fields stay
literal; use absolute paths for the latter. Resolution does not check asset
existence or provider access.

Use one local UTF-8 `.yaml` or `.yml` mapping document, up to 1 MiB and 32 container
levels. Duplicate keys, nonstring keys, anchors, aliases, merge keys, and explicit
tags fail. YAML booleans and null use lowercase `true`, `false`, and `null`;
words such as `on` and `yes` remain strings. Errors identify configuration
locations when available and omit input values and source snippets.

All definitions must have valid structure and literal values, including inactive
ones. Only selected pipelines and process/server settings need environment
values; client settings resolve only when requested. Runtime construction applies
existing presets, device overrides, and credential fallbacks to independent copies.

## Use from Python

```python
from threading import Event

from speech_to_speech.config import load_config, validate_config, resolve_config
from speech_to_speech.configured_runtime import build_configured_server

document = load_config("example_configs/mac.yaml")
validate_config(document)
resolved = resolve_config(document, names=["mac"])
runtime = build_configured_server(resolved, Event())
try:
    runtime.start()
    if not runtime.wait_ready(timeout=30):
        raise RuntimeError("Configured runtime did not become ready before the timeout")
    runtime.wait()
finally:
    runtime.stop()
```

| API | Behavior |
|---|---|
| `load_config(path)` | Parse a file into a document with its absolute path and source locations |
| `validate_config(document)` | Check all definitions without reading environment values or starting resources |
| `resolve_config(document, names=None, environ=None, include_client=False, client_only=False)` | Validate and return independent settings; omitted names select all pipelines and omitted `environ` uses a snapshot of `os.environ` |
| `build_configured_runtime(resolved, stop_event)` | Construct named worker pools without a listener |
| `build_configured_server(resolved, stop_event, local=False)` | Construct one listener for selected named definitions; `local=True` requires one definition and adds the loopback client |
| `runtime.wait_ready(timeout=None)` | Wait for actual listener startup and healthy instances in every selected definition; return `False` on timeout |

Use `include_client=True` when resolving client settings for local startup.
For client-only resolution, use `client_only=True` without pipeline names.
Returned settings contain secrets, so do not log or serialize them without redaction.

Loading, validation, and resolution do not construct handlers or start resources.
Construction can initialize models and providers; `start()` launches managed
workers asynchronously. Use `wait_ready()` or the `ready` event to observe
listener readiness. Worker-only runtimes omit the listener requirement.
Readiness clears during shutdown or when a selected definition loses all healthy
instances; other definitions can keep serving. Library lifecycle methods install
no signal handlers. Always call `stop()`; a stopped or failed runtime cannot restart.
Shutdown closes sessions and joins threads within its bounded deadline. A stuck
worker leaves the runtime stopping and reports a failure instead of claiming cleanup completed.
