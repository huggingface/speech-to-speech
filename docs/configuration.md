# YAML configuration and startup

The optional configuration API loads, validates, and resolves reusable blocks
and named pipelines from a local YAML file. These offline operations do not
construct handlers, download models, call providers, access audio devices, or
start workers and listeners. Explicit runtime construction and CLI file mode
can start the configured pipeline. Existing Python interfaces, CLI commands,
and flat JSON input keep their existing behavior.

Install the YAML parser:

```bash
pip install "speech-to-speech[config]"
```

Importing `speech_to_speech.config` does not require the parser. Loading YAML
without the optional dependency raises `ImportError` with installation guidance.

## Python usage

```python
from speech_to_speech.config import load_config, validate_config, resolve_config

# Read and parse one local YAML file. Preserve locations for diagnostics.
document = load_config("voice.yaml")

# Check every definition without reading environment variables.
validate_config(document)

# Resolve only primary, with an explicit environment snapshot.
resolved = resolve_config(
    document,
    names=["primary"],
    environ={"PRIMARY_STT_KEY": "replace-with-a-secret"},
)
primary = resolved.pipelines["primary"]
assert primary.num_pipelines == 2
assert primary.stages["stt"].block_id == "stt_primary"
assert primary.stages["stt"].settings["openai_stt_api_key"] == "replace-with-a-secret"
```

`load_config(path)` returns a `ConfigDocument`. Its `data` contains unresolved
values, `path` records the absolute source file path, and `locations` records
source locations. It accepts `str` and `Path` paths.

`validate_config(document)` returns `None`. It checks all blocks and pipelines,
including definitions that a caller does not select. It checks field names,
literal types, environment reference syntax, stage references, and static
backend capabilities. It does not read environment values.

`resolve_config(document, *, names=None, environ=None, include_client=False, client_only=False)`
validates again, then resolves the selected definitions. With `names=None`, it
resolves all pipelines. An explicit sequence preserves selection order. An
empty sequence, a bare string, duplicate names, or unknown names raises
`ConfigurationError`.

The resolver copies the supplied environment mapping, or `os.environ` when
`environ` is omitted, once per call. It resolves process settings and selected
pipelines. It resolves `runtime.client` only with `include_client=True`.
Inactive definitions need valid structure and literal values, but their
explicit environment references do not need values.

With `client_only=True`, resolve client and process logging settings without
resolving server or block environment references. The result has no pipelines
or server settings. Pipeline names are not accepted in this mode. The full
document still needs valid structure and literal settings.

The returned `ResolvedConfig` has `runtime`, `client`, and `pipelines` records
and source metadata. Each pipeline has `stages`, `options`, and `num_pipelines`.
Each stage retains its `block_id`, `kind`, `backend`, validated `settings` mapping, and
normalized backend `selection`. Field source metadata maps configuration paths
to `default`, `file`, or `environment`.

Each resolution has independent mutable settings. Pipelines that reference the
same block also receive independent settings. Resolution does not modify the
source document or registry defaults. Returned settings contain resolved secrets
for library use. Default object representations omit settings and environment
contents; do not serialize or log settings without your own redaction policy.

## Schema

```yaml
schema_version: 1
runtime:
  log_level: info
  log_transcripts: false
  server:
    host: 127.0.0.1
    port: 8765
blocks:
  vad_main:
    kind: vad
    backend: silero
    settings:
      sample_rate: 16000
      max_speech_ms: null
  stt_primary:
    kind: stt
    backend: openai
    settings:
      openai_stt_base_url: http://localhost:8000/v1
      openai_stt_api_key: {env: PRIMARY_STT_KEY}
  stt_secondary:
    kind: stt
    backend: openai
    settings:
      openai_stt_base_url: http://localhost:8001/v1
  llm_main:
    kind: llm
    backend: chat-completions
    settings:
      model_name: served-chat-model
      responses_api_base_url: http://localhost:8080/v1
      responses_api_api_key: ""
  tts_main:
    kind: tts
    backend: openai
    settings:
      openai_tts_base_url: http://localhost:8091/v1
pipelines:
  primary:
    num_pipelines: 2
    stages: {vad: vad_main, stt: stt_primary, llm: llm_main, tts: tts_main}
  secondary:
    stages: {vad: vad_main, stt: stt_secondary, llm: llm_main, tts: tts_main}
```

Require integer `schema_version: 1` and nonempty `blocks` and `pipelines`
mappings. Block IDs and pipeline names match `[A-Za-z][A-Za-z0-9_-]{0,63}`.
Their namespaces are separate. Names are case-sensitive and are not trimmed.
Unused blocks are allowed but must be valid.

A block has `kind`, `backend`, and optional `settings`. The four kinds are
`vad`, `stt`, `llm`, and `tts`. Select a backend supported by the existing
registry, or the VAD argument choices. Multiple blocks can use the same kind
and backend with different settings.

A pipeline has `stages`, optional `options`, and optional `num_pipelines`.
The stage mapping must contain exactly `vad`, `stt`, `llm`, and `tts`, each
referencing a block of that kind. Missing, null, unknown, or wrong-kind stage
references fail. `num_pipelines` defaults to `1` and must resolve to a positive
integer. Booleans are not integers.

For STT bypass, define a block with `kind: stt`, `backend: none`, and empty
settings. Its pipeline must use an LLM backend that supports audio input.
The existing diarization/STT and LLM proxy capability checks also apply.
A backend capability does not establish that a remote model supports audio.

## Setting ownership

Use existing argument field names, including backend prefixes. Do not put
selectors such as `stt`, `tts`, or `llm_backend` inside settings. The block's
backend selects the configuration type. Unknown fields fail, except inside
existing open provider dictionaries.

| Location | Supported settings |
|---|---|
| STT, LLM, TTS block `settings` | The selected registry configuration dataclass, including inherited fields |
| VAD block `settings` | `VADHandlerArguments`, except the `vad` selector |
| `runtime.server` | `RealtimeServerArguments` |
| `runtime.log_level`, `runtime.log_transcripts` | The corresponding `ModuleArguments` fields |
| Pipeline `options` | Remaining `ModuleArguments` fields, except selectors and `num_pipelines` |
| `runtime.client` | Serializable `RealtimeAudioClientConfig` fields and nullable `tool_module` |

Diarization belongs in pipeline options. Process fields do not belong there.
`tool_executor` callables cannot appear in YAML. Tool definitions and module
names remain data; validation does not import a tool module.

The resolver uses existing dataclass types, defaults, choices, inherited
fields, and backend normalization. Explicit collections replace default
collections as complete values. It does not recursively merge dictionaries.
Open generation dictionaries must contain JSON-compatible nested values.

Source metadata preserves omitted versus explicit fields for runtime
construction. Offline resolution does not apply presets, global device overrides,
hardware checks, handler credential fallback, or local playback defaults.
An omitted credential retains its dataclass default. A missing explicit
environment reference fails even if a handler could use another credential.

## Environment values

Use `{env: NAME}` at a typed setting or `num_pipelines` position. Names must
match `[A-Za-z_][A-Za-z0-9_]*`. References are not allowed in IDs, kinds,
backend names, or stage references. There are no document variables,
interpolation templates, expressions, shell expansion, or `.env` loading.

- String fields keep environment text as text, including empty strings.
- Boolean fields accept exactly `true` or `false`.
- Integer fields accept signed decimal integers.
- Float fields accept finite decimal numbers, with optional exponents.
- Surrounding whitespace, hexadecimal text, and non-finite numbers fail.
- Collections must be YAML collections; environment text is not parsed as JSON
  or YAML. References can occur in known typed collection elements.

An environment-shaped mapping inside an open provider dictionary remains data.
It does not resolve an environment variable. Empty credentials stay empty.
Null is allowed only for nullable fields, with the VAD exception below.

Literal strings stay strings. For example, quoted `"2"` cannot supply an integer
setting. Explicit `false`, zero, empty strings, and empty dictionaries retain
their documented meanings. Existing required-value constraints still apply.
For VAD `max_speech_ms`, omission or `null` means unbounded duration and resolves
to the existing internal infinity default. Other explicit non-finite values fail.

## Paths

The resolver expands `~` and resolves relative paths against the YAML file's
directory only for these fields:

- `vad_firered_model_dir`, `smart_turn_model_path`
- `omnivoice_ref_audio`, `omnivoice_voice_clone_prompt`, `omnivoice_ref_voices_dir`
- `qwen3_tts_gguf_talker_path`, `qwen3_tts_gguf_codec_path`, `qwen3_tts_ref_cache_dir`
- `qwen3_tts_ref_audio`, `qwen3_tts_ref_spk`, `qwen3_tts_ref_rvq`

Resolution does not check path existence or read referenced files. Literal
paths do not expand environment-like text. Model IDs, URLs, checkpoint names,
importable tool names, and ambiguous Pocket voice strings stay literal.
Use absolute paths for ambiguous path-or-model-ID fields in version one.

## File rules and errors

The loader accepts one UTF-8 `.yaml` or `.yml` document with a mapping root.
The size limit is 1 MiB. Container nesting is limited to 32 levels, with the
root at level one. It rejects empty or multiple documents, nonstring mapping
keys, duplicate keys at every depth, anchors, aliases, merge keys, and explicit
tags. It reads local files only; there are no remote includes.

Scalar rules use lowercase `true`, `false`, `null`, and decimal numbers.
Words such as `on`, `off`, and `yes`, dates, and leading-zero numeric strings
remain strings. Explicit infinity and NaN are not accepted as numeric values.
The loader keeps these rules local rather than changing PyYAML globally.

Invalid configuration raises `ConfigurationError`, a `ValueError` subclass.
Errors include the file, configuration path, line, column, and correction when
available. They omit source snippets, input values, and environment contents.
A missing parser raises actionable `ImportError` instead.

The new loader rejects `.json` with guidance to use the existing flat JSON
interface. Existing JSON parsing, including its extra-key behavior, is unchanged.
Structured JSON support and concurrent named-pipeline serving are separate additions. Offline validation
does not verify provider access, assets, devices, or installed backend extras.

## Configured Python startup

Construct runtime resources explicitly after resolving the file:

```python
from threading import Event
from speech_to_speech.config import load_config, resolve_config
from speech_to_speech.configured_runtime import build_configured_server

document = load_config("voice.yaml")
resolved = resolve_config(document, names=["primary"], include_client=True)
stop_event = Event()
runtime = build_configured_server(resolved, stop_event)
try:
    runtime.start()
    runtime.wait()
finally:
    runtime.stop()
```

Supply the example's `PRIMARY_STT_KEY` environment value before resolution.
Construction initializes handlers and can download models, warm providers,
allocate resources, or use existing backend internal threads. Managed pipeline
workers, the listener, and the packaged client start through `start()`.
With `local=True`, the server constructor also creates one packaged audio client.

`build_configured_runtime(resolved, stop_event)` constructs worker pools without
a listener. It supports multiple selected definitions, exposed through the
runtime's `pools` mapping. Each definition gets its own isolated instances and
child stop event. Setting the caller's event stops the owned resources.

`build_configured_server` publishes exactly one selected definition. Its
`num_pipelines` can create multiple instances. Distinct named definitions are
not combined into the server's anonymous pool. Concurrent named routing is
deferred to a separate addition.

Library lifecycle methods install no signal handlers. Repeated successful
`start()` calls are harmless; a stopped or failed runtime cannot restart.
`stop()` works before startup and is safe to repeat. Construction or startup
failure cleans completed resources. Shutdown uses bounded joins and reports
workers that remain alive. Thread startup alone does not prove listener readiness.

## CLI file mode

```bash
speech-to-speech serve -f voice.yaml --name primary
speech-to-speech local --file voice.yaml --name primary
speech-to-speech talk -f voice.yaml
```

`-f` and `--file` are equivalent. Supply one file option. File mode accepts
pipeline selection and help, but does not accept legacy settings overrides.
Repeated `--name` selects distinct names; duplicates, empty names, unknown names,
and comma-separated names fail. A name without a file fails.

For `serve`, omitted names select all definitions. Server startup currently
requires exactly one selected definition and rejects larger sets before handler
construction. For `local`, select exactly one name when the file has several
definitions. Its packaged client does not use that name as its model.

`talk` resolves only client settings and does not construct server handlers.
It does not require server or block environment values. It rejects `--name`
until named client routing is supported. Help does not load the file.
Configuration errors exit with code 2; construction or startup errors exit
with code 1. CLI errors omit raw provider messages and credentials.

## Runtime defaults and client behavior

Construction applies dataclass defaults, applicable explicitly enabled macOS
preset defaults, explicit file values, then the pipeline-wide device override.
Environment references count as explicit file values. Required stage references
are never replaced by preset backend choices. All transformations use copies.
Lists and dictionaries replace complete values without a general deep merge.
Existing backend credential fallbacks and generation normalization still apply.
On Apple Silicon, several active instances disable live transcription under
the existing contention policy. Final transcription remains available.

Local mode binds loopback only. It rejects an explicit non-loopback host and
derives the client URL from the server port. A conflicting explicit client URL
fails. Local playback buffering defaults to 196 milliseconds for OpenAI TTS
and zero otherwise; an explicit zero remains zero. One client runs even when
the selected definition has multiple instances. Talk retains standalone defaults.

Tool modules load only during explicit client construction. An explicit tools
list conflicts with a tool module. Tools require an executor, and an explicit
`tool_response_create` overrides the module default. Conflicting explicit
runtime and client transcript logging settings fail; otherwise the explicit
value applies, with the existing disabled default when neither is supplied.
