# YAML configuration

Use a YAML file to define reusable VAD, STT, LLM, and TTS blocks, connect them
into named pipelines, and start a selected pipeline from the CLI or Python.

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

`-f` and `--file` are equivalent. `serve` and `local` require exactly one
selected definition; omit `--name` only if the file contains one pipeline.
Its `num_pipelines` controls instance count. Serving different named definitions
concurrently is separate work in [#681](https://github.com/huggingface/speech-to-speech/issues/681).

`local` starts the server and one microphone/speaker client over loopback. Its
client URL follows the server port; an explicit non-loopback host or conflicting
client URL fails.

`talk` reads `runtime.client` and process logging settings. It does not start
pipeline handlers or require server/block environment values, and rejects
`--name`. With this example, it connects to an already running local server.
File mode accepts file selection, pipeline names, and help; put other
settings in YAML rather than mixing in legacy flags. Duplicate file options or
names and unknown names fail. Existing CLI flags and flat JSON input still work
outside file mode.

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
| `build_configured_server(resolved, stop_event, local=False)` | Construct a server for one definition; `local=True` adds the loopback client |

Use `include_client=True` when resolving client settings for local startup.
For client-only resolution, use `client_only=True` without pipeline names.
Returned settings contain secrets, so do not log or serialize them without redaction.

Loading, validation, and resolution do not construct handlers or start resources.
Construction can initialize models and providers; `start()` launches managed
workers but does not establish listener readiness. Library lifecycle methods
install no signal handlers. Always call `stop()`; a stopped or failed runtime cannot restart.
