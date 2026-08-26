# OpenAI-compatible TTS endpoint

The `openai` TTS backend keeps turns, sessions, conversation state, and response
handling inside speech-to-speech while delegating synthesis to an external
server:

```text
LLM text -> POST /v1/audio/speech -> PCM16 at 16 kHz
```

## vLLM-Omni with Qwen3-TTS

Run Qwen3-TTS in a separate vLLM-Omni process. Keep the installed vLLM and
vLLM-Omni versions aligned:

```bash
uv pip install vllm==0.24.0 --torch-backend=auto
uv pip install vllm-omni

vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
  --omni \
  --port 8091 \
  --trust-remote-code \
  --enforce-eager
```

Check the server before starting the pipeline:

```bash
curl http://localhost:8091/v1/audio/voices
```

Then select the remote TTS backend. The STT and LLM flags are only examples and
can point at any supported backends.

```bash
speech-to-speech local \
  --stt parakeet-tdt \
  --llm_backend responses-api \
  --tts openai \
  --openai_tts_base_url http://localhost:8091/v1 \
  --openai_tts_model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --openai_tts_voice aiden \
  --openai_tts_sample_rate 24000 \
  --openai_tts_stream true
```

Qwen3-TTS produces 24 kHz audio. The client incrementally converts its raw
PCM16 response to the pipeline's mono, signed-int16, 16 kHz, 512-sample chunks.
If a server requires an explicit language, pass its expected value, such as
`--openai_tts_language English` for Qwen3-TTS. The value is forwarded unchanged
with every speech request; the TTS handler does not infer it from the pipeline.

## OpenAI and standard-compatible servers

Standard OpenAI-compatible endpoints do not need the vLLM streaming extension.
For example, with `OPENAI_API_KEY` set:

```bash
speech-to-speech local \
  --tts openai \
  --openai_tts_base_url https://api.openai.com/v1 \
  --openai_tts_model gpt-4o-mini-tts \
  --openai_tts_voice alloy \
  --openai_tts_response_format pcm \
  --openai_tts_sample_rate 24000
```

The default request uses the standard `stream_format=audio` field and consumes
the HTTP response body incrementally. It does not send the non-standard
`stream` field.

## Playback buffering

Sometimes the first audio chunks arrive unevenly: an initial burst can be
followed by a short gap before delivery settles into a steady stream. Starting
playback immediately can then empty the speaker queue and cause a brief stutter.
A startup buffer holds a small amount of audio before playback begins so it can
absorb that early jitter. If playback later catches up with the incoming stream,
the client uses the same buffer before restarting.

The buffer is generally 0 ms, so other local TTS backends and
`speech-to-speech talk` start playback as soon as audio arrives. This keeps
latency to a minimum when chunks are delivered evenly. The `talk` client cannot
choose a backend-specific default because it connects to a Realtime endpoint
without knowing which TTS backend produced the audio.

For `speech-to-speech local --tts openai`, the default changes to 196 ms. The
OpenAI-compatible backend receives synthesized audio over a separate HTTP
connection to the `/audio/speech` endpoint, where the first chunks can have more
substantial delivery jitter than in-process TTS. The extra cushion trades a
small startup delay for smoother playback.

The buffer is downstream of TTS and does not alter the TTS request, synthesis,
resampling, or server deployment.

Override the default for either command with `--playback-buffer-ms`. An
explicit `0` disables buffering even for `local --tts openai`. Higher values
improve tolerance for delivery jitter at the cost of a later start to each
response; lower values reduce that startup delay but increase underrun risk.
For example:

```bash
speech-to-speech local \
  --playback-buffer-ms 256 \
  --tts openai \
  --openai_tts_base_url https://api.openai.com/v1
```

Responses shorter than the configured buffer are played as soon as the audio
response completes. Browser, WebRTC, and third-party Realtime clients use their
own playback behavior and are not affected by this option.

## Authentication and compatibility

Set `--openai_tts_api_key` when the endpoint requires bearer authentication.
When it is omitted, the handler uses `OPENAI_API_KEY` if present.

The backend performs a short synthesis request during startup. Invalid endpoint,
authentication, model, voice, request, and audio-response configuration therefore
fail before the Realtime server accepts sessions.

The client accepts:

- raw signed PCM16 with a configured `--openai_tts_sample_rate`; or
- a WAV response with `--openai_tts_stream false` and
  `--openai_tts_response_format wav`.

Both formats are decoded, resampled, and forwarded incrementally as their HTTP
response bodies arrive.

The `stream_format=audio` field is part of the standard request shape.
`stream=true` is a vLLM-Omni extension and is disabled by default; enable it
explicitly with `--openai_tts_stream true` when the server supports it.

Cancellation is best-effort: barge-in and session teardown close the client's
active HTTP response and prevent further audio publication locally. The
standard `/v1/audio/speech` interface has no portable server-side cancellation
operation, so a disconnected client does not guarantee that endpoint
computation stops immediately.
