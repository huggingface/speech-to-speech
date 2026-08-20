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
  --openai_tts_language Auto \
  --openai_tts_sample_rate 24000 \
  --openai_tts_stream true
```

Qwen3-TTS produces 24 kHz audio. The client incrementally converts its raw
PCM16 response to the pipeline's mono, signed-int16, 16 kHz, 512-sample chunks.

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

## Authentication and compatibility

Set `--openai_tts_api_key` when the endpoint requires bearer authentication.
When it is omitted, the handler uses `OPENAI_API_KEY` if present.

The client accepts:

- raw signed PCM16 with a configured `--openai_tts_sample_rate`; or
- a complete WAV response with `--openai_tts_stream false` and
  `--openai_tts_response_format wav`.

The `stream_format=audio` field is part of the standard request shape.
`stream=true` is a vLLM-Omni extension and is disabled by default; enable it
explicitly with `--openai_tts_stream true` when the server supports it.

Cancellation is best-effort: barge-in and session teardown close the client's
active HTTP response and prevent further audio publication locally. The
standard `/v1/audio/speech` interface has no portable server-side cancellation
operation, so a disconnected client does not guarantee that endpoint
computation stops immediately.
