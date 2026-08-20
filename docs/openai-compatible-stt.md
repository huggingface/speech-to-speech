# OpenAI-compatible STT endpoint

The `openai` STT backend keeps VAD, turns, sessions, conversation state, and
response handling inside speech-to-speech while delegating recognition to an
external server:

```text
VAD audio -> POST /v1/audio/transcriptions
```

Each request uploads an in-memory mono PCM16 WAV at 16 kHz and accepts either
JSON with a string `text` field or a plain-text response.

## vLLM with Qwen3-ASR

Run a supported ASR model behind vLLM:

```bash
pip install "vllm[audio]"
vllm serve Qwen/Qwen3-ASR-1.7B --port 8000
```

Check the server, then select the remote backend. The LLM and TTS flags can
point at any other supported backends.

```bash
curl http://localhost:8000/v1/models

speech-to-speech local \
  --stt openai \
  --openai_stt_base_url http://localhost:8000/v1 \
  --openai_stt_model Qwen/Qwen3-ASR-1.7B
```

## NVIDIA Speech NIM

NVIDIA Speech ASR NIM exposes the same multipart endpoint. Select the NIM model
at deployment time, then let the request select its language when appropriate:

```bash
speech-to-speech local \
  --stt openai \
  --openai_stt_base_url http://localhost:9000/v1 \
  --openai_stt_model "" \
  --openai_stt_language en-US
```

## Authentication and compatibility

Set `--openai_stt_api_key` when the endpoint requires bearer authentication.
When the base URL is `https://api.openai.com/v1` and this flag is omitted, the
handler uses `OPENAI_API_KEY` if present. Other endpoints never receive that
environment credential implicitly.

The client accepts JSON and text responses. Use
`--openai_stt_response_format text` for a plain-text server. Transport and HTTP
errors are sanitized before they are surfaced to realtime clients, and failed
final requests do not create LLM work.

During setup, the handler transcribes one second of synthetic silence through
the configured endpoint. Endpoint, authentication, model, or response-format
failures therefore prevent the realtime server from accepting sessions.

This first endpoint adapter intentionally uses the existing serial STT handler
lifecycle. Endpoint-wide concurrency limits, bounded queues, coalescing, and
final-request priority can be added independently as an admission layer.
