# OpenAI-compatible STT endpoint

The `openai` STT backend keeps VAD, turns, sessions, conversation state, and
response handling inside speech-to-speech while delegating recognition to an
external server:

```text
VAD audio -> POST /v1/audio/transcriptions
```

Each request uploads an in-memory mono PCM16 WAV at 16 kHz and accepts either
JSON with a string `text` field or a plain-text response.

The STT endpoint has a process-wide admission controller. Pipelines using the
same normalized endpoint and credentials share its concurrency and queue limits.
Queued progressive windows are latest-only, finals take priority, and final or
new-revision work explicitly cancels superseded queued or active operations.
Closing an active HTTP transport is best-effort server cancellation; stale
results are always discarded locally.

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
  --openai_stt_model Qwen/Qwen3-ASR-1.7B \
  --openai_stt_max_concurrency 1 \
  --openai_stt_progressive_min_interval 0.75
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
When it is omitted, the handler uses `OPENAI_API_KEY` if present.

The client accepts JSON and text responses. Use
`--openai_stt_response_format text` for a plain-text server. Transport and HTTP
errors are sanitized before they are surfaced to realtime clients, and failed
final requests do not create LLM work.

Admission settings are process-wide for each normalized endpoint and credential
pair. When pipelines configure different settings for the same pair, the first
controller remains authoritative and a warning is logged.
