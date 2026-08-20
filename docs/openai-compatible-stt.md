# OpenAI-compatible STT endpoint

The `openai` STT backend keeps VAD, turns, sessions, conversation state, and
response handling inside speech-to-speech while delegating recognition to an
external server:

```text
VAD audio -> POST /v1/audio/transcriptions
```

Each request uploads an in-memory mono PCM16 WAV at 16 kHz and accepts either
JSON with a string `text` field or a plain-text response. With live
transcription enabled, progressive updates upload the accumulated utterance
again, increasing request volume and provider usage.

## vLLM with Qwen3-ASR

Run a supported ASR model behind vLLM. Install vLLM in its own environment on
the machine serving STT; it does not need to be inside this repository.

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

## OpenAI-hosted transcription

The same backend can call OpenAI's hosted
[Transcription API](https://developers.openai.com/api/docs/guides/speech-to-text).
Export an API key and select a transcription model:

```bash
export OPENAI_API_KEY=...

speech-to-speech local \
  --stt openai \
  --openai_stt_base_url https://api.openai.com/v1 \
  --openai_stt_model gpt-transcribe \
  --openai_stt_response_format json
```

The LLM and TTS backends remain independently configurable; using OpenAI for
STT does not require using it for the rest of the pipeline.

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

Progressive requests are best-effort. Each pipeline keeps at most one
progressive request in flight and drops newer progressive updates while it is
running. A final request is submitted independently, so it does not wait for an
in-flight progressive request; late progressive results are discarded.
