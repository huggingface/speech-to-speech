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

The request and response shapes follow OpenAI's Audio API. In particular,
`gpt-transcribe` sends language hints with the official plural `languages[]`
field and reads the first detected language code from the plural `languages`
response. Older models and compatible servers continue to use the singular
`language` field.

The client accepts JSON and text responses. Use
`--openai_stt_response_format text` for a plain-text server. Transport and HTTP
errors are sanitized before they are surfaced to realtime clients, and failed
final requests do not create LLM work.

During setup, the handler transcribes one second of synthetic silence through
the configured endpoint. Endpoint, authentication, model, or response-format
failures therefore prevent the realtime server from accepting sessions.

## Turn and session lifecycle

Each pipeline delivers STT results asynchronously so HTTP work does not block
session teardown. Final requests for distinct turns retain their order within
that pipeline. They run independently of progressive work. Each pipeline retains
at most eight pending finals in addition to its active final request. Once this
queue is full, additional finals receive a sanitized `TranscriptionFailure`
without uploading audio or retrying. Obsolete pending requests are removed before
checking the limit, and accepted finals keep their order. This bounds the number
of utterances retained behind a stalled request; it is not an estimate of server
capacity.

Progressive requests are best-effort: one is active per pipeline, and only the
latest waiting cumulative window is retained. A final request cancels matching
active progressive work and discards its pending window. Newer turn revisions
invalidate older queued work and cancel older active requests. Relevance is
checked before HTTP dispatch and while a request is running.

Session end cancels active work and clears pending work. Results and failures
carry a session generation that is checked atomically with queue publication,
preventing old completions from appearing after teardown or in a reused session.
Shutdown cancels the remaining requests and joins the pipeline's request workers.
If a request worker cannot start, its pending work is cleared, final requests
receive a sanitized failure, and shutdown still reaches the pipeline boundary.

Cancellation interrupts the client's asynchronous HTTP transport, including a
request stalled before headers or in the response body. Server-side inference
cancellation is best-effort: closing the client connection does not guarantee
that the server stops GPU work. Stale-result filtering remains required even
when a request has been cancelled.

Client-side queue bounds apply separately to each pipeline. Pipelines do not
share an admission queue or endpoint concurrency budget. STT server capacity,
fleet routing, and provider quotas belong to the inference service or its shared
proxy.
