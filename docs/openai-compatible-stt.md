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

Progressive requests are best-effort. Each pipeline keeps at most one
progressive request in flight and drops newer progressive updates while it is
running. A final request is submitted independently, so it does not wait for an
in-flight progressive request; late progressive results are discarded.

## Stateful streaming STT

Use `--stt openai-realtime` or `--stt vllm-realtime` to send incremental PCM
over a persistent WebSocket. Local VAD controls which audio is sent and when
an utterance is committed. Streaming preserves the configured pre-speech
padding, trailing silence, and short-fragment merge gaps; idle microphone
audio outside those windows is not uploaded. Smart Turn and speculative
reopening still control when the assistant responds.

Both streaming backends forward provider partial transcripts as they arrive,
regardless of `--enable_live_transcription`. That flag enables repeated
whole-utterance requests for non-streaming STT backends; it is not needed for
native streaming.

### OpenAI Realtime transcription

```bash
export OPENAI_API_KEY=...

speech-to-speech local \
  --stt openai-realtime \
  --openai_realtime_stt_model gpt-live-transcribe
```

The hosted endpoint uses a transcription session with `intent=transcription`;
the model is configured in the session, not in the WebSocket URL. Hosted
OpenAI requires 24 kHz PCM, so the handler resamples the pipeline's 16 kHz
audio. Leave `--openai_realtime_stt_audio_sample_rate` at its default of
`24000`; other values fail during setup. See the
[OpenAI Realtime transcription guide](https://developers.openai.com/api/docs/guides/realtime-transcription).

`--openai_realtime_stt_api_key` accepts an explicit bearer token. The default
hosted endpoint uses `OPENAI_API_KEY` when that option is omitted; custom
endpoints do not receive the environment key implicitly.

### vLLM Realtime transcription (experimental)

The client uses vLLM's separate Realtime transcription protocol with 16 kHz
PCM. `--vllm_realtime_stt_model` must match a Realtime-capable model identifier
served by your endpoint; check `/v1/models`, including any custom served name.
The default `Qwen/Qwen3-ASR-1.7B` only works when that identifier is served.

For a server serving `mistralai/Voxtral-Mini-4B-Realtime-2602`:

```bash
curl http://localhost:8000/v1/models

speech-to-speech local \
  --stt vllm-realtime \
  --vllm_realtime_stt_base_url ws://localhost:8000/v1 \
  --vllm_realtime_stt_model mistralai/Voxtral-Mini-4B-Realtime-2602
```

Set `--vllm_realtime_stt_api_key` if the server requires authentication. See
[vLLM's Realtime API documentation](https://docs.vllm.ai/en/stable/serving/online_serving/speech_to_text/#realtime-api)
for supported models and server configuration.
