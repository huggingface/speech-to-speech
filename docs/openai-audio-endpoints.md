# OpenAI-compatible STT and TTS endpoints

The `openai` STT and TTS backends keep VAD, turns, sessions, conversation state,
and response handling inside speech-to-speech. They send individual inference
requests to external servers:

```text
VAD audio -> POST /v1/audio/transcriptions
LLM text  -> POST /v1/audio/speech -> PCM16 at 16 kHz
```

The STT endpoint has a process-wide admission controller. Pipelines using the
same normalized endpoint and credentials share its concurrency and queue limits.
Queued progressive windows are latest-only, finals take priority, and final or
new-revision work explicitly cancels superseded queued/active operations.
Closing an active HTTP transport is best-effort server cancellation; stale
results are always discarded locally.

## vLLM: Qwen3-ASR and Qwen3-TTS

Current vLLM exposes `/v1/audio/transcriptions` for supported ASR models.
Standalone Parakeet TDT is not currently in vLLM's published transcription
model list, so use Qwen3-ASR for an all-vLLM test:

```bash
pip install "vllm[audio]"
vllm serve Qwen/Qwen3-ASR-1.7B --port 8000
```

Run Qwen3-TTS through a separate vLLM-Omni process. Keep the installed vLLM and
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

Check both servers before starting the pipeline:

```bash
curl http://localhost:8000/v1/models
curl http://localhost:8091/v1/audio/voices
```

Then run speech-to-speech. The LLM flags below are only an example; they can
point at any existing supported LLM backend.

```bash
speech-to-speech \
  --mode realtime \
  --stt openai \
  --openai_stt_base_url http://localhost:8000/v1 \
  --openai_stt_model Qwen/Qwen3-ASR-1.7B \
  --openai_stt_max_concurrency 1 \
  --openai_stt_progressive_min_interval 0.75 \
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

## Parakeet TDT v3

Current core vLLM does not list standalone Parakeet among its transcription
architectures. NVIDIA Speech ASR NIM does expose the same
`/v1/audio/transcriptions` multipart endpoint and supports its Parakeet 0.6B
TDT container. Select the NIM model at deployment time, expose its HTTP port
(9000 by default), and let the request select its language:

```bash
speech-to-speech \
  --stt openai \
  --openai_stt_base_url http://localhost:9000/v1 \
  --openai_stt_model "" \
  --openai_stt_language en-US \
  --tts openai \
  --openai_tts_base_url http://localhost:8091/v1 \
  --openai_tts_model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

NVIDIA currently names the NIM container `parakeet-0.6b-tdt`; its published
profiles include `type=default` and `type=multi`. Follow the
[NVIDIA ASR NIM deployment guide](https://docs.nvidia.com/nim/speech/latest/get-started/tutorials/asr.html)
and
[support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html)
to select the profile for your GPU. This is the direct prebuilt-server path for
Parakeet; using the Hugging Face `nvidia/parakeet-tdt-0.6b-v3` checkpoint
directly still requires a server or compatibility shim that exposes this HTTP
contract.

The server must accept an in-memory mono 16 kHz WAV upload and return either
JSON with a string `text` field or plain text. Use
`--openai_stt_response_format text` for the latter.

## Authentication and compatibility

Set `--openai_stt_api_key` and `--openai_tts_api_key` independently. When a
value is omitted, each handler uses `OPENAI_API_KEY` if present. Keys are never
included in admission-controller labels.

The TTS client currently accepts:

- raw signed PCM16 with a configured `--openai_tts_sample_rate`; or
- a complete WAV response with `--openai_tts_stream false` and
  `--openai_tts_response_format wav`.

The raw streaming fields `stream=true` and `stream_format=audio` are
vLLM-Omni extensions. Disable `--openai_tts_stream` for servers that implement
only the standard request shape.
