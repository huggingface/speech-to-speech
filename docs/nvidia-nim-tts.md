# NVIDIA Speech NIM TTS

`--tts nvidia_nim` sends speech synthesis to an
[NVIDIA Speech NIM](https://docs.nvidia.com/nim/speech/latest/tts/index.html)
(Riva TTS with Magpie voices) over its HTTP API. The NIM is not
OpenAI-compatible (`/v1/audio/speech` returns 404), so this backend uses the
NIM's own endpoints:

- `POST /v1/audio/synthesize_online` (default) streams raw 16-bit LPCM at the
  requested sample rate;
- `POST /v1/audio/synthesize` returns a complete WAV file.

Both take multipart form fields `text`, `language`, `voice`, and
`sample_rate_hz`. Audio is resampled to the pipeline's 16 kHz and reuses the
same cancellation and playback path as `--tts openai`. No extra Python
dependencies are needed.

## Deploy the NIM

Follow NVIDIA's
[deployment guide](https://docs.nvidia.com/nim/speech/latest/tts/deploy-tts-model.html)
for prerequisites, NGC login, and model profiles. The Magpie TTS Multilingual
`docker run` from NVIDIA's docs is:

```bash
export NGC_API_KEY=<your NGC API key>
export CONTAINER_ID=magpie-tts-multilingual
export NIM_TAGS_SELECTOR=name=magpie-tts-multilingual

docker run -it --rm --name=$CONTAINER_ID \
  --runtime=nvidia \
  --gpus '"device=0"' \
  --shm-size=8GB \
  -e NGC_API_KEY \
  -e NIM_HTTP_API_PORT=9000 \
  -e NIM_GRPC_API_PORT=50051 \
  -p 9000:9000 \
  -p 50051:50051 \
  -e NIM_TAGS_SELECTOR \
  nvcr.io/nim/nvidia/$CONTAINER_ID:latest
```

The default model profile uses `batch_size=8`; append another batch size to
`NIM_TAGS_SELECTOR` (for example `name=magpie-tts-multilingual,batch_size=32`)
as listed in the support matrix.

**DGX Spark:** DGX Spark is an Arm64 (Grace Blackwell) system with unified
memory. Use an image tag and model profile that NVIDIA's support matrix lists
for DGX Spark; profiles built only for x86 data-center GPUs will not start there.
speech-to-speech can run on the same machine and reach the NIM at
`http://localhost:9000`.

Check that the NIM is ready and list its voices:

```bash
curl http://localhost:9000/v1/health/ready      # {"status":"ready"}
curl http://localhost:9000/v1/audio/list_voices
```

## Run speech-to-speech

```bash
speech-to-speech serve \
  --tts nvidia_nim \
  --nvidia_nim_tts_base_url http://localhost:9000 \
  --nvidia_nim_tts_voice Magpie-Multilingual.EN-US.Aria.Happy
```

On startup the handler performs one short synthesis as a warm-up, so a wrong
URL, voice, or language fails immediately with the NIM's error message.

## Flags

| Flag | Default | Description |
| --- | --- | --- |
| `--nvidia_nim_tts_base_url` | `http://localhost:9000` | NIM HTTP base URL, without `/v1`. |
| `--nvidia_nim_tts_voice` | `Magpie-Multilingual.EN-US.Aria` | Voice sent when the client does not request a NIM voice. |
| `--nvidia_nim_tts_language` | `en-US` | Language code sent with every request. |
| `--nvidia_nim_tts_sample_rate` | `22050` | `sample_rate_hz` requested from the NIM. |
| `--nvidia_nim_tts_stream` | `true` | Stream PCM from `synthesize_online`; `false` fetches WAV from `synthesize`. |
| `--nvidia_nim_tts_timeout_s` | `30` | Total timeout for one synthesis request. |
| `--nvidia_nim_tts_api_key` | unset | Sent as `Authorization: Bearer ...`, for NVIDIA-hosted endpoints. |
| `--nvidia_nim_tts_voice_override` | unset | Always send this voice and ignore client-requested voices. |
| `--nvidia_nim_tts_fix_date_commas` | `true` | Drop the comma in `Month D, YYYY` so the NIM does not speak it as "sil"; see [Troubleshooting](#troubleshooting). |
| `--nvidia_nim_tts_max_chars` | auto | Split text longer than this; auto is 450 for Chatterbox voices, else 1,800. See [Limits](#limits). |

## Voices and emotional styles

Magpie voice names follow
`Magpie-Multilingual.{LOCALE}.{Speaker}[.{Emotion}]`, for example:

- `Magpie-Multilingual.EN-US.Aria` (neutral)
- `Magpie-Multilingual.EN-US.Aria.Happy`
- `Magpie-Multilingual.EN-US.Mia.Calm`

The available speakers and emotions depend on the deployed model; use
`/v1/audio/list_voices` to see them.

Realtime clients may request a voice per session or response. A requested voice
is used only if it looks like a NIM voice name (`Magpie-...` or
`Model.LOCALE.Speaker[.Emotion]`). Other names, such as the Qwen3-TTS speakers
sent by clients built for the default backend, are replaced by
`--nvidia_nim_tts_voice` with one INFO log per session.
`--nvidia_nim_tts_voice_override` ignores client voices entirely.

## Limits

The NIM rejects requests longer than 2,000 characters, or 500 characters for
Chatterbox TTS Multilingual. Both limits apply after text normalisation, which
expands numbers, abbreviations, and SSML tags; see NVIDIA's
[input text limits](https://docs.nvidia.com/nim/speech/latest/reference/api-references/tts/http-tts.html#input-text-limits).

Text longer than `--nvidia_nim_tts_max_chars` is split on sentence boundaries
(including `。！？` without a following space) and the pieces are synthesized in
order. By default the limit is 450 characters when the configured voice
contains `Chatterbox` and 1,800 otherwise, leaving headroom for normalisation.
Cancelling a response stops all of its remaining pieces.

SSML input (text that starts with `<speak` and ends with `</speak>`) is sent
whole when it fits. Longer SSML is split by its text only: each piece is sent as
its own `<speak>...</speak>` document without the inner tags, and a warning is
logged that pronunciation tags were removed to fit the limit.

## Zero-shot voices

Magpie TTS Zeroshot needs an audio prompt, which this backend does not send yet.
A voice or voice override containing `Zeroshot` fails at startup with a
`ValueError`; use a Magpie Multilingual or Chatterbox voice instead.

## Troubleshooting

### Dates are spoken with "sil"

Some NIM text normalisers (seen with Magpie TTS Multilingual NIM 26.10)
verbalise the comma between a day and a year as the pause token "sil", and the
voice speaks it: "November 4, 2026" comes out as "November fourth sil twenty
twenty six". This is the same bug as
[NeMo-Speech.cpp #18](https://github.com/NVIDIA/NeMo-Speech.cpp/issues/18),
fixed in NeMo-Speech.cpp 0.3.0 but still present in the NIM's normaliser.

`--nvidia_nim_tts_fix_date_commas` (on by default) drops only that comma before
the text is sent, so "Wednesday, November 4, 2026" becomes
"Wednesday, November 4 2026" and "Nov. 4th, 2026" becomes "Nov. 4th 2026". It
matches English month names and three-letter abbreviations followed by a day
and a year from 1900 to 2099; all other commas are left alone. Pass
`--nvidia_nim_tts_fix_date_commas false` if your NIM handles these dates
correctly.
