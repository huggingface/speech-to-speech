# Response latency logs

The server writes one INFO record when a Realtime response finishes. The record
includes its turn, revision, response key, terminal status, and available stage
durations:

```text
Turn turn_3 rev=0 latency: stt=0.18s llm_ttft=0.11s llm=1.24s tts_ttfa=0.12s e2e=1.61s mlx_lock_wait=0.00s status=completed response_key=...
```

The same record is available with `speech-to-speech local` and
`speech-to-speech serve`.
`response_key` distinguishes a tool call from its spoken follow-up; both can
belong to the same turn and revision. A follow-up does not repeat the original
STT duration.

The terminal `response.done` event also carries the unrounded record in the
reserved `response.metadata["speech_to_speech.turn_latency"]` key. Realtime
metadata values are strings, so the value is compact JSON with this schema:

```json
{"e2e_s":1.613482,"llm_s":1.241907,"llm_ttft_s":0.108531,"mlx_lock_wait_s":0.0,"response_key":"...","status":"completed","stt_s":0.181284,"tts_ttfa_s":0.121775,"turn_id":"turn_3","turn_revision":0,"version":1}
```

Durations are raw seconds and are not rounded to the two decimal places used in
the human-readable log. Unavailable measurements are JSON `null`. Existing
client metadata is preserved, except that terminal responses remove any
client-supplied reserved latency key before adding the server measurement.
The measurement is omitted when no attributed turn exists or when all 16
Realtime metadata slots are occupied by other client keys. `response.created`
continues to carry the client-supplied metadata unchanged; server measurements
are added only to terminal responses.

| Stage | Backends with a measured field | Backends with `n/a` pending coverage |
| --- | --- | --- |
| `stt` | `parakeet-tdt`, `openai`, `openai-realtime`, `vllm-realtime` | `whisper`, `whisper-mlx`, `mlx-audio-whisper`, `faster-whisper`, `parakeet-unified`, `paraformer`, `qwen3-asr` |
| `llm` | `transformers`, `mlx-lm`, `responses-api`, `chat-completions` | None of the built-in LLM backends |
| `llm_ttft` | streaming `responses-api`, streaming `chat-completions` | `transformers`, `mlx-lm`, and non-streaming remote requests |
| `tts_ttfa`, `e2e` | `qwen3`, `openai` | `chatTTS`, `facebookMMS`, `omnivoice`, `pocket`, `kokoro`, `supertonic` |

`--stt none` deliberately has no STT measurement. Any field is also `n/a`
when its stage did not run, produced no audio, or its measurement was unavailable.
An abandoned response has no terminal record. `mlx_lock_wait` only measures
the existing MLX lock and remains zero for other backends.

- `stt` covers final transcription processing only. Repeated progressive HTTP
  transcriptions and Realtime partial deltas are excluded. For HTTP STT, it
  starts when the final request worker begins and ends before its result is
  published. For Realtime STT, it starts when the final VAD commit is queued
  and ends when the provider's final transcript is received.
- `llm_ttft` starts before serialization and ends at the first non-whitespace
  provider `TextDelta`. Protocol-only stream events do not complete it. This is
  distinct from the first sentence batch released to TTS. `llm` covers full
  generation, from serialization and provider request through consumption of
  the provider output. It is recorded even when the request fails after
  starting.
- `tts_ttfa` starts when synthesis of the first text segment begins and ends
  when the first provider audio samples arrive. For HTTP TTS, WAV headers are
  excluded, and resampling and output block assembly happen afterward.
- `e2e` runs from the speech-stop timestamp to the first audio block yielded
  by TTS. It does not include client buffering or playback. Later text
  segments cannot replace the first-audio measurements.

Remote timings are measured by this server. They include network transfer and
provider waits within the operation, so they are not provider-only inference
times. Stages can overlap; these fields are not an additive breakdown.
