# Response latency logs

The server writes one INFO record when a Realtime response finishes. The record
includes its turn, revision, response key, terminal status, and available stage
durations:

```text
Turn turn_3 rev=0 latency: stt=0.18s llm=1.24s tts_ttfa=0.12s e2e=2.01s vad_decision=0.36s hold=0.24s smart_turn_status=complete status=completed response_key=...
```

The same record is available with `speech-to-speech local` and
`speech-to-speech serve`.
`response_key` distinguishes a tool call from its spoken follow-up; both can
belong to the same turn and revision. A follow-up does not repeat the original
STT duration.

The terminal `response.done` event also carries the record in the
reserved `response.metadata["speech_to_speech.turn_latency"]` key. Realtime
metadata values are strings, so the value is compact JSON with this schema:

```json
{"e2e_s":2.013482,"hold_s":0.24,"llm_s":1.241907,"response_key":"...","smart_status":"complete","status":"completed","stt_s":0.181284,"tts_ttfa_s":0.121775,"turn_id":"turn_3","turn_revision":0,"vad_decision_s":0.36,"version":2}
```

Version 2 changes `e2e_s` from VAD handoff to **estimated speech end** to
first generated audio. It renames `smart_wait_s` to `hold_s` and removes
`llm_ttft_s`, `smart_analysis_s`, `smart_grace_s`, `smart_delay_s`, and
`mlx_lock_wait_s` from metadata. The demo accepts version 1 records but labels
their old handoff boundary explicitly instead of presenting them as new E2E.

Durations are seconds; existing fields retain their original precision and
the new VAD/Smart Turn fields use nanosecond precision in metadata. The
human-readable log uses two decimal places. Unavailable measurements are JSON `null`. Existing
client metadata is preserved, except that terminal responses remove any
client-supplied reserved latency key before adding the server measurement.
The reserved JSON value is included only when it fits Realtime's 512-character
metadata value limit. Client metadata is never shortened to make room.
The measurement is omitted when no attributed turn exists or when all 16
Realtime metadata slots are occupied by other client keys. `response.created`
continues to carry the client-supplied metadata unchanged; server measurements
are added only to terminal responses.

| Stage | Backends with a measured field | Backends with `n/a` pending coverage |
| --- | --- | --- |
| `stt` | `parakeet-tdt`, `openai`, `openai-realtime`, `vllm-realtime`, `whisper`, `whisper-mlx`, `mlx-audio-whisper`, `faster-whisper`, `qwen3-asr` | `parakeet-unified`, `paraformer` |
| `llm` | `transformers`, `mlx-lm`, `responses-api`, `chat-completions` | None of the built-in LLM backends |
| `tts_ttfa`, `e2e` | `qwen3`, `openai`, `pocket` | `chatTTS`, `facebookMMS`, `omnivoice`, `kokoro`, `supertonic` |

`--stt none` deliberately has no STT measurement. Any field is also `n/a`
when its stage did not run, produced no audio, or its measurement was unavailable.
An abandoned response has no terminal record. `mlx_lock_wait` is included only
in terminal logs on macOS, never in response metadata or the demo. It measures
the existing MLX lock and can be zero when that lock is not used.

- `stt` covers final transcription processing only. Repeated progressive HTTP
  transcriptions and Realtime partial deltas are excluded. For HTTP STT, it
  starts when the final request worker begins and ends before its result is
  published. For Realtime STT, it starts when the final VAD commit is queued
  and ends when the provider's final transcript is received. For local STT
  models, it starts when the handler begins the final transcription and ends
  before its transcript is yielded.
- `llm` covers full generation, from serialization and provider request through
  consumption of provider output. It is recorded even if the request fails.
- `tts_ttfa` starts when synthesis of the first text segment begins and ends
  when the first provider audio samples arrive. For HTTP TTS, WAV headers are
  excluded, and resampling and output block assembly happen afterward.
- `e2e` runs from estimated speech end to the first audio block yielded by TTS.
  It includes VAD decision time, Smart Turn analysis, audio enhancement, and
  subsequent processing/hold time before that block. The timestamp is propagated
  separately from `VADAudio.created_at_s`, so processing gates retain their
  existing timing. It is unavailable when no speech-end estimate exists; there
  is no fallback to the old handoff boundary. Later synthesis segments cannot
  overwrite first audio. This is generated audio, not browser playback.
- `vad_decision_s` starts at an **estimated end of voiced audio** and ends
  immediately when the VAD iterator returns the final segment, before Smart
  Turn analysis. Silero uses the first low-confidence chunk's start; FireRed
  uses its last speech frame's end. Each is a sample position mapped to the
  server's monotonic clock by subtracting the remaining audio samples from
  the time the VAD worker starts processing the final input chunk. Upstream
  queue time, network jitter, chunking, model frame resolution, and audio
  captured before processing limit its accuracy. It is
  `null` if the iterator provides no speech-end sample position.
- `smart_status` is the Smart Turn decision: `complete`, `incomplete`, `failed`,
  or `disabled`. Failure still falls back to the ordinary short reopen grace
  without an extra processing delay; no inference or speculation policy changes.
- `hold_s` (Hold time before response) is actual elapsed time blocked at processing
  or response-release gates. Overlapping worker waits count once. Time spent
  doing useful work during a configured grace window is not hold time. Disabled
  Smart Turn has `null`; an enabled decision without a blocked gate has zero.

Tool follow-ups have separate response keys and do not repeat the originating
turn's VAD/Smart Turn measurements. Their E2E retains the originating speech-end
timestamp and therefore includes intervening tool work. Stages overlap; do not
sum them to reconstruct E2E. Configured grace and delay and Smart Turn analysis
are no longer part of the terminal timing record.

Remote timings are measured by this server. They include network transfer and
provider waits within the operation, so they are not provider-only inference
times. Stages can overlap; these fields are not an additive breakdown.
