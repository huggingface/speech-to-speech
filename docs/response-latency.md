# Response latency logs

The server writes one INFO record when a Realtime response finishes. The record
includes its turn, revision, response key, terminal status, and available stage
durations:

```text
Turn turn_3 rev=0 latency: stt=0.18s llm_ttft=0.11s llm=1.24s tts_ttfa=0.12s e2e=1.61s vad_decision=0.36s smart_turn_analysis=0.03s smart_turn_wait=0.24s smart_turn_status=complete mlx_lock_wait=0.00s status=completed response_key=...
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
{"e2e_s":1.613482,"llm_s":1.241907,"llm_ttft_s":0.108531,"mlx_lock_wait_s":0.0,"response_key":"...","smart_analysis_s":0.031,"smart_delay_s":0.0,"smart_grace_s":0.8,"smart_status":"complete","smart_wait_s":0.24,"status":"completed","stt_s":0.181284,"tts_ttfa_s":0.121775,"turn_id":"turn_3","turn_revision":0,"vad_decision_s":0.36,"version":1}
```

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
  by TTS. In this implementation the timestamp is assigned when final
  `VADAudio` is constructed, **after** VAD silence detection, Smart Turn
  analysis, and any audio enhancement. It can include subsequent processing
  and response-release waits. This established handoff-to-first-audio boundary
  is unchanged; it is not physical speech end or client playback. Later text
  segments cannot replace the first-audio measurements.
- `vad_decision_s` starts at an **estimated end of voiced audio** and ends
  immediately when the VAD iterator returns the final segment, before Smart
  Turn analysis. Silero uses the first low-confidence chunk's start; FireRed
  uses its last speech frame's end. Each is a sample position mapped to the
  server's monotonic clock by subtracting the remaining audio samples from
  the time the VAD worker starts processing the final input chunk. Upstream
  queue time, network jitter, chunking, model frame resolution, and audio
  captured before processing limit its accuracy. It is
  `null` if the iterator provides no speech-end sample position.
- `smart_analysis_s` is elapsed wall time for the complete Smart Turn `predict`
  call, including preprocessing and inference. `smart_status` is `complete`,
  `incomplete`, `failed`, or `disabled`. A failed prediction records elapsed
  work and falls back to the ordinary short reopen grace with no processing
  delay. Disabled Smart Turn has `null` analysis, grace, delay, and wait; an
  enabled decision with no blocked gate has a measured `smart_wait_s` of zero.
- `smart_grace_s` and `smart_delay_s` are the **configured** response grace and
  processing delay selected by that decision. `smart_wait_s` is elapsed time
  actually blocked at the processing or response-release gates. Overlapping
  waits by different workers are counted once. Work that happens during a
  grace period without blocking a gate is not counted as waiting.

All intervals can overlap, including VAD decision and stages measured from the
later handoff; they must not be added to infer total latency. Tool follow-ups
have separate response keys and do not repeat the original VAD/Smart Turn work.

Remote timings are measured by this server. They include network transfer and
provider waits within the operation, so they are not provider-only inference
times. Stages can overlap; these fields are not an additive breakdown.
