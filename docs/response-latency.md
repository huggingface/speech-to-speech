# Response latency logs

The server writes one INFO record when a Realtime response finishes. The record
includes its turn, revision, response key, terminal status, and available stage
durations:

```text
Turn turn_3 rev=0 latency: stt=0.18s llm=1.24s llm_first_tts=0.42s llm_total=1.24s llm_rounds=1 tts_ttfa=0.12s e2e=2.01s vad_settle=0.61s vad_decision=0.36s hold=0.24s smart_turn_status=complete status=completed mlx_lock_wait=0.02s[ParakeetSTT-Progressive:0.02s] mlx_lock_hold=0.88s[Qwen3TTS:0.88s]
```

The same record is available with `speech-to-speech local` and
`speech-to-speech serve`.
`response_key` distinguishes a tool call from its spoken follow-up; both can
belong to the same turn and revision. A follow-up does not repeat the original
STT duration. Optional log-only fields such as `vad_settle`, `mlx_lock_*`, and
`tts=cut` appear only when they have something to report.

The terminal `response.done` event also carries the record in the
reserved `response.metadata["speech_to_speech.turn_latency"]` key. Realtime
metadata values are strings, so the value is compact JSON with this schema:

```json
{"e2e_s":2.013482,"hold_s":0.24,"llm_first_tts_s":0.42,"llm_rounds":1,"llm_s":1.241907,"llm_total_s":1.241907,"response_key":"...","smart_status":"complete","source":"speech","status":"completed","stt_s":0.181284,"tts_ttfa_s":0.121775,"turn_id":"turn_3","turn_revision":0,"vad_decision_s":0.36,"version":2}
```

Version 2 changes `e2e_s` from VAD handoff to **estimated speech end** to
first generated audio. It renames `smart_wait_s` to `hold_s` and removes
`llm_ttft_s`, `smart_analysis_s`, `smart_grace_s`, `smart_delay_s`, and
`mlx_lock_wait_s` from metadata. The demo accepts version 1 records but labels
their old handoff boundary explicitly instead of presenting them as new E2E.
Log-only fields from later work (`vad_settle`, `mlx_lock_hold`, `tts=cut`) stay
out of metadata and the demo.

Durations are seconds; existing fields retain their original precision and
the new VAD/Smart Turn fields use nanosecond precision in metadata. The
human-readable log uses two decimal places. Unavailable measurements are JSON `null`. Existing
client metadata is preserved, except that terminal responses remove any
client-supplied reserved latency key before adding the server measurement.
The reserved JSON value is included only when it fits Realtime's 512-character
metadata value limit. Client metadata is never shortened to make room.
The measurement is omitted when all 16
Realtime metadata slots are occupied by other client keys. `response.created`
continues to carry the client-supplied metadata unchanged; server measurements
are added only to terminal responses.

| Stage | Backends with a measured field | Backends with `n/a` pending coverage |
| --- | --- | --- |
| `stt` | `parakeet-tdt`, `openai`, `openai-realtime`, `vllm-realtime`, `whisper`, `whisper-mlx`, `mlx-audio-whisper`, `faster-whisper`, `qwen3-asr`, `parakeet-unified`, `nemotron-streaming`, `orukeet`, `paraformer`, `sense-voice` | None of the built-in STT backends (`--stt none` excluded) |
| `llm`, `llm_first_tts` | `transformers`, `mlx-lm`, `responses-api`, `chat-completions` | None of the built-in LLM backends |
| `tts_ttfa`, `e2e` | `qwen3`, `openai`, `pocket`, `kokoro`, `facebookMMS`, `chatTTS`, `omnivoice`, `kitten`, `supertonic` | None |

`--stt none` deliberately has no STT measurement. Any field is also `n/a`
when its stage did not run, produced no audio, or its measurement was unavailable.
An abandoned response has no terminal record. `mlx_lock_wait` and
`mlx_lock_hold` are included only in terminal logs on macOS, never in response
metadata or the demo. They measure the existing MLX lock; wait can be zero when
that lock is not contended, and hold is omitted entirely when nothing held it.

- `stt` covers final transcription processing only. Repeated progressive HTTP
  transcriptions and Realtime partial deltas are excluded. For HTTP STT, it
  starts when the final request worker begins and ends before its result is
  published. For Realtime STT, it starts when the final VAD commit is queued
  and ends when the provider's final transcript is received. For local STT
  models, it starts when the handler begins the final transcription and ends
  before its transcript is yielded.
- `llm` / `llm_s` covers generation for this response key, from serialization
  through consumption of provider output, including model lock and provider
  waits. It is recorded even if the request fails.
- `llm_first_tts` / `llm_first_tts_s` uses that same generation-start timestamp
  and ends immediately before the output processor yields its first nonempty,
  audio-enabled text input to TTS. It includes sentence batching or final flush,
  the LLM-to-processor queue, speculative output gates, and language processing.
  It excludes the downstream TTS queue, synthesis, and playback, so it is neither
  provider first-token time nor first-audio time. Nonstreaming replies measure
  the handoff after the complete reply. Failure fallback text also counts when
  it is forwarded; inspect terminal `status` rather than treating that as a
  successful model reply. Later text batches cannot overwrite the measurement.
  Text-only or tool-only responses, missing generation origins, stale output,
  discarded trackers and already discarded prefetches leave it unavailable.
  Tool follow-ups start their own measurement; they do not inherit the parent's
  first-batch time. A forwarded input does not guarantee that TTS later produces
  audio. This optional, additive version 2 field uses nine decimal places and is
  absent from older records.
- `llm_total_s` sums measured LLM generation durations across the initial request
  and its tool-result follow-ups. It excludes tool execution and waits between
  requests. Three generations of 0.7s, 0.9s and 0.8s produce 2.4s total, while
  the last response's `llm_s` remains 0.8s. Totals use nine decimal places in
  metadata.
- `llm_rounds` counts model requests in that chain. A direct answer has one;
  a tool request followed by an answer has two. Parallel tools within one
  generation count as one round. Only started requests count, including failed
  or interrupted requests. Totals include durations measured before the terminal
  event; cancellation can leave a running request's duration unavailable.
  Failed or discarded prefetch work does not contribute to replacement replies.

  These optional version 2 fields appear as unavailable in older records.
  The demo labels `llm_s` as **LLM for this response**.
- `tts_ttfa` starts when synthesis of the first text segment begins and ends
  when the first provider audio samples arrive. For HTTP TTS, WAV headers are
  excluded, and resampling and output block assembly happen afterward.
  Pocket Farsi captures this timestamp before buffering and validating the
  phrase, including across retries; it publishes the measurement only if
  validation produces usable audio. Generated sentence pauses are excluded
  from provider TTFA. Facebook MMS returns a full VITS waveform rather than
  incremental model audio: its TTFA ends after waveform computation completes
  on the selected device, before CPU copying, resampling, and PCM block assembly.
  A language-model reload is excluded from MMS TTFA but remains inside E2E;
  low TTFA alone therefore does not imply that all delay was upstream of TTS.
  OmniVoice likewise returns a full waveform from a blocking `generate` call:
  its TTFA ends when that waveform is available, before reshape, resampling,
  and PCM block assembly.
  Kitten synthesizes ONNX waveforms in blocking calls. Its TTFA ends when the
  first nonempty model waveform is available, before synthesis of later internal
  chunks, concatenation, `resample_poly`, and PCM block assembly. The measurement
  is published only if synthesis completes with usable, uncancelled output;
  E2E still includes all synthesis and assembly before the first yielded block.
  Supertonic likewise returns a full waveform from a blocking `synthesize`
  call: its TTFA ends when that waveform is available, before resampling and
  PCM block assembly.
- `e2e` runs from estimated speech end to the first audio block yielded by TTS.
  It includes VAD decision time, Smart Turn analysis, audio enhancement, and
  subsequent processing/hold time before that block. The timestamp is propagated
  separately from `VADAudio.created_at_s`, so processing gates retain their
  existing timing. It is unavailable when no speech-end estimate exists; there
  is no fallback to the old handoff boundary. Later synthesis segments cannot
  overwrite first audio. This is generated audio, not browser playback.
- `vad_settle` is the whole wait at the final STT input gate before a settled
  revision is accepted. It is not the same as `hold`: `hold` counts only the
  client-requested processing delay, while `vad_settle` covers every cause of
  gate wait. It is recorded only past a 0.05s threshold, so an immediately
  returning gate does not add a permanent `vad_settle=0.00s` to every line. It
  is log-only and is never part of response metadata.
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
- `mlx_lock_wait` / `mlx_lock_hold` (macOS terminal logs only) are time spent
  waiting for and holding the global MLX/Metal lock. Totals are attributed per
  handler name in brackets (`handler:0.88s`, with an `xN` suffix when the same
  handler acquired the lock several times). The bracketed sums always match the
  printed total. Progressive Parakeet binds the pending turn tracker so a failed
  short-timeout acquire still attributes wait time; the final transcription of
  the same revision reuses that slot. Hold is recorded on the outer release of a
  reentrant acquire only, so nested takes are not double-counted.
- `tts=cut` marks a cancelled turn that had already produced TTS audio
  (`tts_ttfa` was recorded). It means the cancellation arrived after synthesis
  had started, not that a sentence was chopped mid-word: a cancellation that
  lands just after the final chunk gets the marker too. It distinguishes
  "cancelled before it spoke" from "cancelled after it spoke".

Terminal records include an additive `source` field: `speech` for the first
response to supplied speech (including a manual create with automatic responses
disabled), `client` for an independent create, and `tool` for a continuation
resolved through function-call identity. This is a server diagnostic extension,
not an OpenAI-defined metadata field. Independent responses also export LLM/TTS
stage timings, with `e2e_s: null`; input ownership can still carry a turn ID for
speculative gating, so turn identity alone does not imply E2E eligibility. New
supplied user input supersedes an earlier tool timing origin, including when
private follow-up generation is rebuilt for that input. Subsequent tool calls
inherit the emitting response's timing eligibility, even when generation
ownership still names the earlier turn.

Tool follow-ups have separate response keys and do not repeat the originating
turn's VAD/Smart Turn measurements. Their E2E retains the originating speech-end
timestamp until the chain first yields audio, including intervening tool work.
Only one response in that chain records speech E2E. A spoken lead-in consumes
the first-audio measurement too; subsequent tool answers keep stage timings but
have no second E2E sample. Completed answers and cancelled/failed responses close
the speech anchor. Idle time before an independent create cannot reuse it.
These attribution changes retain metadata version 2's timing boundaries. `source`
is a new field, and `turn_id` is `null` when a `client` or `tool` record has no
attributed input turn. Out-of-band responses retain their existing metadata
behavior. Stages overlap; do not sum them to reconstruct E2E. Configured grace
and delay and Smart Turn analysis are no longer part of the terminal timing record.

LLM totals follow explicit tool origins within the same session, turn and
revision, including typed chains without a speech turn. The server retains the
last 128 successful tool responses' timings per session for delayed follow-ups
and clears them on session teardown.

Remote timings are measured by this server. They include network transfer and
provider waits within the operation, so they are not provider-only inference
times. Stages can overlap; these fields are not an additive breakdown.

## Correlating remote requests with workers

Remote HTTP STT (including the Nemotron profile), conversational Chat
Completions/Responses requests, and OpenAI-compatible TTS batches send a generated
opaque `X-Speech-Request-Id`. INFO logs from
`speech_to_speech.utils.request_tracing` are bounded JSON records containing only
technical identifiers:

```json
{"event":"speech_response_binding","session_id":"session-example","response_key":"response-example","turn_id":"turn_1","turn_revision":1,"source":"speech"}
{"event":"speech_http_stt_result","request_id":"0123456789abcdef0123456789abcdef","stage":"stt","session_id":"session-example","turn_id":"turn_1","turn_revision":1,"response_key":null,"mode":"final"}
{"event":"speech_http_dispatch","request_id":"abcdef0123456789abcdef0123456789","stage":"tts","session_id":"session-example","turn_id":"turn_1","turn_revision":1,"response_key":"response-example","mode":null}
```

Start with the slow record's `response_key`. Its binding identifies the server
session and turn/revision; match accepted STT results on those fields, and LLM/TTS
dispatches on the response key. Turn names such as `turn_1` reset between sessions,
so never join different sessions by turn name alone. A proxy that implements this
header contract can log each ID alongside its selected worker and forward it to
worker timing logs. These upstream logs alone cannot establish which worker was
selected. Custom-header preservation through a deployment's ingress must be
validated separately.

Each HTTP STT operation, logical LLM provider request and nonempty TTS batch has
its own ID. Multiple synthesis batches can use different workers. The SDK and a
proxy may reuse a logical operation ID across retries: find all matching attempt
records, rather than treating an ID as one physical network attempt. Dispatch
means an attempt, not proof of worker receipt, success or stream completion.
Cancellation before dispatch emits no dispatch record; failed and cancelled
dispatched STT operations retain their owning session/turn identity even when
there is no accepted transcription or response.

Accepted final STT records include the IDs actually used by that result,
including cached prefix windows from earlier revisions. Associations are emitted
only after the current result passes the existing publication gate. Partial
captions do not create final-result associations. Private results retain one
fixed-width ID per already-retained window, with the existing pruning and session
teardown; there is no global request history. Memory follows the existing window
cache, whose window count has no absolute cap for an arbitrarily long utterance.
Each association is a separate bounded log record, not an unbounded metadata
array.

The server session binding is private, distinct from client session parameters,
and snapshotted before worker/thread handoff. Reusing settings/history for a new
session adopts a new wrapper; queued or in-flight calls keep their original
owner. Unbound standalone configurations have an unavailable session ID. Warmup
and background history-compaction calls do not borrow the active response's
identity and are outside this conversational-request contract.

No request ID is added to transcript events or terminal timing metadata, so this
does not consume Realtime metadata space. The logs contain no request/response
content, URLs or credentials; unsafe correlation values are omitted instead of
rewritten. IDs are diagnostic, never authentication or deduplication tokens.
Tracing needs retained technical logs for the relevant time range; the timing
metadata is not a persistent archive of every worker attempt.
