# Big Bench Audio development notes

Historical validation of the evaluation harness. Run instructions are in
[Big Bench Audio system tests](vibe-checks.md). These records apply to the source
revisions stated below; they do not establish validation of later revisions.

## Development validation (2026-09-22)

The [personal-profile L4 smoke Job](https://huggingface.co/jobs/Steveeeeeeen/6ab2595852d0dbd7f1d7e2f6)
completed with all four categories producing audio and zero engine errors. Three
of four answers were correct; median time to first audio was 1.86 seconds.
This is a smoke result, not an accuracy estimate for the full dataset.
The [JSON report](https://huggingface.co/datasets/Steveeeeeeen/s2s-big-bench-audio-results/blob/main/reports/20260922T103259-6ab2595852d0dbd7f1d7e2f6.json)
and engine logs are private to the development account.

The run used the GGML server arguments now selected by default. The optional
Torch TTS configuration failed model initialization with `MimiConfig.rope_theta`
under Transformers 5.17.0 and faster-qwen3-tts 0.4.0; it is not a validated
configuration for this image. Its failed Job and engine log are retained in the
same account. No engine source workaround was introduced for that dependency
incompatibility.

## Four-model development comparison (2026-09-24)

All four configurations completed the same 40 questions on A10G-large Jobs with
Parakeet TDT, Qwen3-TTS GGML, and the default evaluation prompt. All 160 responses
produced audio and were parsed, with zero item-level runtime errors. The image
recorded source revision `123f73e30df19493716032f84a9150ece2f8f1fc`, Python 3.12.3,
Torch 2.11.0, Transformers 5.17.0, nano-parakeet 0.2.1, and faster-qwen3-tts 0.4.0.

| Model / provider | Correct | First audio p50 | First audio p95 | Turn p50 | Run duration |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B / Cerebras | 36/40 | 1.72 s | 2.84 s | 7.12 s | 22.3 min |
| DeepSeek-V4.1-Flash / Baseten | 36/40 | 1.86 s | 2.85 s | 5.54 s | 20.0 min |
| GLM-5.3-Flash / Baseten | 38/40 | 2.02 s | 5.39 s | 5.66 s | 22.6 min |
| GPT-6 Luna / OpenAI | 32/40 | 2.08 s | 2.85 s | 4.60 s | 19.3 min |

First audio and turn latency start at the end of the input recording; turn latency
ends at `response.done` and depends on response length. GLM had one 136.96-second
delay before audio, which is not apparent from its p95 alone. These single-run
results validate the harness and illustrate its reports; they do not establish a
model ranking or measure the full dataset.

Qwen/Cerebras and GPT-6 Luna used explicit `reasoning_effort=none`.
DeepSeek/Baseten and GLM/Baseten used the default chat-template flag described
above. Cerebras's first startup failed because it rejected that default flag;
the replacement Job used the explicit setting and completed. Three earlier
Novita Jobs were canceled before this comparison to change the provider choices.

The [comparison, individual reports, and logs](https://huggingface.co/datasets/Steveeeeeeen/s2s-big-bench-audio-results/blob/main/comparisons/20260924-models.md)
remain private to the development account; the summary here is available to
reviewers without access to that account. GPU validation applies to the source
revision recorded above. Subsequent fixes to Space synchronization, audio-loading
failures, judge verdicts, split detection, and response deadlines are covered by local regression tests and CI; the
GPU comparison has not been rerun for those fixes.


## Follow-up live validation

The [40-question run at `5aa4a3c`](https://huggingface.co/jobs/Steveeeeeeen/6ab530af6b030d633f68e615)
exposed an evaluation bug: finalized speculative transcripts can reopen with a new
input-item ID before any public response. Three such questions were falsely
flagged as splits. A provider HTTP 429 with a 60-second retry kept the last
session draining; the remaining 23 questions exhausted their short connection
retries. Fourteen questions produced correct spoken answers. This failed run is
not a model-accuracy comparison; its report and logs are retained in the private
results dataset.

The runner now permits input-ID changes before a public response and waits for
session capacity within a bounded budget. Regression tests include actual
service serialization with final transcription before reopening, continued
split detection after a response, recovery from capacity refusals, and capacity
timeout. A local WebSocket check also exercises both fixes together.
