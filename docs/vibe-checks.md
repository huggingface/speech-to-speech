# Big Bench Audio system tests

The harness streams a revision-pinned sample of
[ArtificialAnalysis/big_bench_audio](https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio)
through the engine's Realtime WebSocket API. Each recording gets a fresh session.
It exercises VAD, speech recognition, the language model, and speech synthesis.

The bundled `vibe` subset has 40 questions: ten each for formal fallacies,
navigation, object counting, and web of lies. Questions alternate categories,
so `--limit 4` is a small smoke test covering all four. The manifest pins the
dataset commit and official answers. Model weights and hosted providers are
not immutable; hold the server settings fixed and run comparisons close together.

## Development on Hugging Face

Commit the evaluation code, then create/update an image-building **private Docker
Space in your authenticated personal profile** (requires `huggingface_hub`):

```bash
python scripts/prepare_eval_space.py --space-name s2s-big-bench-audio-dev
```

The script uploads only tracked source and container files, records the Git
revision, and removes stale files within the paths it manages while preserving
unrelated Space files. It prints your launch command. Wait for the Space build to finish.
The Space stays on CPU and only runs a small HTTP server; GPU inference happens
in Jobs. This follows the [HF Docker Space image workflow](https://huggingface.co/docs/hub/jobs-images).

Set your personal namespace and the image/results repositories:

```bash
HF_NAMESPACE="your-hf-username"
EVAL_IMAGE="hf.co/spaces/${HF_NAMESPACE}/s2s-big-bench-audio-dev"
EVAL_RESULTS="${HF_NAMESPACE}/s2s-big-bench-audio-results"

hf jobs run --detach --namespace "$HF_NAMESPACE" --flavor a10g-large --timeout 45m --secrets HF_TOKEN \
    -e S2S_LIMIT=4 \
    -e S2S_PUSH_TO_HUB="$EVAL_RESULTS" \
    "$EVAL_IMAGE" vibe-check
```

Remove `S2S_LIMIT` and use `--timeout 90m` for all 40 questions. Hardware and
Inference Providers are billed to the personal account. Only launch the jobs
you need; no recurring jobs or automatic GPU CI are configured.
`S2S_LIMIT` only truncates the selected manifest: setting it to 1000 does not
expand the bundled 40-question subset.

The default stack is Parakeet TDT, `Qwen/Qwen3.5-9B:together` through HF
Inference Providers (Chat Completions), and Qwen3-TTS with its default GGML backend.
The token requires Jobs, repository-write, and Inference Providers permissions.
Credentials are passed as secrets and read from environment variables.

Reports go to `reports/<timestamp>-<label>.json` and server logs to
`logs/<label>.log` in the private results dataset. Job IDs are the default labels;
use unique labels to avoid replacing a previous log. Files also live under
`/output` inside the ephemeral Job. A hard Job timeout can prevent uploads, so
leave time for model downloads, initialization, and all questions.

| Environment variable | Default | Purpose |
|---|---|---|
| `S2S_LIMIT` | all 40 | Number of questions |
| `S2S_SUBSET` | `vibe` | Bundled subset or manifest path |
| `S2S_LABEL` | Job ID | Report label |
| `S2S_PUSH_TO_HUB` | unset | Private dataset for reports/logs |
| `S2S_COMPARE` | unset | Baseline file or `owner/repo:reports/file.json` |
| `S2S_LLM_MODEL` | `Qwen/Qwen3.5-9B:together` | Hosted model |
| `S2S_LLM_BASE_URL` | `https://router.huggingface.co/v1` | LLM endpoint |
| `S2S_SERVE_ARGS` | default stack above | Replace all server arguments, including the model/endpoint shortcuts (shell-style quoting, no expansion) |
| `S2S_EVAL_ARGS` | unset | Extra evaluation arguments, e.g. `--silence-ms 1200` |
| `S2S_REPORT_DIR` | `/output` | Local report/log directory |

`Dockerfile.eval` installs the engine from the uploaded source. Dependencies are
resolved at build time and runtime package versions are recorded in reports.
The build argument `EXTRAS="kokoro supertonic"` installs extra backends.
Model weights and evaluation audio download at runtime. The Space upload script creates the
`source-revision.txt` required by the Dockerfile.

## Model and prompt configurations

The STT and TTS run on the Job GPU. With a hosted LLM, its inference runs at the
selected provider. Changing the model, endpoint, prompt, or existing engine flags
does not require rebuilding the image. Installing a new optional backend does.

For the following 40-question examples, define a launcher using the variables
above. Each invocation submits one paid Job; run only the configurations you need.

```bash
run_eval() {
    hf jobs run --detach --namespace "$HF_NAMESPACE" --flavor a10g-large --timeout 90m \
        --secrets HF_TOKEN -e S2S_LIMIT=40 -e S2S_PUSH_TO_HUB="$EVAL_RESULTS" \
        "$@" "$EVAL_IMAGE" vibe-check
}

# These keep the default Parakeet STT, Qwen3-TTS GGML, and LLM request settings.
run_eval -e S2S_LLM_MODEL="deepseek-ai/DeepSeek-V4.1-Flash:baseten"
run_eval -e S2S_LLM_MODEL="zai-org/GLM-5.3-Flash:baseten"
```

Provider-specific request parameters matter. The engine's default Chat Completions
configuration sends `chat_template_kwargs.enable_thinking=false`. Cerebras
rejects that parameter; explicitly selecting `reasoning_effort=none` replaces it.
`S2S_SERVE_ARGS` replaces the entire default argument list, so include STT, TTS,
the model, and the endpoint:

```bash
EVAL_SERVER_ARGS="--stt parakeet-tdt --tts qwen3 --qwen3_tts_backend ggml --llm_backend chat-completions"

run_eval -e S2S_SERVE_ARGS="$EVAL_SERVER_ARGS --model_name Qwen/Qwen3.8-27B:cerebras --responses_api_base_url https://router.huggingface.co/v1 --responses_api_reasoning_effort none"
```

To use OpenAI, have `OPENAI_API_KEY` exported in the launching shell, then pass
its name as a Job secret. `HF_TOKEN` is still used to download assets and upload
results; an explicit `OPENAI_API_KEY` takes precedence for LLM authentication.
Never put a credential in `S2S_SERVE_ARGS` or the command line.

```bash
run_eval --secrets OPENAI_API_KEY \
    -e S2S_SERVE_ARGS="$EVAL_SERVER_ARGS --model_name gpt-6-luna --responses_api_base_url https://api.openai.com/v1 --responses_api_reasoning_effort none"
```

These are the configurations exercised in the development runs below, not an
exhaustive list of models. Hosted availability and provider behavior can change.
For latency comparisons, record explicit reasoning settings and keep them matched
where providers support the same controls. The default flag is only a request to
disable thinking; it does not verify the provider's internal behavior.

Override the evaluation system prompt through `S2S_EVAL_ARGS`:

```bash
run_eval -e S2S_EVAL_ARGS="--instructions 'Answer the spoken question briefly. End with Final answer: X, where X is Yes, No, valid, invalid, or a whole number.'"
```

This replaces the prompt in every question's fresh session and records it as
`target.instructions` in the report. Keep the `Final answer: X` convention for
reliable automatic extraction. `--instructions-file /path/to/prompt.txt` also
works when the file is available inside the Job container. Both argument
environment variables use shell-style quoting parsed by `shlex.split`; the Job
does not execute shell expressions or expand variables inside their values.

## Local use

```bash
# Inspect the sample without downloading audio or loading models.
python -m speech_to_speech.evals.big_bench_audio run --dry-run

# Start an engine, or use --url to connect to one already running.
python -m speech_to_speech.evals.big_bench_audio run \
    --spawn --limit 4 --out /tmp/smoke.json --spawn-log /tmp/s2s.log \
    -- --stt parakeet-tdt --tts qwen3

# Compare reports, including reports downloaded directly from the Hub.
python -m speech_to_speech.evals.big_bench_audio compare \
    /tmp/baseline.json /tmp/candidate.json

# Create a larger reproducible sample.
python -m speech_to_speech.evals.big_bench_audio build-subset \
    --size 120 --name deep --seed 1 --out /tmp/deep.json
```

Use `run --help` for all options, including custom instructions and an optional
OpenAI-compatible answer-extraction judge. Extra audio decoding dependencies
(`soundfile`, `scipy`) are included in the evaluation Docker image's installation.

## Reading results

Reports include the input transcript, assistant text, generated audio byte count,
per-category accuracy, parse rate, errors, time to first **audio**, and turn
latency. Answer accuracy grades the text sent to TTS; it does **not** transcribe
the synthesized waveform or measure pronunciation/intelligibility. Audio output
is required for a successful turn.

The prompt requests `Final answer: X`. A rule-based extractor compares X with the
official answer, with a fallback scan of unstructured replies. Inspect replies
when extraction is ambiguous. An optional `--judge-model` extracts an answer
using a second model; reports keep both verdicts. A completed judge verdict takes
precedence, including `UNPARSED`, which counts as unparsed even if the rule-based
extractor guessed an answer. A judge outage leaves the rule-based result in use.

Errors (including missing audio, failed/cancelled responses, timeouts, and split
turns) count against accuracy and make the process exit 1 **after** writing the
report. Incorrect reasoning remains a measured result rather than an engine
failure. Report comparisons show observed differences without an automatic
accuracy-regression gate.
Audio download and decoding failures use the per-item `--retries` budget; if
they persist, the runner records an item error and continues, preserving the
other results in the report.

The runner waits for `session.updated` before sending audio, raises the session
VAD silence threshold to 900 ms, and appends two seconds of silence to finish the
turn. A new speech-input item after a public response has begun marks the question
as a split, even if its next response has not arrived. Before any public response,
item IDs may change when speculative speech reopens after final transcription;
these changes are allowed. Multiple responses also mark a split. The last received
transcript is retained for diagnosis but the item is an error. Increase `--silence-ms` and
`--trailing-silence-ms` together if needed. `--speed 1` uses real-time pacing;
faster playback changes queueing and invalidates realistic latency comparisons.
`--response-timeout` bounds the wait after all input audio and trailing silence
have been sent, even while output events continue arriving. A response completed
within that budget remains valid during the final observation window.
Before streaming each question, session-capacity refusals are retried for up to
`--response-timeout` seconds (180 by default), allowing an earlier provider
request to drain after disconnection. This wait does not consume question retries
or send audio. Other connection/configuration errors fail normally.

Forty questions is a regression sample, not a leaderboard benchmark. Comparisons
show accuracy changes in percentage points, sample sizes, and latency changes;
they do not establish statistical significance. Comparisons warn about
different question sets, dataset revisions, or judge usage. Use matched settings
and examine individual failures before claiming a quality improvement.

## Tests

```bash
python -m pytest tests/evals -q
```

Unit tests use synthetic protocol events and do not download models or start
paid jobs. Job-entrypoint tests exercise environment overrides, quoted prompts,
provider authentication, failure exit codes, and redaction before log uploads.
The real GPU smoke run is a separate integration check.

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
