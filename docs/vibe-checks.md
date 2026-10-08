# Big Bench Audio system tests

Run pinned [Big Bench Audio](https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio)
recordings through the engine's Realtime WebSocket API, exercising VAD, STT,
the LLM, and TTS. Each question uses a fresh session. Reports contain answer
accuracy, audio latency, and protocol failures.

## Local use

Install the evaluator's dependencies into your existing engine environment:

```bash
python -m pip install "soundfile>=0.13.0" "websockets>=14.0"

# Inspect the default sample without downloading audio or loading models.
python -m speech_to_speech.evals.big_bench_audio run --dry-run

# Start an engine for a four-question smoke test.
python -m speech_to_speech.evals.big_bench_audio run \
    --spawn --limit 4 --out /tmp/smoke.json --spawn-log /tmp/s2s.log \
    -- --stt parakeet-tdt --tts qwen3

# Run the whole benchmark.
python -m speech_to_speech.evals.big_bench_audio run --subset full \
    --spawn --out /tmp/full.json -- --stt parakeet-tdt --tts qwen3

# Compare saved reports.
python -m speech_to_speech.evals.big_bench_audio compare \
    /tmp/baseline.json /tmp/candidate.json
```

Use `--url` instead of `--spawn` to connect to an existing engine. See
`run --help` for all options, including custom prompts and an optional
OpenAI-compatible answer-extraction judge.

## Question selection

- `vibe` is a committed 40-question sample, stratified by category and official
  answer with seed `0`. It contains ten questions per category, interleaved so
  `--limit 4` covers all four categories. No new sample is drawn for a run.
- `full` loads all 1,000 records from upstream metadata at the same pinned dataset
  revision, in upstream order. Metadata and audio are cached by the Hub client.
  A full-selection `--dry-run` loads metadata but does not download audio.

`--limit` only takes a prefix of the selection; it never adds questions.
A prefix of `full` need not cover every category. To create a custom manifest:

```bash
python -m speech_to_speech.evals.big_bench_audio build-subset \
    --size 120 --name deep --seed 1 --out /tmp/deep.json
```

Pass that path as `--subset /tmp/deep.json`. Use `--size 40 --name vibe --seed 0`
to reproduce the bundled sample. In containers, include or mount a custom
manifest and use its container path.

## Build and run with Docker

From the repository root, build the committed source with its revision recorded:

```bash
EVAL_CONTEXT="$(mktemp -d)"
git archive HEAD | tar -x -C "$EVAL_CONTEXT"
git rev-parse HEAD > "$EVAL_CONTEXT/source-revision.txt"
docker build --platform linux/amd64 \
    -f "$EVAL_CONTEXT/Dockerfile.eval" \
    -t s2s-big-bench-audio:local "$EVAL_CONTEXT"
rm -rf "$EVAL_CONTEXT"
```

Commit changes before building. The image includes evaluator dependencies;
add `--build-arg EXTRAS="kokoro supertonic"` for optional engine backends.
GPU execution requires a Linux/amd64 NVIDIA host with the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Building on another architecture requires amd64 emulation.

Export `HF_TOKEN` in your shell, then run:

```bash
docker run --name s2s-eval-smoke --platform linux/amd64 --gpus all \
    -e HF_TOKEN -e S2S_LIMIT=4 -e S2S_LABEL=local-smoke \
    s2s-big-bench-audio:local vibe-check
docker cp s2s-eval-smoke:/output ./eval-output
docker rm s2s-eval-smoke
```

The container runs the engine and evaluator together; no exposed port is needed.
Copy `/output` before removing it, including after failure, to retain the report
and redacted server log. The settings below also work with `docker run -e`.
For all questions, add `-e S2S_SUBSET=full` and remove `-e S2S_LIMIT=4`.

## Hugging Face Jobs

Use an existing registry image, or commit your changes and build one in a private
personal Docker Space:

```bash
python scripts/prepare_eval_space.py --space-name s2s-big-bench-audio-dev
```

Wait until the build logs confirm the image was pushed. The default command exits,
so a Space runtime error after a successful build is expected; the Space can be
paused. Jobs use the image without a running Space. See the
[HF image workflow](https://huggingface.co/docs/hub/jobs-images).
Space registry images use the latest build and may expire; rebuild if a Job
cannot find the image.

Launch a smoke test with your namespace and image reference:

```bash
HF_NAMESPACE="your-hf-username"
EVAL_IMAGE="hf.co/spaces/${HF_NAMESPACE}/s2s-big-bench-audio-dev"
EVAL_RESULTS="${HF_NAMESPACE}/s2s-big-bench-audio-results"

hf jobs run --detach --namespace "$HF_NAMESPACE" --flavor a10g-large --timeout 45m --secrets HF_TOKEN \
    -e S2S_LIMIT=4 -e S2S_PUSH_TO_HUB="$EVAL_RESULTS" \
    "$EVAL_IMAGE" vibe-check
```

Set `EVAL_IMAGE` to another registry reference to skip the Space build.
For 40 questions, remove `S2S_LIMIT` and choose an appropriate timeout.
For all 1,000, also add `-e S2S_SUBSET=full`. Questions run sequentially at real
time; a full run can require hours. Reports are written after all questions,
and hard Job timeouts can prevent uploads. Checkpoint/resume is not implemented.
Leave time for downloads and initialization as well as evaluation.

GPU Jobs and hosted inference are billed to your account. The default stack is
Parakeet TDT, `Qwen/Qwen3.5-9B:together` through HF Inference Providers, and
Qwen3-TTS GGML. `HF_TOKEN` needs Jobs, repository-write, and Inference Providers
permissions for this configuration.

## Container settings

| Environment variable | Default | Purpose |
|---|---|---|
| `S2S_SUBSET` | `vibe` | `vibe`, `full`, or a manifest path |
| `S2S_LIMIT` | all selected questions | Truncate the selection |
| `S2S_LABEL` | Job ID, otherwise `local` | Report/log label |
| `S2S_PUSH_TO_HUB` | unset | Private dataset for reports/logs |
| `S2S_COMPARE` | unset | Baseline file or `owner/repo:reports/file.json` |
| `S2S_LLM_MODEL` | `Qwen/Qwen3.5-9B:together` | Hosted model |
| `S2S_LLM_BASE_URL` | `https://router.huggingface.co/v1` | LLM endpoint |
| `S2S_SERVE_ARGS` | default stack above | Replace all server arguments |
| `S2S_EVAL_ARGS` | unset | Extra evaluator arguments |
| `S2S_REPORT_DIR` | `/output` | Local report/log directory |

Uploads create a private dataset if needed and refuse existing public datasets,
including for startup-failure logs; visibility is never changed automatically.
Reports use `reports/<timestamp>-<label>.json`; logs use `logs/<label>.log`.
Use unique labels to avoid replacing logs. Copies remain in `S2S_REPORT_DIR`.

To change a model at the configured endpoint, add
`-e S2S_LLM_MODEL="MODEL_ID:PROVIDER"` to the launch command. To replace the entire
server configuration, include the STT, TTS, model, and endpoint arguments:

```bash
EVAL_SERVER_ARGS="--stt parakeet-tdt --tts qwen3 --qwen3_tts_backend ggml --llm_backend chat-completions"
hf jobs run --detach --namespace "$HF_NAMESPACE" --flavor a10g-large --timeout 90m \
    --secrets HF_TOKEN --secrets OPENAI_API_KEY \
    -e S2S_PUSH_TO_HUB="$EVAL_RESULTS" \
    -e S2S_SERVE_ARGS="$EVAL_SERVER_ARGS --model_name MODEL_ID --responses_api_base_url https://provider.example/v1 --responses_api_reasoning_effort none" \
    "$EVAL_IMAGE" vibe-check
```

Replace the model and endpoint placeholders and export `OPENAI_API_KEY` for that
provider. An explicit key takes precedence over `HF_TOKEN` for LLM authentication;
pass credentials as secrets, never inside arguments. Match reasoning settings
when comparing providers, and choose options supported by your endpoint.

Pass prompt overrides through `S2S_EVAL_ARGS`, using `--instructions 'PROMPT'` or
`--instructions-file /path/to/prompt.txt`. Keep `Final answer: X` for reliable
extraction; the report records the prompt. Argument strings use shell-style
quoting, with no shell execution or variable expansion inside their values.
Model and prompt changes need no rebuild; installing another backend does.

## Reading results

Reports include transcripts, generated audio byte counts, per-category accuracy,
parse rates, errors, runtime versions, and latency. Accuracy grades the assistant
transcript; it does not measure synthesized-speech intelligibility. Nonempty
audio is required for a successful turn.

Time to first audio is measured from the end of the streamed question to receipt
of the first audio delta. Turn latency ends at receipt of `response.done`.
These include the VAD wait, but not question-streaming time or browser playback.
Run duration includes engine startup, audio loading, retries, and shutdown;
it excludes the optional judge. Keep `--speed 1` for realistic latency comparisons.

The rule-based extractor looks for `Final answer: X`, with a fallback scan of the
reply. `--judge-model` uses another model to extract an answer; both verdicts are
retained. A completed judge verdict takes precedence, including `UNPARSED`.
If the judge fails, the rule-based verdict remains in use.

Missing audio, failed/cancelled responses, timeouts, and split turns are errors.
They count against accuracy and cause exit code 1 after the report is written.
Incorrect answers alone do not cause failure. Persistent audio-loading failures
are recorded per item so other questions can continue.

The defaults use a 900 ms VAD silence threshold and append two seconds of silence.
Split-turn errors mean a question produced multiple responses or a new input
item after a response began. Adjust `--silence-ms` and `--trailing-silence-ms`
if necessary. `--response-timeout` defaults to 180 seconds after all question
and trailing-silence audio is sent; it also bounds the separate wait for session
capacity before sending audio.

The 40-question sample is a regression check. Comparisons show observed score and
latency differences without significance claims or an automatic regression gate.
Use matched question sets, judge settings, and engine configurations; pinned
questions do not make hosted models/providers immutable. Inspect individual
failures before claiming an improvement.

## Tests

```bash
python -m pytest tests/evals -q
```

Tests use synthetic metadata and protocol events, without downloading models or
starting paid Jobs. GPU smoke runs are separate integration checks.
