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
revision, and prints your launch command. Wait for the Space build to finish.
The Space stays on CPU and only runs a small HTTP server; GPU inference happens
in Jobs. This follows the [HF Docker Space image workflow](https://huggingface.co/docs/hub/jobs-images).

For this development setup:

```bash
hf jobs run --namespace Steveeeeeeen --flavor l4x1 --timeout 45m --secrets HF_TOKEN \
    -e S2S_LIMIT=4 \
    -e S2S_PUSH_TO_HUB=Steveeeeeeen/s2s-big-bench-audio-results \
    hf.co/spaces/Steveeeeeeen/s2s-big-bench-audio-dev vibe-check
```

Remove `S2S_LIMIT` and use `--timeout 90m` for all 40 questions. Hardware and
Inference Providers are billed to the personal account. Only launch the jobs
you need; no recurring jobs or automatic GPU CI are configured.

The default stack is Parakeet TDT, `Qwen/Qwen3.5-9B:together` through HF
Inference Providers (Chat Completions), and Qwen3-TTS with its Torch backend.
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
| `S2S_SERVE_ARGS` | default stack above | Replace server arguments (shell-style quoting, no expansion) |
| `S2S_EVAL_ARGS` | unset | Extra evaluation arguments, e.g. `--silence-ms 1200` |
| `S2S_REPORT_DIR` | `/output` | Local report/log directory |

`Dockerfile.eval` installs the engine from the uploaded source. Dependencies are
resolved at build time and runtime package versions are recorded in reports.
Optional build arguments: `EXTRAS="kokoro supertonic"` installs extra backends;
`PREFETCH=1` warms the default speech-model/audio caches. Prefetch is a download
optimization, not a model-revision pin. The Space upload script creates the
`source-revision.txt` required by the Dockerfile.

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
using a second model; reports keep both verdicts.

Errors (including missing audio, failed/cancelled responses, timeouts, and split
turns) count against accuracy and make the process exit 1 **after** writing the
report. Incorrect reasoning remains a measured result rather than an engine
failure. `--fail-on-regression` additionally exits 1 for a large accuracy drop.

The runner waits for `session.updated` before sending audio, raises the session
VAD silence threshold to 900 ms, and appends two seconds of silence to finish the
turn. If pauses split a recording into multiple responses, the last transcript
is retained for diagnosis but the item is an error. Increase `--silence-ms` and
`--trailing-silence-ms` together if needed. `--speed 1` uses real-time pacing;
faster playback changes queueing and invalidates realistic latency comparisons.

Forty questions is a regression sample, not a leaderboard benchmark. Comparisons
use an approximate independent-proportions uncertainty check; the actual samples
are paired, so interpret it only as a coarse diagnostic. Fewer than ten questions
are marked underpowered and cannot trigger significance. Comparisons warn about
different question sets, dataset revisions, or judge usage. Use matched settings
and examine individual failures before claiming a quality improvement.

## Tests

```bash
python -m pytest tests/evals -q
```

Unit tests use synthetic protocol events and do not download models or start
paid jobs. The real GPU smoke run is a separate integration check.
