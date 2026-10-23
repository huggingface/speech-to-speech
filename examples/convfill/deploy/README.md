# Deploy the hosted ConvFill example

Only Talker text goes to TTS. The browser sends audio through VAD, STT,
concurrent Reasoner/Talker generation, and TTS. One session is admitted at a time.
The example has no camera, browser tools, RAG or MCP integration.

## Model serving

Run two independent vLLM instances so the Reasoner and Talker stay resident
and can generate concurrently. The example uses `nvidia/Gemma-4-31B-IT-NVFP4`
for the Reasoner and `vysri/Qwen-0.6B-ConvFill` for the Talker. Public model
revisions and the vLLM container digest are pinned in `compose.yaml`.

The original fine-tuned Talker tokenizer is required. Its special IDs are
`<|im_start|>=151644`, `<|im_end|>=151645`, and `<sil>=151669`. The client sends
serialized prompt token IDs to raw `/v1/completions`; another chat template
must not be applied. The existing Transformers/MLX inference paths are retained.

The sample Compose configuration places the engines on separate GPUs. It sets
an 8192-token Reasoner context, four active sequences and a 90% memory budget;
the Talker gets 4096 tokens, two sequences and a 35% budget. These are example
budgets, not a description of an existing deployment. Check actual memory with
`nvidia-smi` on the hosts. STT and TTS need separate capacity if colocated.

The speech stages accept operator-configured OpenAI-compatible HTTP services.
The sample model names are the public Qwen3-ASR and Qwen3-TTS checkpoints.
Set all service addresses and credentials in your own environment.

## Configuration

The JSON files under `configs/` set models, generation settings and timeouts.
Environment overrides are listed in `.env.example`. Reasoner thinking defaults
to disabled. Reasoning fields and serialized control tokens are filtered before
knowledge sentences reach the Talker.

Reasoner connect/read/total deadlines are 10/15/45 seconds; Talker deadlines
are 10/15/30 seconds. The whole turn is bounded to 90 seconds with a 3-second
worker shutdown limit. Cancellation closes active streams. Errors roll back
history; a worker that cannot stop quarantines its pipeline.

## Run locally

From the repository root, use Python 3.11/3.12 and Node 24:

```bash
uv sync --group dev
uv pip install -r examples/convfill/requirements.txt
npm ci --prefix demo
uv run python -m nltk.downloader -d .venv/share/nltk_data punkt_tab averaged_perceptron_tagger_eng
# Export your service URLs and keys; see examples/convfill/deploy/.env.example.
bash examples/convfill/run.sh
```

Open http://localhost:7860 and grant microphone access. Ctrl+C stops both
processes. Run the commands below from `examples/convfill` with the environment
activated unless indicated otherwise.

## Containers and HTTPS

```bash
cp deploy/.env.example deploy/.env
# Fill your endpoint URLs/keys; install NVIDIA Container Toolkit on the GPU host.
docker compose --env-file deploy/.env -f deploy/compose.yaml up --build -d
docker compose --env-file deploy/.env -f deploy/compose.yaml logs -f demo
docker compose --env-file deploy/.env -f deploy/compose.yaml down
```

Inference ports stay on the private container network. Configure `DEMO_HOST`
and use `deploy/Caddyfile` for HTTPS and secure WebSockets. Put the demo behind
your own access control. Use a single gateway worker: admission is process-local.
Trust forwarded headers only from your ingress.

To upload to a private Space you own, explicitly select its destination:

```bash
python deploy/hf_space.py --repo YOUR_ORG/YOUR_PRIVATE_SPACE
python scripts/validate_browser_access.py --repo YOUR_ORG/YOUR_PRIVATE_SPACE
```

The upload requires `HF_TOKEN` for the Hub API and the four `CONVFILL_*_URL`
settings. Supply endpoint credentials separately using `CONVFILL_*_API_KEY`;
the Hub upload token is never used as an inference credential implicitly.
The uploader requires a private destination. No deployment names or live service
addresses are included in this example. Manage its lifecycle in your own account.

## Health and validation

`/healthz` reports gateway liveness; `/readyz` checks the initialized pipeline
and both model services. vLLM exposes `/health` and `/v1/models`.

```bash
PYTHONPATH=.:../../src python -m pytest convfill_demo/tests -q
npm --prefix ../../demo run test:ui
PYTHONPATH=.:../../src python scripts/validate_models.py
python scripts/validate_realtime.py --audio /path/to/generated-test.wav
python scripts/validate_bargein.py --audio /path/to/generated-test.wav
```

The inference/speech scripts require your configured live services. They use
synthetic prompts and generated audio. Keep results and deployment telemetry
private unless separately cleared for publication. `VALIDATION.md` records
source-level checks without publishing deployment topology or telemetry.

For local fault injection only, use `scripts.fault_proxy:app` on loopback with
`CONVFILL_REAL_REASONER_URL` and `CONVFILL_REASONER_API_KEY`, then run
`PYTHONPATH=.:../../src python scripts/validate_faults.py`. Never expose this
fault-injection helper publicly.

Full prompts/transcripts are not logged by default. `CONVFILL_LOG_CONTENT=1`
explicitly enables content diagnostics. The delivery ledger observes dispatched
text/audio and estimates playback from PCM duration; it has no playback
acknowledgements. Interrupted history is a best-effort dispatched prefix, not
an exact record of what a listener heard.
