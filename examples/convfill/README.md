# Hosted ConvFill speech-to-speech

Browser microphone → VAD → STT → concurrent Gemma Reasoner / ConvFill Talker → TTS → browser.
Only the fine-tuned Talker is spoken. This opt-in example retains ConvFill’s filler scheduler,
conversation state and cancellation while using this checkout’s realtime pipeline and `demo/` UI.
No private repository checkout or commercial model key is required. Normal conversation only;
RAG, MCP, browser tools and camera are disabled.

## Run from this checkout

Use Python 3.11/3.12 and Node 24. From the repository root:

```bash
uv sync --group dev
uv pip install -r examples/convfill/requirements.txt
npm ci --prefix demo
uv run python -m nltk.downloader -d .venv/share/nltk_data punkt_tab averaged_perceptron_tagger_eng
# Export the four endpoint URLs and server-side keys in deploy/.env.example.
bash examples/convfill/run.sh
```

Open http://localhost:7860 and allow microphone access. Ctrl+C stops both processes.
The Reasoner defaults to `nvidia/Gemma-4-31B-IT-NVFP4`, with thinking disabled.
The Talker uses the original `vysri/Qwen-0.6B-ConvFill` checkpoint and tokenizer.
Two vLLM instances keep both resident and generating concurrently. Deployment instructions,
explicit context/memory budgets and recorded validation follow in `deploy/`.

## Tests

```bash
PYTHONPATH=examples/convfill:src uv run pytest examples/convfill/convfill_demo/tests -q
npm --prefix demo run test:ui
```

The runtime and adapter are adapted from the MIT-licensed ConvFill inference implementation
(copyright 2026 Vidya Srinivas; see `LICENSE.convfill`). They are kept in this example rather than
importing a private repository. This example reuses the current browser sources; only hosted normal-mode Reasoner support is included.
