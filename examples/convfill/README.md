# Hosted ConvFill speech-to-speech

Browser microphone → VAD → STT → concurrent Gemma Reasoner / ConvFill Talker → TTS → browser.
Only the fine-tuned Talker is spoken. This opt-in example retains ConvFill’s filler scheduler,
conversation state and cancellation while using this checkout’s realtime pipeline and `demo/` UI.
No private repository checkout or commercial model key is required. The Reasoner can call the
UI’s web search and opt-in camera tools. RAG, MCP, and browser automation remain disabled.

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

## Greeting, tools, settings, and timings

The demo sends a short greeting after connection. Set `STARTUP_GREETING` to customize
its instruction, or set it to an empty string to disable it. The gear menu uses the current
shared UI, including voice, instructions, audio buffering, noise gate, and connection settings.
This gateway uses WebSocket transport; WebRTC remains unavailable here.

Web search uses `SERPER_API_KEY` on the UI server. Only its availability is returned to the
browser. Alternatively set `CONVFILL_SEARCH_URL` and `CONVFILL_SEARCH_API_KEY` to reuse
an existing server-side search proxy whose Serper secret cannot be exported. That URL and
credential stay on the UI server. Without either option, users can provide a key in the tools dialog. The camera is opt-in
and requires a multimodal Reasoner. vLLM must enable native tool parsing (the Compose example
uses `--enable-auto-tool-choice --tool-call-parser gemma4`). Tool arguments, results, and image
inputs go to the Reasoner; only the Talker’s text reaches TTS. Session instructions also reach
the Reasoner. Tool-only planning turns do not duplicate the user’s ConvFill conversation turn.

History shows server stage timings after each response. Server E2E ends at first TTS audio,
not at the end of generation. WebSocket playback additionally shows a compact **First audio** timing disclosure as soon
as the first non-silent output frame is rendered, including network and playback buffering.
This uses the browser output-device clock and microphone chunk timestamps: it is an estimate,
not a physical speaker measurement. Unsupported output clocks, greetings without a speech turn,
and cancelled playback do not fabricate a measurement. Tool follow-ups share the original
speech anchor and report only its first audio. Audio dispatch is still not proof of what the
user heard; this timing does not change interrupted-history guarantees.

The Talker may acknowledge a question while the Reasoner is still working; the
Reasoner supplies new substantive content, not a second conversational preamble.
Native tool follow-ups retain the current Talker prefix, and receive a one-second
knowledge head start before filler scheduling resumes. Exact repeated sentence
prefixes (three or more words) are suppressed before TTS and state updates. This
guard does not remove paraphrases, change facts, or deduplicate the whole history.
Interrupted-history text remains an estimate, not proof of physical playback.
