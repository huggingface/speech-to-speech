# ConvFill

[ConvFill](https://arxiv.org/abs/2511.07397) pairs a small fine-tuned Talker,
which starts answering immediately, with a larger Reasoner that streams
knowledge in parallel. This example registers ConvFill as an opt-in LLM backend:

Microphone → VAD → STT → Gemma Reasoner + ConvFill Talker → TTS → speaker.

Only the Talker's text is spoken. The Reasoner is `nvidia/Gemma-4-31B-IT-NVFP4`
with thinking disabled, and the Talker is the original
[`vysri/Qwen-0.6B-ConvFill`](https://huggingface.co/vysri/Qwen-0.6B-ConvFill)
checkpoint and tokenizer. Nothing changes for pipelines that do not select
`--llm_backend convfill`.

## Serve the models

Run two vLLM servers so both models stay resident and generate concurrently,
for example on separate GPUs (or set `--gpu-memory-utilization` so both fit):

```bash
vllm serve nvidia/Gemma-4-31B-IT-NVFP4 --revision 4135a98a9b728a548947683219633b25682223ac \
  --quantization modelopt --reasoning-parser gemma4 --max-model-len 8192 --generation-config vllm --port 8001
```

```bash
vllm serve vysri/Qwen-0.6B-ConvFill --revision 9195fc16f0ba56743d956f0fab5eed2386665aff \
  --max-model-len 4096 --generation-config vllm --port 8002
```

The Talker receives serialized token IDs on raw `/v1/completions`; no chat
template is applied. Set `CONVFILL_REASONER_URL` and `CONVFILL_TALKER_URL`
(including `/v1`) if the servers are elsewhere, and `CONVFILL_REASONER_API_KEY`
or `CONVFILL_TALKER_API_KEY` if they require a key.

## Run

From the repository root, with Python 3.11 or 3.12:

```bash
uv sync --group dev
uv pip install -r examples/convfill/requirements.txt
```

Start the pipeline with any STT and TTS backends. This uses OpenAI-compatible
speech services:

```bash
PYTHONPATH=examples/convfill:src uv run python -m convfill_demo.pipeline serve \
  --llm_backend convfill \
  --stt openai --openai_stt_base_url "$STT_URL" --openai_stt_model Qwen/Qwen3-ASR-1.7B \
  --tts openai --openai_tts_base_url "$TTS_URL" --openai_tts_model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --openai_tts_voice aiden --openai_tts_stream true \
  --speculative_reopen_ms 500 --smart_turn_max_wait_ms 1700
```

Then open the [web demo](../../demo/README.md) against it:

```bash
npm ci --prefix demo
uv pip install -r demo/requirements.txt
SPEECH_TO_SPEECH_URL=ws://localhost:8765/v1/realtime STARTUP_GREETING= \
  uv run uvicorn --app-dir demo server:app --port 7860
```

`STARTUP_GREETING=` disables the demo's hidden greeting prompt, which ConvFill
would otherwise answer as a user turn. ConvFill ignores session tools, so the
demo's web search and camera tools have no effect; session instructions reach
the Reasoner.

For several concurrent sessions, pass `--num_pipelines N`. Each slot keeps its
own ConvFill conversation and cancellation state. The demo's load-balancer mode
can route sessions to such workers; see the demo README.

## Behavior

- **Fillers.** While knowledge is late, the Talker speaks a filler phrase.
  `max_filler_phrases` in `configs/demo_mode/convfill_hosted_config.json` allows
  one per reply; remove it (or use `null`) for unlimited fillers, as in the
  original ConvFill, or set `0` to disable them. Fillers wait until the estimated
  playback of dispatched audio has finished; there is no client acknowledgement.
- **Echoes.** An exact repeated sentence of three or more words at a Talker
  handoff is dropped before speech. Paraphrases are kept.
- **Interruptions.** As in the core pipeline, a cancelled reply leaves no
  ConvFill history.
- **Holds.** The flags above hold output for 500 ms after a complete Smart Turn
  decision and 1700 ms after an incomplete one, 300 ms below the defaults.

## Tests

```bash
PYTHONPATH=examples/convfill:src uv run pytest examples/convfill/convfill_demo/tests -q
```

The runtime is adapted from the MIT-licensed
[ConvFill implementation](https://github.com/vysri/conversational-infill)
(copyright 2026 Vidya Srinivas; see `LICENSE.convfill`). It is copied here
because the upstream package installs as a top-level `src` package and requires
Python older than 3.12.
