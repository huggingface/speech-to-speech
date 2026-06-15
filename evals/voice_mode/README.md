# Voice Mode Evals

These are manual model-backed checks for the default voice-mode system prompt
and the local tool-call parser. They are intentionally outside `tests/` so
normal `pytest` runs do not make network calls or spend API tokens.

Run all checks from the repository root:

```bash
uv run python evals/voice_mode/run_voice_mode_eval.py --model gpt-5.4-mini
```

The runner uses `OPENAI_API_KEY` by default. To use another OpenAI-compatible
Responses API endpoint:

```bash
uv run python evals/voice_mode/run_voice_mode_eval.py \
  --model Qwen/Qwen3.5-27B-FP8 \
  --base-url https://example.test/v1 \
  --api-key-env HF_TOKEN
```

Some OpenAI-compatible providers work better through Chat Completions. For
example, Hugging Face Inference Providers with Cerebras GLM 4.7 can use the
default no-reasoning flag there:

```bash
uv run python evals/voice_mode/run_voice_mode_eval.py \
  --api chat-completions \
  --base-url https://router.huggingface.co/v1 \
  --api-key-env HF_TOKEN \
  --model zai-org/GLM-4.7:cerebras
```

Write a machine-readable report for before/after prompt comparisons:

```bash
uv run python evals/voice_mode/run_voice_mode_eval.py \
  --model gpt-5.4-mini \
  --json-output evals/voice_mode/results/latest.json
```

The suite has two layers:

- Parser fixtures run offline and exercise `extract_function_calls_from_text`
  plus schema validation through `FunctionToolCall.to_realtime_function_tool_call`.
- Model cases call the Responses API with the current voice prompt. They check
  that spoken text is brief, speech-friendly, free of markdown/action text, and
  that local tool calls appear as exactly one parseable `<code>...</code>` block
  when a tool is appropriate.

The default run reports 20 results: 4 parser fixtures and 16 model cases.
Model runs also report aggregate model latency: total, mean, and case count.
Model calls default to `--reasoning-effort none` for lower latency. Use
`--reasoning-effort omit` for providers or models that reject reasoning
controls.

Use `--parser-only` when changing only parser code, or `--case NAME` to run a
single model case while iterating on prompts.

Use `--include-stress` for harder checks, such as a user explicitly asking for
bullets or a physical expression request that weaker models may answer without
emitting the expected local tool block. Stress cases are useful while hardening
prompts, but are kept out of the default run so the baseline suite tracks
ordinary voice-mode behavior.
