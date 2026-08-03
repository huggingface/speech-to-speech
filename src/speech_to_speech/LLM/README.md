# LLM Summary

## Available LLM backends (`--llm_backend`)

Runtime-supported values in `s2s_pipeline.py`:

- `transformers` → `language_model.py` (Transformers backend)
- `mlx-lm` → `language_model.py` (MLX backend)
- `responses-api` → `responses_api_language_model.py`

## Usage

### 1) Transformers (`--llm_backend transformers`)

- Handler: `LanguageModelHandler`
- Typical use: local GPU/CPU inference using Hugging Face Transformers
- Backend-specific args prefix: `--llm_*`
- Shared args (from base): `--model_name`, `--chat_size`, `--init_chat_prompt`, `--enable_lang_prompt`

```bash
python s2s_pipeline.py \
  --llm_backend transformers \
  --model_name Qwen/Qwen3-4B-Instruct-2507 \
  --llm_device cuda \
  --llm_torch_dtype float16 \
  --llm_gen_max_new_tokens 128
```

Common options:
- `--llm_gen_min_new_tokens`
- `--llm_gen_temperature`
- `--llm_gen_do_sample`
- `--chat_size`
- `--init_chat_prompt`

### 2) MLX-LM (`--llm_backend mlx-lm`)

- Handler: `LanguageModelHandler`
- Typical use: Apple Silicon local inference
- Backend-specific args prefix: same as Transformers (`--llm_*`)

```bash
python s2s_pipeline.py \
  --llm_backend mlx-lm \
  --model_name mlx-community/Qwen3-4B-Instruct-2507-bf16 \
  --llm_device mps \
  --llm_gen_max_new_tokens 128
```

Common options:
- `--llm_gen_temperature`
- `--llm_gen_do_sample`
- `--chat_size`
- `--init_chat_prompt`

### 3) OpenAI-compatible API (`--llm_backend responses-api`)

- Handler: `ResponsesApiModelHandler`
- Typical use: remote model serving via OpenAI-compatible endpoints
- Backend-specific args prefix: `--responses_api_*`
- Shared args (from base): `--model_name`, `--chat_size`, `--init_chat_prompt`, `--enable_lang_prompt`

```bash
python s2s_pipeline.py \
  --llm_backend responses-api \
  --model_name gpt-5.4-mini \
  --responses_api_api_key YOUR_API_KEY \
  --responses_api_base_url https://api.example.com/v1 \
  --responses_api_stream true
```

Common options:
- `--chat_size`
- `--init_chat_prompt`
- `--user_role`

## LLM Behavior

When STT is set to language auto-detection (`--language auto`), LLM handlers can receive `(text, language_code)` and prepend a language control instruction like:

- `Please reply to my message in <language>.`

This helps the assistant respond in the detected language. The behavior is opt-in via `--enable_lang_prompt` (shared across all backends); it defaults to `False`.

## Setup

### CUDA setup

```bash
python s2s_pipeline.py \
  --llm_backend transformers \
  --model_name microsoft/Phi-3-mini-4k-instruct
```

### Local Mac setup

```bash
python s2s_pipeline.py \
  --local_mac_optimal_settings \
  --model_name mlx-community/Qwen3-4B-Instruct-2507-bf16
```

`--local_mac_optimal_settings` already sets `--llm_backend mlx-lm` and will default the model to `mlx-community/Qwen3-4B-Instruct-2507-bf16` if not overridden.

### Realtime (OpenAI-compatible) setup

Run the server in realtime mode, then connect with the realtime client:

```bash
# 1. Start the pipeline in realtime mode
python s2s_pipeline.py \
  --mode realtime \
  --llm_backend mlx-lm \
  --model_name mlx-community/Qwen3-4B-Instruct-2507-bf16 \
  --ws_host 0.0.0.0 \
  --ws_port 8765

# 2. Connect with the realtime client
python listen_and_play_realtime.py --host 127.0.0.1 --port 8765
```

Or with `--local_mac_optimal_settings` on Apple Silicon:

```bash
python s2s_pipeline.py \
  --local_mac_optimal_settings \
  --mode realtime \
  --ws_host 0.0.0.0 \
  --ws_port 8765
```

### Remote API setup

```bash
python s2s_pipeline.py \
  --llm_backend responses-api \
  --model_name gpt-5.4-mini \
  --responses_api_api_key YOUR_API_KEY
```
