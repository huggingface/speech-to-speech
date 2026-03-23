# LLM Summary

## Available LLM modes (`--llm`)

Runtime-supported values in `s2s_pipeline.py`:

- `transformers` → `language_model.py`
- `mlx-lm` → `mlx_language_model.py`
- `open_api` → `openai_api_language_model.py`

## Usage

### 1) Transformers (`--llm transformers`)

- Handler: `LanguageModelHandler`
- Typical use: local GPU/CPU inference using Hugging Face Transformers
- Primary args prefix: `--lm_*`

```bash
python s2s_pipeline.py \
  --llm transformers \
  --lm_model_name Qwen/Qwen3-4B-Instruct-2507 \
  --lm_device cuda \
  --lm_torch_dtype float16 \
  --lm_gen_max_new_tokens 128
```

Common options:
- `--lm_gen_min_new_tokens`
- `--lm_gen_temperature`
- `--lm_gen_do_sample`
- `--lm_chat_size`
- `--init_chat_prompt`

### 2) MLX-LM (`--llm mlx-lm`)

- Handler: `LanguageModelHandler`
- Typical use: Apple Silicon local inference
- Primary args prefix: same as Transformers (`--lm_*`)

```bash
python s2s_pipeline.py \
  --llm mlx-lm \
  --lm_model_name mlx-community/Qwen3-4B-Instruct-2507-bf16 \
  --lm_device mps \
  --lm_gen_max_new_tokens 128
```

Common options:
- `--lm_gen_temperature`
- `--lm_gen_do_sample`
- `--lm_chat_size`
- `--lm_init_chat_prompt`

### 3) OpenAI-compatible API (`--llm open_api`)

- Handler: `OpenApiModelHandler`
- Typical use: remote model serving via OpenAI-compatible endpoints
- Primary args prefix: `--open_api_*`

```bash
python s2s_pipeline.py \
  --llm open_api \
  --open_api_model_name deepseek-chat \
  --open_api_api_key YOUR_API_KEY \
  --open_api_base_url https://api.example.com/v1 \
  --open_api_stream true
```

Common options:
- `--open_api_chat_size`
- `--open_api_init_chat_prompt`
- `--open_api_user_role`

## LLM Behavior

When STT is set to language auto-detection (`--language auto`), LLM handlers can receive `(text, language_code)` and prepend a language control instruction like:

- `Please reply to my message in <language>.`

This helps the assistant respond in the detected language.

## Setup

### CUDA setup

```bash
python s2s_pipeline.py \
  --llm transformers \
  --lm_model_name microsoft/Phi-3-mini-4k-instruct
```

### Local Mac setup

```bash
python s2s_pipeline.py \
  --local_mac_optimal_settings \
  --llm mlx-lm
```

### Remote API setup

```bash
python s2s_pipeline.py \
  --llm open_api \
  --open_api_model_name deepseek-chat \
  --open_api_api_key YOUR_API_KEY
```
