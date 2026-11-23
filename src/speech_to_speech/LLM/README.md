# Language models

The LLM runs locally by default. After [installation](../../../README.md#installation), `speech-to-speech local` picks a model for your platform:

| Platform | Backend | Model | Device |
|---|---|---|---|
| Apple Silicon macOS | `mlx-lm` | `mlx-community/Qwen3.5-4B-4bit` | MLX |
| Linux / Windows | `transformers` | `Qwen/Qwen3.5-4B` | Available accelerator, otherwise CPU (slow) |

Both backends use `LanguageModelHandler` in `language_model.py`. `--llm_backend` selects a backend and its default model; `--model_name` replaces the model. Use MLX conversions with `mlx-lm` and Transformers checkpoints with `transformers`:

```bash
speech-to-speech local --model_name mlx-community/Qwen3.5-9B-4bit   # Apple Silicon
speech-to-speech local --model_name Qwen/Qwen3.5-9B                 # Linux / Windows
```

`--llm_device` changes only the LLM's device (for example `cuda` or `cpu`); `--device` changes every stage. `--llm_torch_dtype` sets the Transformers dtype. `--mac-optimal-settings` is optional on Mac and only sets MPS device defaults.

## Local generation options

Local backends use the `--llm_*` arguments. For example, cap reply length:

```bash
speech-to-speech local --llm_gen_max_new_tokens 128
```

Transformers also accepts `--llm_gen_temperature`, `--llm_gen_do_sample`, and `--llm_gen_min_new_tokens`. The MLX text path forwards the maximum token count but not those sampling settings. Shared conversation settings include `--chat_size`, `--init_chat_prompt`, `--user_role`, and `--stream_batch_sentences`. See the [argument definitions](../arguments_classes/language_model_arguments.py) for defaults.

For image input, select a vision-language model and pass `--llm_is_vlm` to use `VisionLanguageModelHandler`. MLX vision inference needs `mlx-vlm`, included in `speech-to-speech[mlx-lm]`.

## API backends

Use an API backend when the LLM runs in a separate server or at a hosted provider. Neither is the default.

| Backend | Endpoint | Handler |
|---|---|---|
| `responses-api` | `/v1/responses` | `ResponsesApiModelHandler` |
| `chat-completions` | `/v1/chat/completions` | `ChatCompletionsApiModelHandler` |

Both use the `--responses_api_*` connection flags plus the shared settings above; `--llm_*` generation flags do not apply.

A local inference server:

```bash
speech-to-speech local \
  --llm_backend chat-completions \
  --model_name YOUR_SERVED_MODEL \
  --responses_api_base_url http://127.0.0.1:8000/v1 \
  --responses_api_api_key ""
```

A hosted provider, here OpenAI with its default `gpt-5.6-terra`:

```bash
export OPENAI_API_KEY=...
speech-to-speech local --llm_backend responses-api
```

Speech recognition and synthesis stay local either way. With a hosted provider, transcribed text and conversation history are sent to it. See [LLM backends](../../../README.md#llm-backends) for the llama.cpp setup, provider examples, and direct audio input.

## Language behavior

`--enable_lang_prompt` adds a per-turn instruction asking the LLM to reply in the language reported by STT. It is off by default. Parakeet chooses its transcription language automatically; the Whisper backends accept `--language auto` or a fixed code.

`--detect_llm_output_language` separately detects the assistant's output language for TTS. See [Multi-language support](../../../README.md#multi-language-support) for how these options interact.
