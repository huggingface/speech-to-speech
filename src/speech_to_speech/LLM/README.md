# Language models

The pipeline runs a local language model by default. After [installation](../../../README.md#installation), start a conversation without an API key or a separate inference server:

```bash
speech-to-speech local
```

## Local defaults

| Platform | Backend | Model | Device |
|---|---|---|---|
| Apple Silicon macOS | `mlx-lm` | `mlx-community/Qwen3.5-4B-4bit` | MLX on Apple Silicon |
| Linux / Windows | `transformers` | `Qwen/Qwen3.5-4B` | Available accelerator, otherwise CPU |

Parakeet TDT transcribes your speech and Qwen3-TTS speaks the replies. The first run downloads the models and supporting assets; subsequent runs reuse the cache. For disconnected use, see [Offline operation](../../../README.md#offline-operation). CPU generation can be slow; see the [hardware guidance](../../../README.md#start-a-conversation).

Both local backends use `LanguageModelHandler` in `language_model.py`. An explicit `--llm_backend` selects that backend's defaults; `--model_name` overrides its model. Use MLX-converted weights for `mlx-lm` and Transformers-compatible weights for `transformers`.

### Apple Silicon

`local` and `serve` already select MLX LM on macOS. To make the model choice explicit or replace it:

```bash
speech-to-speech local \
  --llm_backend mlx-lm \
  --model_name mlx-community/Qwen3.5-4B-4bit
```

`--mac-optimal-settings` is optional. It also supplies MPS defaults for supported component devices; explicit backend, model, and device settings take precedence.

### Linux / Windows

Transformers selects an available accelerator or CPU automatically. To require an NVIDIA GPU for the LLM:

```bash
speech-to-speech local \
  --llm_backend transformers \
  --model_name Qwen/Qwen3.5-4B \
  --llm_device cuda
```

Omit `--llm_device cuda` to keep automatic selection, or pass `--llm_device cpu` explicitly. `--llm_torch_dtype` controls the Transformers model's dtype. `--device` overrides devices across supported pipeline handlers; `--llm_device` changes only the LLM.

### Local generation options

Local backends use the `--llm_*` arguments. For example, cap reply generation:

```bash
speech-to-speech local --llm_gen_max_new_tokens 128
```

Transformers also accepts `--llm_gen_temperature`, `--llm_gen_do_sample`, and `--llm_gen_min_new_tokens`. The current MLX text generation path forwards the maximum token count, but does not forward those sampling settings. Shared conversation settings include `--chat_size`, `--init_chat_prompt`, `--user_role`, and `--stream_batch_sentences`. See the [argument definitions](../arguments_classes/language_model_arguments.py) for defaults.

The default handler consumes text transcribed by STT. For image inputs, explicitly select a compatible vision-language model and pass `--llm_is_vlm` to use `VisionLanguageModelHandler`. MLX vision inference also needs the `mlx-vlm` dependency included in `speech-to-speech[mlx-lm]`.

## Serve the same local pipeline to a client

`serve` uses the same models as `local`, but waits for a Realtime client instead of opening your microphone and speakers:

```bash
# Terminal 1: start the local pipeline server
speech-to-speech serve

# Terminal 2: connect the packaged audio client
speech-to-speech talk --url ws://127.0.0.1:8765/v1/realtime
```

The server binds to loopback by default. Browser and LiveKit clients can use this server too; see the [browser demo](../../../demo/README.md) and [LiveKit example](../../../demo/livekit/README.md). Its OpenAI-compatible Realtime interface works with local models and does not require an OpenAI provider key.

## Optional API backends

Use an API backend when the LLM runs in a separate local inference server or at a hosted provider. Select it explicitly; neither API backend is the pipeline default.

| Backend | Endpoint | Handler |
|---|---|---|
| `responses-api` | `/v1/responses` | `ResponsesApiModelHandler` |
| `chat-completions` | `/v1/chat/completions` | `ChatCompletionsApiModelHandler` |

Both use `--responses_api_*` connection flags and shared settings such as `--model_name`, `--chat_size`, and `--init_chat_prompt`. Choose the endpoint your server or provider supports; local generation flags do not configure a remote model.

### Separate local LLM server

Start your inference server first, then supply its URL and model name:

```bash
speech-to-speech local \
  --llm_backend chat-completions \
  --model_name YOUR_SERVED_MODEL \
  --responses_api_base_url http://127.0.0.1:8000/v1 \
  --responses_api_api_key ""
```

This keeps Parakeet TDT and Qwen3-TTS in the speech pipeline. The empty key is for an unauthenticated local server; supply credentials if your server requires them. For a complete setup, see [Combining with llama.cpp](../../../README.md#combining-with-llamacpp).

### Hosted LLM

To keep speech local and use OpenAI for reply generation:

```bash
export OPENAI_API_KEY=...
speech-to-speech local --llm_backend responses-api
```

The selected Responses API backend defaults to `gpt-5.6-terra` with reasoning effort `none`. Override the model with `--model_name`; for another provider, also set `--responses_api_base_url` and `--responses_api_api_key`. Transcribed text, instructions, and conversation history are sent to the selected provider, with its connectivity and billing requirements. The local speech models still need to fit in memory.

See [LLM backends](../../../README.md#llm-backends) for provider examples, endpoint differences, and direct audio input. Direct audio requires an explicitly selected audio-capable model and backend; the default local pipeline uses STT.

## Language behavior

`--enable_lang_prompt` adds a per-turn instruction asking the LLM to reply in the language reported by STT. It is off by default. Language detection depends on the selected STT backend: Parakeet chooses its transcription language automatically; the Whisper backends accept `--language auto` or a fixed code.

`--detect_llm_output_language` separately detects the assistant's output language for TTS. See [Multi-language support](../../../README.md#multi-language-support) for coverage and how these options interact.
