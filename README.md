<div align="center">
  <div>&nbsp;</div>
  <img src="https://raw.githubusercontent.com/huggingface/speech-to-speech/main/logo.png" width="600"/>
</div>

# Speech To Speech: Build local voice agents with open-source models

## 📖 Quick Index
* [Approach](#approach)
  - [Structure](#structure)
  - [Modularity](#modularity)
* [Setup](#setup)
* [Usage](#usage)
  - [Realtime approach](#realtime-approach)
  - [Server/Client approach](#serverclient-approach)
  - [WebSocket approach](#websocket-approach)
  - [Local approach](#local-approach-running-on-mac)
  - [LLM Backend](#llm-backend)
  - [Realtime mode](#realtime-mode)
  - [Docker Server approach](#docker-server)
* [Command-line usage](#command-line-usage)
  - [Model parameters](#model-parameters)
  - [Generation parameters](#generation-parameters)
  - [Notable parameters](#notable-parameters)

## Approach

### Structure
This repository implements a speech-to-speech cascaded pipeline consisting of the following parts:
1. **Voice Activity Detection (VAD)**
2. **Speech to Text (STT)**
3. **Language Model (LM)**
4. **Text to Speech (TTS)**

### Modularity
The pipeline provides a fully open and modular approach, with a focus on leveraging models available through the Transformers library on the Hugging Face hub. The code is designed for easy modification, and we already support device-specific and external library implementations:

**VAD** 
- [Silero VAD v5](https://github.com/snakers4/silero-vad)

**STT**
- Any [Whisper](https://huggingface.co/docs/transformers/en/model_doc/whisper) model checkpoint on the Hugging Face Hub through Transformers 🤗, including [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) and [distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3)
- [Lightning Whisper MLX](https://github.com/mustafaaljadery/lightning-whisper-mlx?tab=readme-ov-file#lightning-whisper-mlx)
- [MLX Audio Whisper](https://github.com/huggingface/mlx-audio) - Fast Whisper inference on Apple Silicon
- [Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-1.1b) - Real-time streaming STT with sub-100ms latency on Apple Silicon (CUDA/CPU via nano-parakeet, no NeMo)
- [Paraformer - FunASR](https://github.com/modelscope/FunASR)

**LLM**
- Any instruction-following model on the [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending) via Transformers 🤗
- [mlx-lm](https://github.com/ml-explore/mlx-examples/blob/main/llms/README.md)
- [OpenAI API](https://platform.openai.com/docs/quickstart)

**TTS**
- [ChatTTS](https://github.com/2noise/ChatTTS?tab=readme-ov-file)
- [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) - Streaming TTS with voice cloning from Kyutai Labs
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) - Fast and high-quality TTS optimized for Apple Silicon
- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)

## Setup

Install the default package from PyPI:
```bash
pip install speech-to-speech
```

The default install is scoped to the standard realtime voice-agent path:
- Parakeet TDT for STT
- OpenAI-compatible API for the language model
- Qwen3-TTS for speech output
- local audio and realtime server modes

Optional backends are installed with extras:
```bash
pip install "speech-to-speech[kokoro]"
pip install "speech-to-speech[pocket]"
pip install "speech-to-speech[faster-whisper]"
pip install "speech-to-speech[paraformer]"
pip install "speech-to-speech[mlx-lm]"
pip install "speech-to-speech[websocket]"
```

Deprecated model implementations, including MeloTTS, live in [`archive/`](./archive) and are no longer wired into the CLI.

For development from source:
```bash
git clone https://github.com/huggingface/speech-to-speech.git
cd speech-to-speech
uv sync
```

This installs the `speech_to_speech` package in editable mode and makes the `speech-to-speech` CLI command available. The project uses a single `pyproject.toml` with platform markers, so macOS and non-macOS dependencies are resolved automatically from one file.

**Note on DeepFilterNet:** DeepFilterNet (used for optional audio enhancement in VAD) requires `numpy<2` and conflicts with Pocket TTS, which requires `numpy>=2`. Install DeepFilterNet manually only in environments where you are not using Pocket TTS.


## Usage

The default CLI is equivalent to a realtime Parakeet + OpenAI-compatible LLM + Qwen3-TTS setup. It uses `OPENAI_API_KEY` from the environment unless `--responses_api_api_key` is provided:
```bash
speech-to-speech
```

The pipeline can be run in four ways:
- **Realtime approach**: Models run locally or on a server, and an OpenAI Realtime-compatible WebSocket API is exposed for another app.
- **Server/Client approach**: Models run on a server, and audio input/output are streamed from a client using TCP sockets.
- **WebSocket approach**: Models run on a server, and audio input/output are streamed from a client using WebSockets.
- **Local approach**: Runs locally.

### Recommended setup 

### Realtime Approach

The default realtime setup uses `--llm_backend responses-api`, which works with any provider supporting the OpenAI Responses API protocol. Export `OPENAI_API_KEY` with your provider's key before launching, or pass it explicitly with `--responses_api_api_key`. For a non-OpenAI provider, also set `--responses_api_base_url`.

```bash
export OPENAI_API_KEY=...
```

The default mode starts the OpenAI Realtime-compatible server:
```bash
speech-to-speech
```

This is equivalent to:
```bash
speech-to-speech \
    --thresh 0.6 \
    --stt parakeet-tdt \
    --llm_backend responses-api \
    --tts qwen3 \
    --qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --qwen3_tts_speaker Aiden \
    --qwen3_tts_language auto \
    --qwen3_tts_non_streaming_mode True \
    --qwen3_tts_mlx_quantization 6bit \
    --model_name gpt-5.4-mini \
    --chat_size 30 \
    --responses_api_stream \
    --enable_live_transcription \
    --mode realtime
```

### Server/Client Approach

1. Run the pipeline on the server:
   ```bash
   speech-to-speech --recv_host 0.0.0.0 --send_host 0.0.0.0
   ```

2. Run the client locally to handle microphone input and receive generated audio:
   ```bash
   python scripts/listen_and_play.py --host <IP address of your server>
   ```

### WebSocket Approach

1. Run the pipeline with WebSocket mode:
   ```bash
   speech-to-speech --mode websocket --ws_host 0.0.0.0 --ws_port 8765
   ```

2. Connect to the WebSocket server from your client application at `ws://<server-ip>:8765`. The server handles bidirectional audio streaming:
   - Send raw audio bytes to the server (16kHz, int16, mono)
   - Receive generated audio bytes from the server

### Local Approach (Mac)

1. For optimal settings on Mac:
   ```bash
   speech-to-speech --local_mac_optimal_settings
   ```

   You can also specify a particular LLM model:
   ```bash
   speech-to-speech \
       --local_mac_optimal_settings \
       --model_name mlx-community/Qwen3-4B-Instruct-2507-bf16
   ```

This setting:
   - Adds `--device mps` to use MPS for all models.
   - Sets [Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-1.1b) for STT (fast streaming ASR on Apple Silicon)
   - Sets MLX LM as the LLM backend
   - Sets Qwen3-TTS for TTS
   - `--tts pocket` and `--tts kokoro` are also valid TTS options on macOS.
   - Qwen3 on Apple Silicon uses `mlx-audio` and defaults to the `6bit` MLX variant unless you explicitly select a different quantization or model suffix.
   - To compare the MLX variants locally, run:
     ```bash
     python scripts/benchmark_tts.py \
         --handlers qwen3 \
         --iterations 3 \
         --qwen3_mlx_quantizations bf16 4bit 6bit 8bit
     ```


### Realtime mode

Realtime mode (`--mode realtime`) streams audio over a WebSocket using the OpenAI Realtime protocol, with live transcription and low-latency turn-taking. The server exposes a WebSocket endpoint at `/v1/realtime` that any OpenAI Realtime-compatible client can connect to.

#### Connecting with the OpenAI Realtime client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8765/v1", api_key="not-needed")

with client.beta.realtime.connect(model="model_name") as conn:
    conn.session.update(
      session={
        "instructions": "You are a helpful assistant.",
        "turn_detection": {"type": "server_vad", "interrupt_response": True},
      }
    )

    # send audio, receive events, etc.
    for event in conn:
        print(event.type)
```

#### Supported events

**Client -> Server**

| Event | Description |
|---|---|
| `input_audio_buffer.append` | Stream base64 PCM audio. Decoded, resampled to 16 kHz, and chunked for the VAD. |
| `session.update` | Deep-merge session config (instructions, tools, voice, turn detection, audio format). |
| `conversation.item.create` | Inject `input_text` or `function_call_output` into the LLM context without triggering generation. |
| `response.create` | Trigger LLM generation. Supports per-response `instructions` and `tool_choice` overrides. |
| `response.cancel` | Cancel the in-progress response and re-enable listening. |

**Server -> Client**

| Event | Description |
|---|---|
| `session.created` | Sent on connection with current session config. |
| `error` | Protocol errors (`session_limit_reached`, `unknown_or_invalid_event`, `invalid_session_type`, `conversation_already_has_active_response`, etc.) |
| `input_audio_buffer.speech_started` | VAD detected user speech. |
| `input_audio_buffer.speech_stopped` | End of user speech segment. |
| `conversation.item.created` | Acknowledges injected `input_text` from `conversation.item.create`. |
| `conversation.item.input_audio_transcription.delta` | Streaming partial transcript (when live transcription is enabled). |
| `conversation.item.input_audio_transcription.completed` | Final transcript for the user turn (with duration usage). |
| `response.created` | Emitted on the first outbound audio chunk (response is `in_progress`). |
| `response.output_audio.delta` | Base64 PCM audio chunk from TTS. |
| `response.output_audio.done` | Audio stream complete for the current output item. |
| `response.output_audio_transcript.done` | Full assistant text transcript for the turn. |
| `response.function_call_arguments.done` | Tool call with `call_id`, `name`, and JSON `arguments`. |
| `response.done` | Response finished (`completed`, `cancelled` with reason `turn_detected` or `client_cancelled`). |

For the full architecture and design details, see the [Realtime Engine README](./src/speech_to_speech/api/openai_realtime/README.md).

### LLM Backend

The LLM is the most compute-intensive and highest-latency component in the pipeline. A single forward pass through a large model can easily dominate the end-to-end response time, so choosing the right backend for your hardware and latency budget matters. To give users the most flexibility, we support the full spectrum of inference solutions:

- **Local inference** — `transformers` (CUDA / CPU) and `mlx-lm` (Apple Silicon) run the model entirely on your machine with no external dependency.
- **Self-hosted servers** — `--llm_backend responses-api` can point at a local [vLLM](https://github.com/vllm-project/vllm) or [llama.cpp](https://github.com/ggerganov/llama.cpp) server, giving you control over quantization, batching, and hardware while keeping traffic on-premise.
- **Provider APIs** — the same `responses-api` backend works with OpenAI, [HuggingFace Inference Providers](https://huggingface.co/inference-providers), [OpenRouter](https://openrouter.ai), and any other provider that implements the OpenAI Responses API.

Select a backend with `--llm_backend` (`responses-api` by default) and pair it with `--model_name`. Backend-specific options (`--responses_api_base_url`, `--responses_api_api_key`, `--responses_api_stream`, etc.) are only needed for the `responses-api` backend.

> The examples below pair Parakeet TDT (local STT) and Qwen3-TTS (local TTS) with different LLM backends.

#### OpenAI-compatible backends (`--llm_backend responses-api`)

`--llm_backend responses-api` works with any server that implements the OpenAI Chat Completions API — point `--responses_api_base_url` at the right endpoint and set `--model_name` accordingly:

| Backend | `--responses_api_base_url` | `--responses_api_api_key` |
|---|---|---|
| OpenAI | *(omit, uses OpenAI default)* | `$OPENAI_API_KEY` |
| HF Inference Providers | `https://router.huggingface.co/v1` | `$HF_TOKEN` |
| OpenRouter | `https://openrouter.ai/api/v1` | `$OPENROUTER_API_KEY` |
| Astraflow (global) | `https://api-us-ca.umodelverse.ai/v1` | `$ASTRAFLOW_API_KEY` |
| Astraflow (China) | `https://api.modelverse.cn/v1` | `$ASTRAFLOW_CN_API_KEY` |
| vLLM (local) | `http://localhost:8000/v1` | *(omit or any string)* |
| llama.cpp (local) | `http://localhost:8080/v1` | *(omit or any string)* |

```bash
# OpenAI
speech-to-speech \
    --mode local \
    --stt parakeet-tdt \
    --llm_backend responses-api \
    --tts qwen3 \
    --qwen3_tts_mlx_quantization 6bit \
    --model_name "gpt-4o-mini" \
    --responses_api_api_key "$OPENAI_API_KEY" \
    --responses_api_stream \
    --enable_live_transcription
```

```bash
# HF Inference Providers — Qwen3.5-9B via Together
speech-to-speech \
    --mode local \
    --stt parakeet-tdt \
    --llm_backend responses-api \
    --tts qwen3 \
    --qwen3_tts_mlx_quantization 6bit \
    --model_name "Qwen/Qwen3.5-9B:together" \
    --responses_api_base_url "https://router.huggingface.co/v1" \
    --responses_api_api_key "$HF_TOKEN" \
    --responses_api_stream \
    --enable_live_transcription
```

```bash
# HF Inference Providers — GPT-oss-20B via Groq
speech-to-speech \
    --stt parakeet-tdt \
    --llm_backend responses-api \
    --tts qwen3 \
    --qwen3_tts_mlx_quantization 6bit \
    --model_name "openai/gpt-oss-20b:groq" \
    --responses_api_base_url "https://router.huggingface.co/v1" \
    --responses_api_api_key "$HF_TOKEN" \
    --responses_api_stream \
    --enable_live_transcription
```

```bash
# Astraflow (global) — OpenAI-compatible platform supporting 200+ models
# Sign up at https://astraflow.ucloud-global.com
speech-to-speech \
    --mode local \
    --stt parakeet-tdt \
    --llm_backend responses-api \
    --tts qwen3 \
    --qwen3_tts_mlx_quantization 6bit \
    --model_name "<model-name>" \
    --responses_api_base_url "https://api-us-ca.umodelverse.ai/v1" \
    --responses_api_api_key "$ASTRAFLOW_API_KEY" \
    --responses_api_stream \
    --enable_live_transcription
```

```bash
# Astraflow (China) — OpenAI-compatible platform supporting 200+ models
# Sign up at https://astraflow.ucloud.cn
speech-to-speech \
    --mode local \
    --stt parakeet-tdt \
    --llm_backend responses-api \
    --tts qwen3 \
    --qwen3_tts_mlx_quantization 6bit \
    --model_name "<model-name>" \
    --responses_api_base_url "https://api.modelverse.cn/v1" \
    --responses_api_api_key "$ASTRAFLOW_CN_API_KEY" \
    --responses_api_stream \
    --enable_live_transcription
```

#### Fully local (Apple Silicon)

```bash
# MLX backend (Apple Silicon)
speech-to-speech \
    --mode local \
    --stt parakeet-tdt \
    --llm_backend mlx-lm \
    --tts qwen3 \
    --qwen3_tts_mlx_quantization 6bit \
    --model_name "mlx-community/Qwen3-4B-Instruct-2507-bf16" \
    --enable_live_transcription
```

```bash
# Transformers backend
speech-to-speech \
    --mode local \
    --stt parakeet-tdt \
    --llm_backend transformers \
    --tts qwen3 \
    --model_name "Qwen/Qwen3-4B-Instruct-2507" \
    --enable_live_transcription
```

### Docker Server

#### Install the NVIDIA Container Toolkit

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

#### Start the docker container
```docker compose up```



### Recommended usage with Cuda

Leverage Torch Compile for Whisper with Pocket TTS for a simple low-latency setup:

```bash
speech-to-speech \
    --stt parakeet-tdt \
    --llm_backend transformers \
    --tts qwen3 \
    --model_name "Qwen/Qwen3-4B-Instruct-2507" \
    --enable_live_transcription
```

### Multi-language Support

The pipeline currently supports English, French, Spanish, Chinese, Japanese, and Korean.  
Two use cases are considered:

- **Single-language conversation**: Enforce the language setting using the `--language` flag, specifying the target language code (default is 'en').
- **Language switching**: Set `--language` to 'auto'. The STT detects the language of each spoken prompt and forwards it to the LLM. Optionally, opt in with `--enable_lang_prompt` to also append a "`Please reply to my message in ...`" instruction so the LLM replies in the detected language. This flag defaults to `False` — large LLMs usually pick up the language from context on their own, but the explicit instruction can help smaller models stay in the right language.

Please note that you must use STT and LLM checkpoints compatible with the target language(s). For multilingual TTS, use ChatTTS or another backend that supports the target language.

#### With the server version:

For automatic language detection:

```bash
speech-to-speech \
    --stt parakeet-tdt \
    --language auto \
    --llm_backend mlx-lm \
    --model_name "mlx-community/Qwen3-4B-Instruct-2507-bf16"
```

Or for one language in particular, chinese in this example

```bash
speech-to-speech \
    --stt whisper-mlx \
    --stt_model_name large-v3 \
    --language zh \
    --llm_backend mlx-lm \
    --model_name mlx-community/Qwen3-4B-Instruct-2507-bf16
```

#### Local Mac Setup

For automatic language detection (note: `--stt whisper-mlx` overrides the default parakeet-tdt from optimal settings, since Whisper `large-v3` has broader language coverage):

```bash
speech-to-speech \
    --local_mac_optimal_settings \
    --stt parakeet-tdt \
    --language auto \
    --model_name mlx-community/Qwen3-4B-Instruct-2507-bf16
```

Or for one language in particular, chinese in this example

```bash
speech-to-speech \
    --local_mac_optimal_settings \
    --stt whisper-mlx \
    --stt_model_name large-v3 \
    --language zh \
    --model_name mlx-community/Qwen3-4B-Instruct-2507-bf16
```

### Using Pocket TTS

Pocket TTS from Kyutai Labs provides streaming TTS with voice cloning capabilities. To use it:

```bash
speech-to-speech \
    --tts pocket \
    --pocket_tts_voice jean \
    --pocket_tts_device cpu
```

Available voice presets: `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`. You can also use custom voice files or HuggingFace paths.

## Command-line Usage

> **_NOTE:_** References for all the CLI arguments can be found directly in the [arguments classes](./src/speech_to_speech/arguments_classes) or by running `speech-to-speech -h`.

### Module level Parameters 
See [ModuleArguments](./src/speech_to_speech/arguments_classes/module_arguments.py) class. Allows to set:
- a common `--device` (if one wants each part to run on the same device)
- `--mode`: `realtime` (default), `local`, `socket`, or `websocket`
- chosen STT implementation (`--stt`)
- chosen LLM backend (`--llm_backend`: `transformers`, `mlx-lm`, or `responses-api`)
- chosen TTS implementation (`--tts`)
- logging level

### VAD parameters
See [VADHandlerArguments](./src/speech_to_speech/arguments_classes/vad_arguments.py) class. Notably:
- `--thresh`: Threshold value to trigger voice activity detection.
- `--min_speech_ms`: Minimum duration of detected voice activity to be considered speech.
- `--min_silence_ms`: Minimum length of silence intervals for segmenting speech, balancing sentence cutting and latency reduction.


### STT, LLM and TTS parameters

`model_name`, `torch_dtype`, and `device` are exposed for each implementation of the Speech to Text, Language Model, and Text to Speech. STT and TTS parameters use the handler prefix (e.g. `--stt_model_name`, `--llm_device`). LLM model selection and chat settings are shared across backends via unprefixed flags (e.g. `--model_name`, `--chat_size`); backend-specific flags use the `responses_api_` prefix for the `responses-api` backend and `llm_` prefix for local backends. See the [arguments classes](./src/speech_to_speech/arguments_classes) for the full list.

For example:
```bash
# Local transformers/mlx-lm backend
--model_name google/gemma-2b-it

# OpenAI-compatible backend
--llm_backend responses-api --model_name deepseek-chat --responses_api_base_url https://api.deepseek.com
```

### Generation parameters

Other generation parameters can be set using the handler prefix + `_gen_`, e.g., `--stt_gen_max_new_tokens 128` or `--llm_gen_temperature 0.7`. These parameters can be added to the pipeline part's arguments class if not already exposed.

## Citations

### Silero VAD
```bibtex
@misc{Silero VAD,
  author = {Silero Team},
  title = {Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/snakers4/silero-vad}},
  commit = {insert_some_commit_here},
  email = {hello@silero.ai}
}
```

### Distil-Whisper
```bibtex
@misc{gandhi2023distilwhisper,
      title={Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling},
      author={Sanchit Gandhi and Patrick von Platen and Alexander M. Rush},
      year={2023},
      eprint={2311.00430},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Parler-TTS
```bibtex
@misc{lacombe-etal-2024-parler-tts,
  author = {Yoach Lacombe and Vaibhav Srivastav and Sanchit Gandhi},
  title = {Parler-TTS},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/parler-tts}}
}
```
