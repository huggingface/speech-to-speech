# Gemma 4 12B speech-to-speech on Apple Silicon

This example runs the complete realtime voice loop locally on a Mac:

```text
microphone -> browser demo -> Realtime server -> VAD -> Gemma 4 12B (native audio)
speakers   <- audio deltas <-                  Qwen3-TTS <- assistant response
```

There is no separate speech-to-text model. Each completed VAD segment is sent
to Gemma through llama.cpp's OpenAI-compatible Chat Completions endpoint. The
speech-to-speech server and microphone client communicate through the OpenAI
Realtime-compatible WebSocket API, so turn revisions, cancellation, and
barge-in use the same path as other Realtime clients.

## Requirements

- An Apple Silicon Mac. The Q4 model is about 7.2 GB; 24 GB or more of unified
  memory is recommended for comfortable headroom alongside Qwen3-TTS.
- A recent llama.cpp build with Gemma 4 Unified multimodal support.
- A source checkout that includes PR #298.
- `uv`, Homebrew, and a modern browser.

Install or update llama.cpp:

```bash
brew install llama.cpp
brew upgrade llama.cpp
llama-server --version
```

An older build may load the language model but fail on the multimodal projector
with `unknown projector type: gemma4uv`. Upgrade llama.cpp if that happens.

Install the speech-to-speech environment from the repository root:

```bash
uv sync
```

The first run downloads the Gemma GGUF, its multimodal projector, and the local
MLX Qwen3-TTS model.

## Terminal 1: serve Gemma with llama.cpp

```bash
llama-server \
    -hf ggml-org/gemma-4-12B-it-GGUF:Q4_0 \
    -c 16384 \
    -np 1 \
    -fa on \
    --host 127.0.0.1 \
    --port 8080
```

`-hf` downloads and loads both the Q4 model and its multimodal projector. Wait
until the server prints both of these messages before starting the pipeline:

```text
loaded multimodal model
listening on http://127.0.0.1:8080
```

The warning that audio input is experimental comes from llama.cpp and is
expected.

## Terminal 2: start the realtime speech pipeline

```bash
uv run speech-to-speech \
    --mode realtime \
    --stt none \
    --llm_backend chat-completions \
    --tts qwen3 \
    --model_name "ggml-org/gemma-4-12B-it-GGUF" \
    --responses_api_base_url "http://127.0.0.1:8080/v1" \
    --responses_api_api_key "" \
    --responses_api_audio_content_type input_audio \
    --responses_api_stream \
    --qwen3_tts_mlx_quantization 6bit \
    --min_silence_ms 300
```

Wait until the server is listening at `ws://127.0.0.1:8765/v1/realtime`.

## Terminal 3: start the browser demo

```bash
SPEECH_TO_SPEECH_URL="ws://localhost:8765/v1/realtime" \
STARTUP_GREETING="" \
uv run --with-requirements demo/requirements.txt \
    uvicorn --app-dir demo server:app --port 7860
```

Open <http://localhost:7860>, click the orb, and allow microphone access. Speak
normally and pause to end the turn. Use headphones while testing barge-in so
TTS playback does not feed back into the microphone. Press `Ctrl-C` in the demo
and server terminals to stop.

The demo defaults to WebSocket, which is the path used by this example. It also
shows WebRTC in **Settings -> Transport** because the backend URL is pinned. To
try WebRTC, first install its backend dependencies with
`uv sync --extra webrtc`. Otherwise, leave the transport set to WebSocket.

The browser UI shows conversation history, user and assistant turn state,
replayable user audio, voice and instruction settings, and barge-in. Setting
`STARTUP_GREETING` to an empty value keeps this native-audio test focused on the
first spoken turn.

The important options are:

- `--stt none`: bypasses STT and forwards the captured WAV directly to Gemma.
- `--llm_backend chat-completions`: uses llama.cpp's `/v1/chat/completions`
  endpoint, which accepts native audio.
- `--responses_api_audio_content_type input_audio`: uses the payload shape
  supported by current llama.cpp builds.
- `--mode realtime`: enables speculative turn revisions, response cancellation,
  and the standard Realtime event lifecycle.
- `--min_silence_ms 300`: avoids treating very short pauses between words as
  completed turns while keeping endpointing responsive.

## Headless client alternative

To test without the browser demo, replace Terminal 3 with the packaged
microphone/speaker client:

```bash
uv run python scripts/listen_and_play_realtime.py \
    --host 127.0.0.1 \
    --port 8765
```

## Troubleshooting

- **`unknown projector type: gemma4uv`**: upgrade llama.cpp, then verify
  `llama-server --version` changed.
- **`unsupported content[].type`**: keep
  `--responses_api_audio_content_type input_audio`; `audio_url` is not accepted
  by the tested llama.cpp endpoint.
- **No microphone input**: allow microphone access in the browser. If access
  was previously denied, enable it for the browser in **System Settings ->
  Privacy & Security -> Microphone**, then reload the page.
- **The UI cannot connect**: verify the speech-to-speech server is listening on
  port `8765` and the demo was started with the exact
  `SPEECH_TO_SPEECH_URL` above.
- **WebRTC returns 501**: keep the demo on its default WebSocket transport, or
  install the `webrtc` extra before starting the backend.
- **The assistant hears itself**: use headphones. Do not pass
  `--block_mic_during_playback` when testing barge-in, because that option
  intentionally pauses microphone capture in the headless client during
  playback.
- **Turns end too early**: raise `--min_silence_ms` to `500` or `700`. Higher
  values add the same amount of endpointing latency after the user stops.
- **Memory pressure**: stop other local models. Keep `-np 1`, and use
  `--qwen3_tts_mlx_quantization 4bit` if the 6-bit TTS model leaves too little
  headroom.
