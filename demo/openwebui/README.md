# Open WebUI local example

Use an ordinary saved Open WebUI chat by typing or speaking, with typed context
updates available during the same Realtime call. This optional example complements
the [existing voice demo](../README.md); it does not change the speech server.

## Supported setup

Apple Silicon macOS, Docker Desktop with Compose v2, Python 3.12 and
[uv](https://docs.astral.sh/uv/getting-started/installation/). The host runs native
Parakeet TDT, a separate MLX Chat Completions endpoint serving
`mlx-community/Qwen3-4B-Instruct-2507-4bit`, and Kokoro (`bm_fable`). Allow memory
for the models, Docker and macOS together. First startup downloads model weights
and auxiliary assets; wait for both servers to finish loading.

Compose pins the official **Open WebUI 0.11.4-dev** image (commit
`538f9c9090dab742b4d8e3a1b3a058956ea8f39b`, digest
`sha256:06f24056e2fc15dacf23f7ee1f0a19cdf7cb0b8c61c5a6afb61c76ecaec8e15c`).
**This example requires that development build**, whose Realtime call path was
verified; do not substitute a moving `dev` tag or infer support from Standard
voice mode. Use a speech-to-speech checkout containing merged #687 and #689.
Other operating systems, model combinations and Open WebUI builds are not tested
by this example.

## Quickstart

1. From the repository root, install the host dependencies:

   ```bash
   uv sync --python 3.12
   ```

2. In one terminal at the repository root, start the text/model endpoint:

   ```bash
   uv run mlx_lm.server \
     --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
     --host 0.0.0.0 --port 8089 --max-tokens 256
   ```

3. In another terminal at the repository root, start the speech server using
   the same endpoint for voice-model inference:

   ```bash
   uv run speech-to-speech serve \
     --host 0.0.0.0 --port 8765 \
     --stt parakeet-tdt --parakeet_tdt_device mps \
     --llm_backend chat-completions \
     --model_name mlx-community/Qwen3-4B-Instruct-2507-4bit \
     --responses_api_base_url http://127.0.0.1:8089/v1 \
     --responses_api_api_key none --responses_api_stream \
     --tts kokoro --kokoro_device mps --kokoro_voice bm_fable \
     --enable_live_transcription
   ```

4. In a third terminal, start Open WebUI:

   ```bash
   cd demo/openwebui
   cp .env.example .env
   docker compose config --quiet
   docker compose up -d
   docker compose logs -f openwebui
   ```

   Open <http://localhost:3000> once startup completes. This is a local
   single-user example with Open WebUI login disabled, bound to host loopback.
   The two host servers bind to all interfaces so Docker can reach them and
   provide no authentication: run only on a trusted local network. Provider
   keys belong in the ignored `.env`, never in committed files.

5. Select **mlx-community/Qwen3-4B-Instruct-2507-4bit** in the model picker,
   open a new chat and type a message. Click **Voice mode** beside the chat
   input and allow microphone access. Compose enables Realtime calls
   automatically. Keep using `localhost` so the browser permits microphone
   capture.

The text-chat connection goes to port **8089**; the Realtime connection goes to
speech-to-speech on **8765**. These provider settings are supplied by Compose,
so no manual Admin provider configuration is needed. Inside Docker,
`host.docker.internal` means the Mac host; `localhost` means the container.
On the host, the speech server uses `127.0.0.1:8089`. The Realtime base URL must
end in `/v1`: Open WebUI appends `/realtime` and bridges the browser audio over
WebSocket. This example does not use the speech server's optional LLM proxy.

`ENABLE_PERSISTENT_CONFIG=False` keeps provider settings controlled by `.env`
on restart; Admin setting edits are not retained. Chat history still lives in
the named `openwebui-data` volume. After editing `.env`, run
`docker compose up -d` to recreate the container with the new values.

## Same-call smoke test

1. Type: **For this verification, the project code word is teal. Confirm in one
   short sentence.** Wait for the text reply, then start Voice mode.
2. Ask: **What is the project code word in the current chat?** Expect one
   completed spoken reply identifying **teal**, then **Listening**.
3. Leave the call open. Type: **Update the project code word to amber. Confirm
   in one short sentence.** Wait for the text reply.
4. Ask the same question again. Expect one completed spoken reply identifying
   **amber**, then **Listening**, with no provider error or disconnection.
5. Click **End call**. Reload the page and confirm the chat is still saved.

The supplied `REALTIME_CALL_PROMPT_TEMPLATE` makes the voice model answer facts
already present in the current snapshot directly. This keeps the smoke focused
on context refresh; tool delegation and other workflows are not acceptance
criteria. Do not use physical microphone/speaker quality or reported timing
metrics as evidence that snapshot refresh works. Headphones can help prevent
speaker output from being captured as another question.

## Verification

Independently tested on 9 October 2026 with this Compose file, Python 3.12,
`mlx-lm` 0.31.3 and `mlx-audio` 0.4.7 on Apple Silicon. Chromium fed controlled
synthesized microphone speech through the actual UI/AudioWorklet, with real
Parakeet STT, MLX inference and Kokoro TTS. The direct connection produced two
explicit requests and exactly two completed replies (teal, then amber), each
rendered once, with no provider/bridge errors. Seven snapshots received six
matching deletion acknowledgements. Speech-server debug request records
confirmed instructions and the updated snapshot in the next voice-model request;
no diagnostic proxy was used. The saved chat survived reload and Compose restart.
This establishes the smoke flow, not physical microphone/speaker quality,
performance, tool delegation or full Realtime API parity.

## Stop and restart

From `demo/openwebui`:

```bash
docker compose down       # Stops Open WebUI; keeps saved chats.
docker compose up -d      # Restart after starting both host servers again.
```

Stop each host server with Ctrl-C in its terminal. `docker compose down --volumes`
**deletes this example's saved chats**; use it only for an intentional reset.
The existing `demo/` startup and storage are independent.

## Troubleshooting and references

- Missing model or failed text reply: check that the MLX server is ready and
  `.env` points to its `/v1` endpoint, not the speech endpoint.
- Call uses Standard mode: confirm `AUDIO_REALTIME_ENABLED=True` in the
  container configuration and restart with `docker compose up -d`.
- Rejected provider request: check the pinned image, speech checkout and
  Realtime `/v1` address; inspect `docker compose logs openwebui` and the host
  speech logs. Keep voice `bm_fable` paired with Kokoro.
- Changing container provider settings does not change the speech server's
  configured model or upstream address; keep both startup commands and `.env`
  matched when adapting the example.

Upstream [provider connection documentation](https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-openai-compatible/)
explains text endpoints and Docker host addresses. [Voice Mode](https://docs.openwebui.com/features/chat-conversations/chat-features/voice-mode/)
describes the UI; the pinned development source is authoritative for this
Realtime setup. The prior real-model verification is recorded in
[#687](https://github.com/huggingface/speech-to-speech/pull/687).
Existing limitations in [#686](https://github.com/huggingface/speech-to-speech/issues/686)
(function-call IDs) and [#485](https://github.com/huggingface/speech-to-speech/issues/485)
(ambiguous transcription terminals) remain outside this example.
