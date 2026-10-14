# LiveKit Agents example

Run a [LiveKit Agents](https://docs.livekit.io/agents/) voice agent with
speech-to-speech as its realtime model. LiveKit's OpenAI Realtime plugin accepts
a custom `base_url`, so the agent talks to the local Realtime server instead of
OpenAI. The speech server needs no changes or extra flags for this.

```text
LiveKit room ──audio──> agent.py (LiveKit Agents) ──Realtime WebSocket──> speech-to-speech
                                                   <──audio + transcripts──  (VAD, STT, LLM, TTS)
```

## Supported setup

Apple Silicon macOS, Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/)
and, for the `dev` mode below, a local [LiveKit server](https://docs.livekit.io/home/self-hosting/local/)
(`brew install livekit`). The speech server uses the
[local defaults](../../README.md#quickstart): Parakeet TDT, Qwen3.5-4B
through MLX LM and Qwen3-TTS. The agent uses `livekit-agents` 1.8.5, pinned in
`requirements.txt`. Other versions are not tested by this example.

## Quickstart

1. In one terminal at the repository root, start the speech server:

   ```bash
   uv sync --python 3.12
   uv run speech-to-speech serve
   ```

   Wait for `Uvicorn running on http://127.0.0.1:8765`. First startup downloads
   the model weights. This terminal uses the repository's environment, which
   `uv run` selects automatically.

2. In a second terminal, start a local LiveKit server:

   ```bash
   livekit-server --dev --bind 127.0.0.1
   ```

3. In a third terminal, create and activate the agent's own environment, then
   install and start the agent.

   ```bash
   cd demo/livekit
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install -r requirements.txt
   LIVEKIT_URL=ws://127.0.0.1:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret \
     python agent.py dev
   ```

   The agent joins every new room on that LiveKit server. Join a room with any
   LiveKit client and a token for API key `devkey` and secret `secret`; the
   verification below used a scripted participant, not a specific client app.

To talk to the agent through your Mac's microphone and speakers without a LiveKit
server, use the [LiveKit CLI](https://docs.livekit.io/reference/developer-tools/livekit-cli/)
console in the agent's terminal from step 3, instead of running `livekit-server` and
`agent.py dev` (`python agent.py console` is deprecated):

```bash
lk agent console agent.py
```

Use headphones, otherwise the agent hears its own voice and interrupts itself.

`S2S_BASE_URL` (default `http://127.0.0.1:8765/v1`) overrides the server address.
`S2S_VOICE` sets the session voice, which Qwen3-TTS uses as its speaker; only the
default `Aiden` was tested. The plugin turns the base URL into
`ws://127.0.0.1:8765/v1/realtime?model=gpt-realtime`; the server ignores the model
name and the API key.

## Verification

Tested on 12 October 2026 on Apple Silicon (macOS 26.6) with speech-to-speech at
`14f9e33`, `livekit-agents` 1.8.5 and `livekit-server` 1.13.9: spoken round trip
and the `get_current_time` tool work by voice with `lk agent console` (LiveKit CLI
2.19.0) and with a scripted participant in a local LiveKit room. Interruption was
tested with the scripted participant.

## Known limitations

- **Answers after a tool call start about 5 seconds late.** The server
  acknowledges the `function_call_output` with `conversation.item.created`,
  but the plugin only handles the newer `conversation.item.added` event. It
  waits 5 seconds for that event, logs `failed to update chat context before
  generating the function calls results` with `update_chat_ctx timed out.`, and
  then sends `response.create`. The answer itself is correct.
- **The server owns turn detection and STT.** speech-to-speech always
  runs its own VAD; from `turn_detection` it reads only `threshold`,
  `silence_duration_ms`, `create_response` and `interrupt_response`. The
  transcription model and noise reduction settings are ignored. Only the server
  VAD settings in `agent.py` were tested; other turn detection modes and
  `modalities=["text"]` were not.
- **One session per pipeline.** The server starts one pipeline by default
  (`--num_pipelines`); a second simultaneous room is refused with
  `session_limit_reached` until the first session ends.
