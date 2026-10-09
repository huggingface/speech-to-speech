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
(`brew install livekit`). The speech server uses the Apple Silicon preset from the
[main README](../../README.md#apple-silicon-fully-local): Parakeet TDT, Qwen3-4B
through MLX LM and Qwen3-TTS. The agent uses `livekit-agents` 1.8.5, pinned in
`requirements.txt`. Other versions are not tested by this example.

## Quickstart

1. In one terminal at the repository root, start the speech server:

   ```bash
   uv sync --python 3.12
   uv run speech-to-speech serve \
     --mac-optimal-settings \
     --model_name mlx-community/Qwen3-4B-Instruct-2507-4bit
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

Tested on 9 October 2026 on Apple Silicon (macOS 26.6) with speech-to-speech at
`1e9bd0a`, `livekit-agents` 1.8.5 and `livekit-server` 1.13.9: spoken round trip
and interruption work; tool calls fail as described below.

## Known limitations

- **Function tools do not complete.** The server generates `call_id`s of 37
  characters (`call_` plus a UUID). The plugin shortens any `call_id` longer
  than 32 characters, the OpenAI limit, to a SHA-256 prefix before sending the
  `function_call_output`. The server finds no matching function call, rejects
  the output and then the follow-up `response.create` with
  `function_call_output_pending`, so the agent stays silent after running the
  tool. With the shortening disabled in a local diagnostic run, the same tool
  call completed. `agent.py` therefore defines no tools. Related:
  [#686](https://github.com/huggingface/speech-to-speech/issues/686). To
  reproduce, add this tool to the `Assistant` class (with
  `from datetime import datetime` and `function_tool` imported from
  `livekit.agents`) and ask "What time is it?":

  ```python
  @function_tool
  async def get_current_time(self) -> str:
      """Return the current local time."""
      return datetime.now().strftime("%H:%M")
  ```

- **The server owns turn detection and STT.** speech-to-speech always
  runs its own VAD; from `turn_detection` it reads only `threshold`,
  `silence_duration_ms`, `create_response` and `interrupt_response`. The
  transcription model and noise reduction settings are ignored. Only the server
  VAD settings in `agent.py` were tested; other turn detection modes and
  `modalities=["text"]` were not.
- **One session per pipeline.** The server starts one pipeline by default
  (`--num_pipelines`); a second simultaneous room is refused with
  `session_limit_reached` until the first session ends.
