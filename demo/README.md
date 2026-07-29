---
title: HF Realtime Voice
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Voice chat over WebSocket against a HF speech-to-speech
hf_oauth: true
---

# Realtime Voice Demo

Browser voice-chat UI for the
[huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech)
backend, speaking the OpenAI Realtime **GA** protocol over **WebSocket**
(default) or **WebRTC** (Settings → Transport, env-pinned deploys only — see
[WebRTC transport](#webrtc-transport)).

## Quick start (local)

1. **Start the speech-to-speech backend** in realtime mode (from the repo root;
   see the [backend README](https://github.com/huggingface/speech-to-speech/blob/main/src/speech_to_speech/api/openai_realtime/README.md)
   for more model combinations):

   ```bash
   uv run speech-to-speech \
     --mode realtime \
     --stt parakeet-tdt \
     --llm_backend transformers \
     --tts kokoro \
     --model_name "Qwen/Qwen3-4B-Instruct-2507" \
     --llm_device mps \
     --llm_torch_dtype float16 \
     --enable_live_transcription
   ```

   The realtime server listens on `ws://localhost:8765/v1/realtime` by default
   (`--ws_host` / `--ws_port` to change).

2. **Start this app**, pointing it at the backend with `SPEECH_TO_SPEECH_URL`:

   ```bash
   uv pip install -r demo/requirements.txt
   export SPEECH_TO_SPEECH_URL=ws://localhost:8765/v1/realtime
   export SERPER_API_KEY=...   # optional; web search is disabled without it
   uv run uvicorn --app-dir demo server:app --reload --port 7860
   ```

   Or with Docker:

   ```bash
   docker build -t s2s-demo demo/
   docker run -p 7860:7860 -e SPEECH_TO_SPEECH_URL=ws://host.docker.internal:8765/v1/realtime s2s-demo
   ```

   > **Docker + host backend: WebSocket and WebRTC need different hostnames.**
   > The two transports dial the backend from different network namespaces:
   >
   > - **WebRTC** is dialed **server-side** — the browser POSTs its SDP offer to
   >   the demo's `/api/calls` proxy, which forwards it from *inside the
   >   container*. There `host.docker.internal` resolves to your host, so the
   >   command above works.
   > - **WebSocket** is dialed **client-side** — the demo hands the URL straight
   >   to the browser, which opens the socket itself. The browser runs on your
   >   *host*, where `host.docker.internal` is not a real DNS name, so the
   >   connection never reaches the backend and the server logs nothing.
   >
   > A single Docker `SPEECH_TO_SPEECH_URL` can therefore only make one transport
   > work at a time (`localhost:8765` for WebSocket, `host.docker.internal:8765`
   > for WebRTC). To exercise **both** without swapping the env, run the demo
   > **without Docker** (the `uvicorn` command above) so host and container
   > namespaces collapse — then `ws://localhost:8765/v1/realtime` works for both.

3. Open <http://localhost:7860/>, click the orb, allow the mic, talk.

> Browsers require **HTTPS or `localhost`** for `getUserMedia()` (mic + camera).
> `127.0.0.1` and `localhost` both work; plain `http://192.168.x.y` does NOT.

Smoke-test the backend from the shell:

```bash
websocat ws://localhost:8765/v1/realtime
# -> you should get a session.created event back immediately
```

## How it works

1. The browser opens a WebSocket on the configured `/v1/realtime` URL.
2. Server pushes `session.created` on connect. Client replies with
   `session.update` (OpenAI Realtime **GA** schema: `session.audio.input`,
   `session.audio.output`, `session.output_modalities`).
3. Client streams mic audio as PCM16 16 kHz mono base64 chunks
   (`input_audio_buffer.append`, one frame every ~40 ms).
4. Server pushes `response.output_audio.delta` (PCM16 24 kHz mono base64)
   and transcript deltas.

The backend exposes one concurrent session per pipeline unit
(`--num_pipelines` to serve more).

## WebRTC transport

With `SPEECH_TO_SPEECH_URL` set, **Settings → Transport** offers WebRTC as an
alternative to the WebSocket. Same conversation, different plumbing:

1. The browser adds its mic track and an `oai-events` data channel to an
   `RTCPeerConnection` and POSTs the SDP offer to the same-origin
   `/api/calls` proxy, which forwards it to the backend's
   `POST /v1/realtime/calls` (the OpenAI GA handshake). The proxy exists
   because the s2s server has no CORS middleware — and it forwards **only**
   to the env-pinned URL, never to a client-supplied one, so it can't be used
   as an open proxy. That's why the toggle is locked to WebSocket when the
   URL isn't pinned (user-typed URLs, LB mode).
2. Only the handshake goes through the proxy: the negotiated audio (Opus RTP
   both ways) and the data channel flow directly browser ↔ backend.
3. JSON events on the data channel are the same GA protocol as the WebSocket,
   minus the audio: mic audio rides the media track (never
   `input_audio_buffer.append`, which the backend rejects over WebRTC), and
   the assistant's voice arrives as a remote audio track (never
   `response.output_audio.delta`). Barge-in flushing is server-side.

Backend requirement: the `webrtc` extra
(`pip install "speech-to-speech[webrtc]"`), otherwise `/v1/realtime/calls`
answers 501 and the handshake fails with a clear message.

Caveats vs. WebSocket:

- **NAT**: host ICE candidates only by default — fine when browser and backend
  are on the same machine/LAN. Across the internet, set `RTC_ICE_SERVERS` on
  *this* app (a JSON list of `RTCIceServer` dicts, or comma-separated
  STUN/TURN URLs; served to the browser via `/api/config`) and
  `SPEECH_TO_SPEECH_ICE_SERVERS` on the backend. There is no TURN relay
  fallback, so symmetric-NAT setups may still not connect.
- **Noise gate**: implemented in the WebSocket capture worklet, so it's
  hidden on WebRTC — the raw mic track (with the browser's own
  `noiseSuppression`) is sent instead.
- **Camera snapshots** are re-encoded to fit one data-channel message
  (~60 KB), so the model may see a smaller frame than over WebSocket.
- **Load-balancer mode is WebSocket-only** for now.

## Connecting to a backend

Three modes, picked by env (`/api/config` tells the client which one is active):

- **`SPEECH_TO_SPEECH_URL` env** — the mode you want for local use, and the
  highest priority. The browser connects **directly** to this realtime
  WebSocket URL; it's shown read-only in Settings. Setting it disables the
  load-balancer logic entirely (no `/api/session` proxy, no queue, no
  metering, no sign-in). Unlike the LB address it is not a secret. Accepts a
  full `ws(s)://host/v1/realtime` URL or a bare host like `localhost:8765`
  (the app adds `/v1/realtime`).
- **Neither env set** — **Settings → Speech-to-speech server URL**: paste a
  full connect URL or a bare host, and the browser connects to it directly.
- **`LOAD_BALANCER_URL` env** — multi-compute deployments only: the browser
  POSTs the same-origin `/api/session` proxy, the server forwards to the LB,
  and the browser dials the per-session compute URL the LB hands back. The LB
  address never reaches the browser; the Settings URL field is hidden.

| `SPEECH_TO_SPEECH_URL` | `LOAD_BALANCER_URL` | `SPACE_ID` | Connection | URL field | Transport | Metering |
|:---:|:---:|:---:|---|---|---|---|
| ✅ | any | any | direct → pinned URL | visible, locked | WS or WebRTC | off |
| – | – | any | direct → user URL | editable | WS only | off |
| – | ✅ | ✅ | LB proxy | hidden | WS only | **on** |
| – | ✅ | – | LB proxy | hidden | WS only | off |

**Settings → Restart** reconnects with the current voice, instructions and URL.

## Tools

The assistant can call two tools mid-conversation (toggle them from the **Tools**
button, top-right):

- **Web search** — Google results via Serper.dev, proxied server-side so the key
  never reaches the browser. Set `SERPER_API_KEY` as an env var / Space secret.
  Without it, the tool is disabled unless the user pastes their own key in the
  Tools panel.
- **Camera** — while enabled, a live self-view shows bottom-left; when the model
  calls the tool, the current frame is sent to the vision-language model so it can
  see what you're showing it.

## Usage limits (deployed Space only)

Conversation time is metered per UTC day by sign-in tier (see `limiter.py` /
`auth.py`), but **only on the deployed Space** — metering turns on only when BOTH
`LOAD_BALANCER_URL` and `SPACE_ID` (injected automatically by the HF Space
runtime) are present. Running locally — even with `LOAD_BALANCER_URL` exported —
leaves the app unmetered. Tunable via env:

| Env | Default | What |
|-----|---------|------|
| `LIMIT_ANON_SEC` | `300` | Daily seconds for anonymous visitors (5 min) |
| `LIMIT_FREE_SEC` | `600` | Daily seconds for signed-in non-PRO users (10 min) |
| `UNLIMITED_ORGS` | _(adds to defaults)_ | Extra HF org names whose members get **unlimited** usage, like PRO |
| `USAGE_HASH_SECRET` | _(random)_ | HMAC secret for hashing identity keys + signing the anon cookie |

PRO members are always unlimited. Members of `cerebras`, `HuggingFaceM4`,
`smolagents`, and `pollen-robotics` are unlimited out of the box (shown as
"Team", not "PRO"); set `UNLIMITED_ORGS=my-team` to add more. Matched
case-insensitively against the user's organisations from HF OAuth.

## Settings (stored in `localStorage`)

| Key | What |
|-----|------|
| Speech-to-speech server URL | Direct realtime WebSocket URL (hidden/locked when pinned by env) |
| Transport | WebSocket (default) or WebRTC; selectable only with an env-pinned URL |
| Voice | Qwen3-TTS speaker name (Aiden, Ryan, Dylan, Eric, Ono_Anna, Serena, Sohee, Uncle_Fu, Vivian) |
| Instructions | System prompt sent in `session.update` once the connection opens |

LocalStorage keys are namespaced `s2s.ws.*` (plus `s2s.transport` for the
transport pick).

## Files

| File | Role |
|------|------|
| `index.html` | Single page, orb + settings modal (identical UI to the WebRTC app) |
| `main.js` | State machine, settings, tools, camera, noise-gate UI wiring |
| `ui/chat.js` | `ChatView`: history panel, ephemeral bubbles, transcript/tool streaming |
| `ui/account.js` | `Account`: HF login chip + popover, daily-limit modal |
| `ui/dom.js` | Shared helpers: `$`, `escHtml`, `truncateError`, `DEBUG` |
| `auth.py` | HF OAuth + per-request identity (tier, hashed keys) |
| `limiter.py` | SQLite per-day talk-time budget (chunked server-clock reservation) |
| `ws/s2s-ws-client.js` | WebSocket handshake + OpenAI Realtime GA protocol |
| `rtc/s2s-rtc-client.js` | WebRTC sibling: SDP handshake via `/api/calls`, events over the data channel, track audio |
| `ws/codec.js` | base64 <-> PCM helpers + transcript extraction (pure) |
| `ws/orb-visualizer.js` | `OrbVisualiser`: FFT bands -> orb CSS custom properties |
| `worklets/mic-capture.js` | AudioWorklet: 48 kHz Float32 -> 16 kHz Int16 PCM, posts ~40 ms chunks |
| `worklets/audio-playback.js` | AudioWorklet: 24 kHz Float32 ring buffer -> 48 kHz, linear interp, fade in/out |
| `style.css` | Orb animations, layout, dark theme (verbatim from the WebRTC app) |

## Audio pipeline notes

- **Input**: `getUserMedia({ echoCancellation, noiseSuppression, autoGainControl })`
  feeds the `mic-capture` worklet at the `AudioContext` rate. The worklet
  resamples to 16 kHz (boxcar lowpass + decimation on the 48 -> 16 fast
  path, linear interpolation fallback for odd rates) and packs Int16 LE.
- **Output**: `response.output_audio.delta` decodes to Int16 -> Float32
  and is posted to the `audio-playback` worklet. The worklet maintains a
  per-context ring buffer, linearly interpolates 24 -> 48, and applies
  short 32-frame fades on entry/exit to suppress clicks.
- **Barge-in**: when the server VAD detects user speech mid-response
  (`input_audio_buffer.speech_started` while `ai-speaking`), the client
  posts `{ kind: "clear" }` to the playback worklet to wipe the queue
  immediately. The server itself cancels the in-flight response.

## Credits

- Backend: [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech)
- UI verbatim from `amir-tfrere/minimal-conversation-app-s2s-backend` (Pollen Robotics × Hugging Face)
