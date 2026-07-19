# Using the s2mlt WebSocket API

`s2mlt` accepts live speech as raw PCM audio over a WebSocket and returns JSON
events containing live transcription and translations into exactly two target
languages. It does not return synthesized audio.

## Start the s2mlt service

Install the project from source first:

```bash
git clone https://github.com/huggingface/speech-to-speech.git
cd speech-to-speech
uv sync
```

The default LLM backend is `chat-completions`. It requires an OpenAI-compatible
Chat Completions server that supports `POST /v1/chat/completions`. The LLM
server must already be reachable when s2mlt starts because startup includes a
warm-up request.

For example, with a model served locally by vLLM at `http://127.0.0.1:8000/v1`:

```bash
uv run s2mlt \
  --llm_backend chat-completions \
  --llm_base_url http://127.0.0.1:8000/v1 \
  --llm_api_key not-needed \
  --llm_model_name Qwen/Qwen3-4B-Instruct-2507 \
  --target_languages de fr \
  --ws_host 0.0.0.0 \
  --ws_port 8765
```

`--llm_model_name` must match the model name advertised by the Chat
Completions server. Some local servers accept any non-empty API key; others
require their configured key. For the official OpenAI endpoint, use
`--llm_base_url https://api.openai.com/v1` and supply a valid key through
`--llm_api_key` or `OPENAI_API_KEY`.

To load the LLM directly with Transformers instead:

```bash
uv run s2mlt \
  --llm_backend transformers \
  --llm_model_name Qwen/Qwen3-4B-Instruct-2507 \
  --llm_device cuda \
  --llm_torch_dtype float16 \
  --target_languages de fr \
  --ws_host 0.0.0.0 \
  --ws_port 8765
```

For a CPU-only deployment, use `--llm_device cpu --llm_torch_dtype float32`.
This is functional but is generally too slow for low-latency translation.

The equivalent direct module command is:

```bash
uv run python -m speech_to_speech.s2mlt [arguments...]
```

### Required and important arguments

| Argument | Meaning |
| --- | --- |
| `--target_languages de fr` | Exactly two distinct output language codes. Their order is also the order requested from the LLM, so put the most latency-sensitive language first. `corrected` is reserved and cannot be a language code. |
| `--llm_backend` | `chat-completions` or `transformers`. Defaults to `chat-completions`. |
| `--llm_model_name` | Served model name for Chat Completions, or Hugging Face model ID for Transformers. |
| `--llm_base_url` | Required in practice for a local or third-party Chat Completions server; normally ends in `/v1`. |
| `--llm_api_key` | API key for the Chat Completions server. |
| `--ws_host` / `--ws_port` | WebSocket listener. Defaults to `0.0.0.0:8765`. |
| `--stt_model_name` | Whisper model ID. Defaults to `openai/whisper-large-v3-turbo`. |
| `--stt_device` | `auto`, `cuda`, `mps`, or `cpu`. Defaults to `auto`. |
| `--stt_torch_dtype` | `float16` or `float32`. CPU inference is automatically changed to `float32`. |
| `--stt_language` | Leave unset or use `auto` for language auto-detection and code-switching. Set a language code only when all input must be forced to one language. |
| `--realtime_processing_pause` | Target interval between progressive STT updates. Defaults to 0.5 seconds and increases automatically for long turns. |
| `--speculative_reopen_ms` | Window in which resumed speech can reopen a recently stopped turn. Defaults to 1000 ms. |
| `--llm_request_timeout_s` | Chat Completions request timeout. Defaults to 20 seconds. |
| `--log_level` | Common values are `debug`, `info`, and `warning`. |

Keep `--sample_rate` at its default of `16000`. The WebSocket input and Whisper
handler currently use 16 kHz audio internally.

Run `uv run s2mlt --help` for the complete VAD and model argument list.

### JSON configuration

Instead of CLI flags, s2mlt accepts one JSON configuration file. A practical
configuration is:

```json
{
  "llm_backend": "chat-completions",
  "llm_base_url": "http://127.0.0.1:8000/v1",
  "llm_api_key": "not-needed",
  "llm_model_name": "Qwen/Qwen3-4B-Instruct-2507",
  "llm_request_timeout_s": 30.0,
  "target_languages": ["de", "fr"],
  "ws_host": "0.0.0.0",
  "ws_port": 8765,
  "stt_model_name": "openai/whisper-large-v3-turbo",
  "stt_device": "auto",
  "stt_torch_dtype": "float16",
  "stt_language": null,
  "realtime_processing_pause": 0.5,
  "speculative_reopen_ms": 1000,
  "log_level": "info"
}
```

Start it with:

```bash
uv run s2mlt ./s2mlt.production.json
```

## WebSocket wire protocol

Connect to `ws://<s2mlt-host>:8765`. There is no handshake message and no
required URL path.

This is a small raw WebSocket protocol, not the OpenAI Realtime protocol:

- Client to s2mlt: binary WebSocket messages containing raw audio.
- s2mlt to client: text WebSocket messages containing JSON events.
- Do not base64-encode audio.
- Do not wrap audio in `input_audio_buffer.append` or another JSON envelope.
- s2mlt has no explicit `commit` or end-of-turn request. VAD detects the turn
  boundary from trailing silence in the audio stream.

### Input audio format

Audio must be:

- 16,000 Hz
- mono
- signed 16-bit PCM
- little-endian
- headerless (do not send a WAV/MP3/Opus container)

Send binary frames continuously and at approximately real-time pace, including
silence. A 20-100 ms frame cadence works well. For example, a 40 ms frame is
640 samples or 1,280 bytes. WebSocket frame boundaries do not need to align
with the internal VAD chunks; s2mlt buffers remainders and processes 512-sample
(1,024-byte) chunks.

Do not send only voice-active frames unless your backend also appends enough
silence afterward. Without trailing silence, VAD cannot close the segment and
`input.transcription.done` or translation events may never arrive.

### Output events

All server output is JSON text. Every segment is identified by `turn_id` and
`turn_revision`.

Speech boundary events:

```json
{
  "type": "speech_started",
  "audio_start_ms": 1240,
  "turn_id": "turn_abc123",
  "turn_revision": 0,
  "reopened": false
}
```

```json
{
  "type": "speech_stopped",
  "duration_s": 2.48,
  "audio_end_ms": 3720,
  "turn_id": "turn_abc123",
  "turn_revision": 0
}
```

Live transcription is a full snapshot, despite the `delta` name:

```json
{
  "type": "input.transcription.delta",
  "text": "Good morning every",
  "turn_id": "turn_abc123",
  "turn_revision": 0
}
```

Replace the previous text for that turn with the new `text`; do not append it.
The final transcription event is:

```json
{
  "type": "input.transcription.done",
  "text": "Good morning, everyone.",
  "language_code": "en",
  "turn_id": "turn_abc123",
  "turn_revision": 0
}
```

Translation snapshots arrive as the LLM generates its JSON response. Early
snapshots may contain only the first target language, and `corrected` may still
be empty:

```json
{
  "type": "translation.delta",
  "corrected": "",
  "translations": {
    "de": "Guten Morgen"
  },
  "turn_id": "turn_abc123",
  "turn_revision": 0
}
```

Each `translation.delta` is also a full snapshot of all fields parsed so far.
Replace the prior translation state for the same turn and revision rather than
appending strings.

A successful final event looks like:

```json
{
  "type": "translation.done",
  "corrected": "Good morning, everyone.",
  "translations": {
    "de": "Guten Morgen zusammen.",
    "fr": "Bonjour à toutes et à tous."
  },
  "turn_id": "turn_abc123",
  "turn_revision": 0,
  "error": null
}
```

Generation or JSON parsing failures still close the segment:

```json
{
  "type": "translation.done",
  "corrected": "",
  "translations": {},
  "turn_id": "turn_abc123",
  "turn_revision": 0,
  "error": "Model output could not be parsed as JSON"
}
```

Treat `translation.done` as terminal for that particular `(turn_id,
turn_revision)` pair, but allow a later revision of the same turn to replace
it.

## Turn revisions and backend state

When speech resumes shortly after VAD stopped, s2mlt can merge it into the same
logical turn. The new events retain `turn_id` and increment `turn_revision`.
Backend state should therefore be keyed by `turn_id` and follow these rules:

1. Ignore an event whose revision is lower than the stored revision.
2. When a higher revision arrives, replace/reset the stored data for that turn.
3. Within the same revision, replace transcription and translation snapshots;
   do not concatenate them.
4. Mark the revision complete on `translation.done`.

Translations are stateless per final segment. A new segment can be captured
while an earlier segment is translating, but LLM requests are processed by the
pipeline in queue order.

## Python backend client example

The following single-utterance adapter accepts an asynchronous stream of
correctly encoded PCM frames, forwards them to s2mlt, and maintains an
up-to-date segment map. In an HTTP backend, `audio_frames` could come from the
browser through another WebSocket. Ensure the iterable ends with PCM silence.

```python
import asyncio
import json
from collections.abc import AsyncIterable
from typing import Any

from websockets.asyncio.client import connect


class S2MLTClient:
    def __init__(self, url: str = "ws://127.0.0.1:8765") -> None:
        self.url = url
        self.segments: dict[str, dict[str, Any]] = {}

    def apply_event(self, event: dict[str, Any]) -> None:
        turn_id = event.get("turn_id")
        revision = event.get("turn_revision")
        if turn_id is None or revision is None:
            return

        current = self.segments.get(turn_id)
        if current is not None and revision < current["turn_revision"]:
            return
        if current is None or revision > current["turn_revision"]:
            current = {
                "turn_id": turn_id,
                "turn_revision": revision,
                "transcript": "",
                "language_code": None,
                "corrected": "",
                "translations": {},
                "done": False,
                "error": None,
            }
            self.segments[turn_id] = current

        event_type = event["type"]
        if event_type in {"input.transcription.delta", "input.transcription.done"}:
            current["transcript"] = event["text"]
            if event_type == "input.transcription.done":
                current["language_code"] = event.get("language_code")
        elif event_type in {"translation.delta", "translation.done"}:
            current["corrected"] = event.get("corrected", "")
            current["translations"] = event.get("translations", {})
            if event_type == "translation.done":
                current["done"] = True
                current["error"] = event.get("error")

    async def run(self, audio_frames: AsyncIterable[bytes]) -> None:
        async with connect(self.url, max_size=None) as websocket:
            translation_done = asyncio.Event()

            async def send_audio() -> None:
                async for pcm16_frame in audio_frames:
                    await websocket.send(pcm16_frame)

            async def receive_events() -> None:
                async for message in websocket:
                    if not isinstance(message, str):
                        continue
                    event = json.loads(message)
                    self.apply_event(event)
                    # Publish to your application, database, or downstream WS here.
                    print(json.dumps(event, ensure_ascii=False))
                    if event.get("type") == "translation.done":
                        translation_done.set()

            sender = asyncio.create_task(send_audio())
            receiver = asyncio.create_task(receive_events())
            try:
                await sender
                # Keep receiving after the last speech frame until the final
                # translation arrives, bounded by an application timeout.
                await asyncio.wait_for(translation_done.wait(), timeout=60)
            finally:
                sender.cancel()
                receiver.cancel()
                await asyncio.gather(sender, receiver, return_exceptions=True)
```

In a real service, keep receiving until the expected `translation.done`, then
close the WebSocket or continue streaming the next segment. If the source audio
ends with speech, append roughly 0.5-1 second of PCM silence before waiting for
the final event.

## Deployment notes

- The current WebSocket streamer owns one shared pipeline. Audio received from
  every connected client enters the same input queue, and output events are
  broadcast to every connected client. Use one active client per s2mlt process,
  or add a session router/process pool before using it as a multi-tenant API.
- The raw WebSocket listener has no authentication or TLS. Keep it on a private
  network or place it behind an authenticated TLS reverse proxy (`wss://`).
- Apply WebSocket connection limits, message-size limits, idle timeouts, and
  backpressure in the service or proxy in front of s2mlt.
- When the last client disconnects, s2mlt sends a session-reset signal through
  the pipeline. The first connection of the next session drains stale outbound
  events before listening resumes.
- Handle disconnects by reconnecting with bounded exponential backoff. Do not
  assume an interrupted turn will receive a final translation after its socket
  has closed.
- Preserve the event's `turn_id` and `turn_revision` when forwarding it to
  browsers or other consumers; they are required for correct upserts.
