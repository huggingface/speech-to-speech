"""Manual smoke test: out-of-band, text-only conversation against a local server.

Run a speech-to-speech server in realtime mode first, e.g.:

    speech-to-speech --mode realtime --ws_host 0.0.0.0 --ws_port 8765

Then run this script and type messages at the prompt:

    python test.py                 # defaults to localhost:8765, model "test"
    python test.py 127.0.0.1:8765 my-model

Each line you type is sent as a `response.create` with:
  - conversation="none"          -> out-of-band (never threaded into the chat)
  - output_modalities=["text"]   -> text-only (no audio)
  - input=[<your text>]          -> seeds the throwaway context for this turn

Every event the server sends back is logged in full. Type `quit`, `exit`, or
an empty line (or Ctrl-D / Ctrl-C) to stop.
"""

from __future__ import annotations

import asyncio
import uuid
import sys
import time

from openai import AsyncOpenAI

START = time.monotonic()


def _ts() -> str:
    return f"{time.monotonic() - START:7.3f}s"


def _log_event(event: object) -> None:
    etype = getattr(event, "type", "<unknown>")
    try:
        body = event.model_dump_json(indent=2, exclude_none=True)  # type: ignore[attr-defined]
    except Exception:
        body = repr(event)
    print(f"\n[{_ts()}] <= {etype}\n{body}", flush=True)


async def _reader(conn) -> None:
    """Log every event the server sends, for the life of the connection."""
    try:
        async for event in conn:
            _log_event(event)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # connection closed / transport error
        print(f"\n[{_ts()}] reader stopped: {exc!r}", flush=True)


async def main() -> None:
    hostport = sys.argv[1] if len(sys.argv) > 1 else "localhost:8765"
    model = sys.argv[2] if len(sys.argv) > 2 else "test"

    client = AsyncOpenAI(
        api_key="not-needed",
        base_url=f"http://{hostport}/v1",
        websocket_base_url=f"ws://{hostport}/v1",
    )

    print(f"Connecting to ws://{hostport}/v1 (model={model}) ...", flush=True)
    async with client.realtime.connect(model=model) as conn:
        reader_task = asyncio.create_task(_reader(conn))
        print("Connected. Type a message (out-of-band, text-only). 'quit' to exit.\n", flush=True)
        try:
            while True:
                try:
                    text = await asyncio.to_thread(input, "you> ")
                except (EOFError, KeyboardInterrupt):
                    break
                if text.strip().lower() in {"", "quit", "exit"}:
                    break

                event = {
                    "type": "response.create",
                    "response": {
                        # conversation: how the response relates to the default chat.
                        #   "auto" / absent -> IN-BAND: response is threaded into the
                        #       default conversation; `input` items are appended to
                        #       history and the assistant reply is committed back.
                        #   "none"          -> OUT-OF-BAND: generated against a
                        #       throwaway chat; nothing is committed to the default
                        #       conversation and response.done.conversation_id is null.
                        "conversation": "none",

                        # output_modalities: what the response emits.
                        #   ["audio"] / absent / [] -> AUDIO: TTS output, with
                        #       response.output_audio.* + audio-transcript events.
                        #   ["text"]                -> TEXT-ONLY: no TTS; emits
                        #       response.output_text.delta then one ...output_text.done
                        #       at close. The proxy backend runs the model non-streaming.
                        "output_modalities": ["text"],

                        # instructions: system prompt for this response. Falls back to
                        #   the session instructions when empty. Empty here AND no input
                        #   AND no context -> failed response with a clear error.
                        "instructions": "You are a helpful assistant. You are currently in a conversation with a user. You are to respond to the user's message.",

                        # input: the context for this response.
                        #   None / absent -> reuse existing context (in-band: the default
                        #       conversation; out-of-band: a read-only copy of it).
                        #   []            -> clear context: only `instructions` is used.
                        #   [items...]    -> fresh context seeded with these items
                        #       (out-of-band: default conversation is excluded;
                        #        in-band: items are also appended to history).
                        "input": [
                            {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": text}],
                            }
                        ],
                        "metadata": {
                            "request_id": str(uuid.uuid4()),
                        },
                    },
                }
                print(f"[{_ts()}] => response.create (conversation=none, text-only)", flush=True)
                await conn.send(event)
                # Give the server a beat to stream events before the next prompt.
                await asyncio.sleep(0.3)
        finally:
            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass

    print("\nClosed.", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
