"""Send real-time PCM, then barge in: no manual commits or response requests."""

import argparse
import asyncio
import base64
import json
import time
import wave

import websockets
from huggingface_hub import get_token


async def main(args):
    events = []
    first_audio = asyncio.Event()
    completed = asyncio.Event()
    models_ready = asyncio.Event()
    started = time.monotonic()
    async with websockets.connect(
        args.url, additional_headers={"Authorization": f"Bearer {get_token()}"}, max_size=2**24
    ) as ws:

        async def receive():
            async for raw in ws:
                event = json.loads(raw)
                if event["type"] == "speech_to_speech.models.ready":
                    models_ready.set()
                event["elapsed"] = round(time.monotonic() - started, 3)
                if event["type"] == "response.output_audio.delta":
                    if args.interrupt_on == "audio":
                        first_audio.set()
                    event["audio_bytes"] = len(base64.b64decode(event.pop("delta")))
                if event["type"] == "response.created" and args.interrupt_on == "response":
                    first_audio.set()
                events.append(event)
                if sum(e["type"] == "response.done" for e in events) >= 2:
                    completed.set()
                if "delta" not in event["type"]:
                    print(event, flush=True)

        async def send_audio(path, stop_padding=None):
            with wave.open(path) as wav:
                assert wav.getnframes() > 0
                assert wav.getframerate() == 16000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
                pcm = wav.readframes(wav.getnframes())
            speech_bytes = len(pcm)
            pcm += bytes(16000 * 2 * 3)
            for offset in range(0, len(pcm), 1024):
                if offset >= speech_bytes and stop_padding is not None and stop_padding.is_set():
                    break
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(pcm[offset : offset + 1024]).decode(),
                        }
                    )
                )
                await asyncio.sleep(0.032)

        receiver = asyncio.create_task(receive())
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "instructions": "You are a friendly assistant. Remember the user name. When asked for a story, tell a long story.",
                        "audio": {
                            "input": {
                                "transcription": {"language": "en"},
                                "turn_detection": {
                                    "type": "server_vad",
                                    "create_response": True,
                                    "interrupt_response": True,
                                },
                            }
                        },
                    },
                }
            )
        )
        if args.wait_models:
            await asyncio.wait_for(models_ready.wait(), timeout=190)
        await send_audio(args.first, first_audio)
        await asyncio.wait_for(first_audio.wait(), timeout=180)
        await send_audio(args.interrupt)
        await asyncio.wait_for(completed.wait(), timeout=90)
        receiver.cancel()
    with open(args.output, "w") as f:
        json.dump(events, f, indent=2)
    kinds = [e["type"] for e in events]
    assert kinds.count("input_audio_buffer.speech_started") >= 2, "Both speech starts must be detected"
    assert any(e["type"] == "response.done" and e.get("response", {}).get("status") == "cancelled" for e in events), (
        "Barge-in must cancel a response"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("first")
    parser.add_argument("interrupt")
    parser.add_argument("--output", default="/tmp/zerogpu-realtime-events.json")
    parser.add_argument("--interrupt-on", choices=["audio", "response"], default="audio")
    parser.add_argument("--wait-models", action="store_true")
    asyncio.run(main(parser.parse_args()))
